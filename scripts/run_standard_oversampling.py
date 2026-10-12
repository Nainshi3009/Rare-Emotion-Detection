
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import psutil
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preparation import (
    create_standard_oversampling,
    get_target_training_counts,
    validate_label_ids,
    check_split_integrity,
)
from src.dataset import GoEmotionsVADDataset
from src.model import RobertaEmotionVADModel
from src.training import (
    TrainingConfig,
    evaluate_model,
    load_checkpoint,
    load_tokenizer,
    train_model,
)

CONFIG_PATH = PROJECT_ROOT / "configs" / "gold_only_baseline.json"
RESOURCE_CONFIG_PATH = PROJECT_ROOT / "configs" / "model_resource_efficient.json"
FORMULATION_CONFIG_PATH = PROJECT_ROOT / "configs" / "model_training_formulation.json"
PREPARED_DIRECTORY = PROJECT_ROOT / "data" / "processed" / "goemotions_vad_v1"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "experiments" / "B_standard_oversampling"

PROTECTED_PATHS = (
    Path("Rare_Emotion_Detection_EDA.ipynb"),
    Path("data/processed/goemotions_vad_v1/train.parquet"),
    Path("data/processed/goemotions_vad_v1/validation.parquet"),
    Path("data/processed/goemotions_vad_v1/test.parquet"),
    Path("data/processed/goemotions_vad_v1/metadata.json"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protected_hashes() -> dict[str, str]:
    return {
        path.as_posix(): sha256(PROJECT_ROOT / path)
        for path in PROTECTED_PATHS
    }


def load_split(name: str) -> pd.DataFrame:
    expected_sizes = {"train": 43410, "validation": 5426, "test": 5427}
    frame = pd.read_parquet(PREPARED_DIRECTORY / f"{name}.parquet")
    if len(frame) != expected_sizes[name]:
        raise ValueError(f"Unexpected {name} split size: {len(frame)}")
    if not frame["split"].eq(name).all():
        raise ValueError(f"Unexpected split values in {name}.parquet")
    frame = frame.copy()
    frame["source"] = "gold"
    frame["oversampled"] = False
    return frame


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run or resume Experiment B: Gold plus standard oversampling."
    )
    parser.add_argument(
        "--resume-run",
        type=Path,
        help=(
            "Resume from an existing run directory under "
            "outputs/experiments/B_standard_oversampling/."
        ),
    )
    args = parser.parse_args(argv)

    started = time.perf_counter()
    before_hashes = protected_hashes()
    resume_run_dir = None
    recovery_checkpoint = None
    if args.resume_run is not None:
        requested_run_dir = args.resume_run
        if not requested_run_dir.is_absolute():
            requested_run_dir = PROJECT_ROOT / requested_run_dir
        resume_run_dir = requested_run_dir.resolve()
        output_root = OUTPUT_ROOT.resolve()
        if resume_run_dir.parent != output_root:
            raise ValueError(
                "--resume-run must point directly to an existing Experiment B run "
                f"under {output_root}."
            )
        if not resume_run_dir.is_dir():
            raise FileNotFoundError(f"Experiment B run directory not found: {resume_run_dir}")
        recovery_checkpoint = (
            resume_run_dir / "checkpoints" / "recovery" / "latest.pt"
        )
        if not recovery_checkpoint.is_file():
            raise FileNotFoundError(
                f"No recovery checkpoint exists in the selected run: {recovery_checkpoint}"
            )

    condition_cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    resource_cfg = json.loads(RESOURCE_CONFIG_PATH.read_text(encoding="utf-8"))
    formulation = json.loads(FORMULATION_CONFIG_PATH.read_text(encoding="utf-8"))

    if formulation.get("classification_loss") != "BCEWithLogitsLoss":
        raise ValueError("Expected BCEWithLogitsLoss.")
    if formulation.get("target_encoding") != "multi_hot":
        raise ValueError("Expected original multi-hot targets.")
    if formulation.get("class_weighting") is not False:
        raise ValueError("Class weighting must remain disabled.")
    if resource_cfg.get("freeze_strategy") != "last_2_layers":
        raise ValueError("Expected last_2_layers freeze strategy.")

    cfg = {**condition_cfg, **resource_cfg}
    cfg["epochs"] = condition_cfg["epochs"]
    cfg["num_labels"] = condition_cfg["num_labels"]
    cfg["num_workers"] = condition_cfg["num_workers"]
    cfg["cpu_num_threads"] = condition_cfg["cpu_num_threads"]
    cfg["device"] = condition_cfg["device"]

    print("[CONDITION B] Loading original prepared splits.", flush=True)
    gold = {
        name: load_split(name)
        for name in ("train", "validation", "test")
    }

    print("[CONDITION B] Creating deterministic standard oversampling.", flush=True)
    original_train = gold["train"].copy()
    oversampled_train = create_standard_oversampling(
        original_train,
        random_seed=cfg["seed"],
        target_count=164,
    )

    counts = get_target_training_counts(oversampled_train)
    expected = {"grief": 164, "pride": 164, "relief": 164, "nervousness": 164}
    if counts != expected:
        raise ValueError(f"Unexpected oversampled counts: {counts}")

    additions = int(oversampled_train["oversampled"].fillna(False).sum())
    if len(oversampled_train) != len(original_train) + additions:
        raise AssertionError("Oversampled row count does not reconcile.")

    splits_for_integrity = {
        "train": oversampled_train,
        "validation": gold["validation"],
        "test": gold["test"],
    }
    validate_label_ids(splits_for_integrity)

    # Repeated training texts are intentional for oversampling.
    # The official validation/test data must remain untouched.
    if not gold["validation"]["source"].eq("gold").all():
        raise AssertionError("Validation provenance changed.")
    if not gold["test"]["source"].eq("gold").all():
        raise AssertionError("Test provenance changed.")

    if resume_run_dir is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        run_dir = OUTPUT_ROOT / f"run-{run_id}"
        run_dir.mkdir(parents=True, exist_ok=False)
        recovery_checkpoint = (
            run_dir / "checkpoints" / "recovery" / "latest.pt"
        )
    else:
        run_dir = resume_run_dir

    checkpoint = run_dir / "checkpoints" / "best_model.pt"
    report_path = run_dir / "reports" / "metrics.json"
    per_class_path = run_dir / "reports" / "per_class_metrics.json"
    rare_path = run_dir / "reports" / "rare_emotion_metrics.json"

    training_cfg = TrainingConfig(
        model_name=cfg["model_name"],
        freeze_strategy=cfg["freeze_strategy"],
        max_length=cfg["max_length"],
        batch_size=cfg["batch_size"],
        epochs=cfg["epochs"],
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        warmup_ratio=cfg["warmup_ratio"],
        lambda_vad=cfg["lambda_vad"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        seed=cfg["seed"],
        num_workers=cfg["num_workers"],
        classification_threshold=cfg["classification_threshold"],
        cpu_num_threads=cfg["cpu_num_threads"],
    )

    print(f"[CONDITION B] Original training rows: {len(original_train)}", flush=True)
    print(f"[CONDITION B] Added oversampled rows: {additions}", flush=True)
    print(f"[CONDITION B] Final training rows: {len(oversampled_train)}", flush=True)
    print(f"[CONDITION B] Target counts: {counts}", flush=True)
    print(
        f"[CONDITION B] {'Resuming' if args.resume_run else 'Output'}: "
        f"{run_dir.relative_to(PROJECT_ROOT)}",
        flush=True,
    )

    tokenizer = load_tokenizer(cfg["model_name"])
    model = RobertaEmotionVADModel(
        model_name=cfg["model_name"],
        num_labels=cfg["num_labels"],
        lambda_vad=cfg["lambda_vad"],
        freeze_strategy=cfg["freeze_strategy"],
    )

    print("[CONDITION B] Starting training.", flush=True)
    training_result = train_model(
        model,
        tokenizer,
        oversampled_train,
        gold["validation"],
        config=training_cfg,
        checkpoint_path=checkpoint,
        device=cfg["device"],
        experiment_name="CONDITION B - STANDARD OVERSAMPLING",
        recovery_checkpoint_path=recovery_checkpoint,
        resume_from_recovery=args.resume_run is not None,
    )

    load_checkpoint(model, checkpoint, map_location=cfg["device"])
    model.to(cfg["device"])
    test_dataset = GoEmotionsVADDataset(
        gold["test"], tokenizer, max_length=cfg["max_length"]
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
    )

    print("[CONDITION B] Evaluating the selected checkpoint on the untouched test set.", flush=True)
    test_metrics = evaluate_model(
        model,
        test_loader,
        device=cfg["device"],
        classification_threshold=cfg["classification_threshold"],
    )

    report = {
        "condition": "CONDITION B - STANDARD OVERSAMPLING",
        "research_result_status": "REAL_EXPERIMENT",
        "training_data_definition": (
            "Original GoEmotions train split plus deterministic random oversampling "
            "with replacement; no generated synthetic text."
        ),
        "dataset_sizes": {
            "original_train": len(original_train),
            "oversampled_additions": additions,
            "final_train": len(oversampled_train),
            "validation": len(gold["validation"]),
            "test": len(gold["test"]),
        },
        "target_counts_before": get_target_training_counts(original_train),
        "target_counts_after": counts,
        "oversampling_seed": cfg["seed"],
        "oversampling_target_per_emotion": 164,
        "validation_test_unchanged": True,
        "model": {
            "name": cfg["model_name"],
            "freeze_strategy": cfg["freeze_strategy"],
            "max_length": cfg["max_length"],
            "batch_size": cfg["batch_size"],
            "gradient_accumulation_steps": cfg["gradient_accumulation_steps"],
            "epochs": cfg["epochs"],
            "learning_rate": cfg["learning_rate"],
            "lambda_vad": cfg["lambda_vad"],
            "classification_threshold": cfg["classification_threshold"],
        },
        "training_runtime_seconds": training_result["runtime_seconds"],
        "end_to_end_runtime_seconds": time.perf_counter() - started,
        "best_validation_macro_f1": training_result["best_validation_macro_f1"],
        "checkpoint": str(checkpoint.relative_to(PROJECT_ROOT)),
        "run_directory": str(run_dir.relative_to(PROJECT_ROOT)),
        "test_metrics": test_metrics,
        "vad_target_interpretation": (
            "NRC-VAD V1 lexicon-derived pseudo-targets, not human-annotated "
            "sentence-level VAD ground truth."
        ),
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    per_class_path.write_text(
        json.dumps(
            {
                name: {
                    "precision": test_metrics["per_class_precision"][name],
                    "recall": test_metrics["per_class_recall"][name],
                    "f1": test_metrics["per_class_f1"][name],
                }
                for name in test_metrics["per_class_f1"]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    rare_path.write_text(
        json.dumps(test_metrics["rare_target_f1"], indent=2),
        encoding="utf-8",
    )

    after_hashes = protected_hashes()
    if before_hashes != after_hashes:
        raise AssertionError("A protected source file changed during Experiment B.")

    print(f"[CONDITION B] Metrics saved: {report_path}", flush=True)
    print("[CONDITION B] COMPLETED successfully.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[CONDITION B] INTERRUPTED by user.", flush=True)
        raise SystemExit(130)
    except Exception as exc:
        print(f"[CONDITION B] ERROR: {type(exc).__name__}: {exc}", flush=True)
        raise
