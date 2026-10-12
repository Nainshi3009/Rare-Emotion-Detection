from __future__ import annotations

import hashlib
import json
import platform
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

from src.data_preparation import EXPECTED_LABEL_MAPPING  # noqa: E402
from src.dataset import GoEmotionsVADDataset  # noqa: E402
from src.model import RobertaEmotionVADModel  # noqa: E402
from src.training import (  # noqa: E402
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
EXPERIMENT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "experiments" / "A_gold_only"
PROTECTED_DATA_PATHS = (
    Path("Rare_Emotion_Detection_EDA.ipynb"),
    Path("data/processed/goemotions_vad_v1/train.parquet"),
    Path("data/processed/goemotions_vad_v1/validation.parquet"),
    Path("data/processed/goemotions_vad_v1/test.parquet"),
    Path("data/processed/goemotions_vad_v1/metadata.json"),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _protected_hashes() -> dict[str, str]:
    return {
        path.as_posix(): _sha256(PROJECT_ROOT / path)
        for path in PROTECTED_DATA_PATHS
    }


def _load_original_split(split: str) -> pd.DataFrame:
    path = PREPARED_DIRECTORY / f"{split}.parquet"
    frame = pd.read_parquet(path)
    if len(frame) != {"train": 43410, "validation": 5426, "test": 5427}[split]:
        raise ValueError(f"Unexpected original {split} size: {len(frame)}")
    if not frame["split"].eq(split).all():
        raise ValueError(f"Prepared {split} file contains rows from another split.")
    label_ids = {int(label) for labels in frame["labels"] for label in labels}
    if not label_ids.issubset(set(range(28))):
        raise ValueError(f"{split} contains labels outside the original 0-27 space.")
    frame = frame.copy()
    frame["source"] = "gold"
    frame["oversampled"] = False
    return frame


def main() -> int:
    condition_config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    resource_config = json.loads(RESOURCE_CONFIG_PATH.read_text(encoding="utf-8"))
    formulation_config = json.loads(FORMULATION_CONFIG_PATH.read_text(encoding="utf-8"))
    config_data = {**condition_config, **resource_config}
    config_data["epochs"] = condition_config["epochs"]
    config_data["num_labels"] = condition_config["num_labels"]
    config_data["num_workers"] = condition_config["num_workers"]
    config_data["cpu_num_threads"] = condition_config["cpu_num_threads"]
    config_data["device"] = condition_config["device"]

    if formulation_config.get("classification_loss") != "BCEWithLogitsLoss":
        raise ValueError("Gold-only run requires the approved BCEWithLogitsLoss formulation.")
    if formulation_config.get("target_encoding") != "multi_hot":
        raise ValueError("Gold-only run requires original binary multi-hot targets.")
    if formulation_config.get("class_weighting") is not False:
        raise ValueError("Class weighting is disabled for this baseline.")
    if formulation_config.get("lambda_vad") != resource_config["lambda_vad"]:
        raise ValueError("VAD loss weight differs between resource and loss configurations.")
    if formulation_config.get("classification_threshold") != resource_config["classification_threshold"]:
        raise ValueError("Emotion threshold differs between resource and loss configurations.")
    if condition_config.get("condition") != "CONDITION A - GOLD ONLY":
        raise ValueError("The configuration is not explicitly for CONDITION A — GOLD ONLY.")
    if condition_config["condition_definition"]["oversampling"] is not False:
        raise ValueError("Gold-only baseline must not include oversampled rows.")
    if condition_config["condition_definition"]["synthetic_data"] is not False:
        raise ValueError("Gold-only baseline must not include synthetic data.")
    if config_data["freeze_strategy"] != "last_2_layers":
        raise ValueError("Gold-only baseline must use the approved last_2_layers freeze strategy.")

    started = time.perf_counter()
    protected = _protected_hashes()
    print(
        "[CONDITION A - GOLD ONLY] Loading original Gold train, validation, and test datasets.",
        flush=True,
    )
    train_frame = _load_original_split("train")
    validation_frame = _load_original_split("validation")
    test_frame = _load_original_split("test")

    if train_frame["source"].ne("gold").any() or train_frame["oversampled"].any():
        raise AssertionError("Condition A training split must contain original gold rows only.")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_directory = EXPERIMENT_OUTPUT_ROOT / f"run-{run_id}"
    run_directory.mkdir(parents=True, exist_ok=False)
    checkpoint_path = run_directory / "checkpoints" / "best_model.pt"
    metrics_path = run_directory / "reports" / "metrics.json"
    per_class_path = run_directory / "reports" / "per_class_metrics.json"
    rare_path = run_directory / "reports" / "rare_emotion_metrics.json"

    training_config = TrainingConfig(
        model_name=config_data["model_name"],
        freeze_strategy=config_data["freeze_strategy"],
        max_length=config_data["max_length"],
        batch_size=config_data["batch_size"],
        epochs=config_data["epochs"],
        learning_rate=config_data["learning_rate"],
        weight_decay=config_data["weight_decay"],
        warmup_ratio=config_data["warmup_ratio"],
        lambda_vad=config_data["lambda_vad"],
        gradient_accumulation_steps=config_data["gradient_accumulation_steps"],
        seed=config_data["seed"],
        num_workers=config_data["num_workers"],
        classification_threshold=config_data["classification_threshold"],
        cpu_num_threads=config_data["cpu_num_threads"],
    )
    print(
        "[CONDITION A - GOLD ONLY] Initializing tokenizer and RoBERTa model "
        f"({config_data['model_name']}, freeze_strategy={config_data['freeze_strategy']}).",
        flush=True,
    )
    tokenizer = load_tokenizer(config_data["model_name"])
    model = RobertaEmotionVADModel(
        model_name=config_data["model_name"],
        num_labels=config_data["num_labels"],
        lambda_vad=config_data["lambda_vad"],
        freeze_strategy=config_data["freeze_strategy"],
    )

    print(
        "[CONDITION A - GOLD ONLY] Starting training; "
        f"output directory: {run_directory.relative_to(PROJECT_ROOT)}.",
        flush=True,
    )
    training_result = train_model(
        model,
        tokenizer,
        train_frame,
        validation_frame,
        config=training_config,
        checkpoint_path=checkpoint_path,
        device=config_data["device"],
        experiment_name="CONDITION A - GOLD ONLY",
    )
    load_checkpoint(model, checkpoint_path, map_location=config_data["device"])
    model.to(config_data["device"])
    test_dataset = GoEmotionsVADDataset(
        test_frame,
        tokenizer,
        max_length=config_data["max_length"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config_data["batch_size"],
        shuffle=False,
        num_workers=config_data["num_workers"],
    )
    print(
        "[CONDITION A - GOLD ONLY] Starting final test evaluation using the best "
        "validation checkpoint.",
        flush=True,
    )
    test_metrics = evaluate_model(
        model,
        test_loader,
        device=config_data["device"],
        classification_threshold=config_data["classification_threshold"],
    )

    model_revision = getattr(model.encoder.config, "_commit_hash", None)
    report_directory = metrics_path.parent
    report_directory.mkdir(parents=True, exist_ok=True)
    per_class_path.parent.mkdir(parents=True, exist_ok=True)
    rare_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "condition": "CONDITION A - GOLD ONLY",
        "research_result_status": "REAL_BASELINE_EXPERIMENT",
        "warning": (
            "VAD targets are NRC-VAD V1 lexicon-derived pseudo-targets, not "
            "human-annotated VAD ground truth."
        ),
        "dataset_sizes": {
            "train": len(train_frame),
            "validation": len(validation_frame),
            "test": len(test_frame),
        },
        "training_data_definition": (
            "Original GoEmotions train split only; no oversampling and no synthetic examples."
        ),
        "validation_data_definition": "Original GoEmotions validation split.",
        "test_data_definition": "Original GoEmotions test split; evaluated only after best validation checkpoint selection.",
        "seed": training_config.seed,
        "model": {
            "name": config_data["model_name"],
            "revision": model_revision,
            "num_labels": config_data["num_labels"],
            "vad_outputs": 3,
        },
        "hyperparameters": training_result["training_config"],
        "training_runtime_seconds": training_result["runtime_seconds"],
        "end_to_end_runtime_seconds": time.perf_counter() - started,
        "device": config_data["device"],
        "hardware": {
            "platform": platform.platform(),
            "cpu": platform.processor(),
            "logical_cpus": __import__("os").cpu_count(),
            "ram_total_bytes": psutil.virtual_memory().total,
            "ram_available_at_report_bytes": psutil.virtual_memory().available,
            "cuda_available": torch.cuda.is_available(),
        },
        "best_checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        "final_selected_checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        "checkpoint_selection": "best validation macro-F1; same checkpoint is final evaluated model",
        "resume_supported": False,
        "run_directory": str(run_directory.relative_to(PROJECT_ROOT)),
        "best_validation_macro_f1": training_result["best_validation_macro_f1"],
        "test_metrics": test_metrics,
    }
    metrics_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    per_class_path.write_text(
        json.dumps(
            {
                emotion: {
                    "precision": test_metrics["per_class_precision"][emotion],
                    "recall": test_metrics["per_class_recall"][emotion],
                    "f1": test_metrics["per_class_f1"][emotion],
                }
                for emotion in test_metrics["per_class_f1"]
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    rare_path.write_text(
        json.dumps(test_metrics["rare_target_f1"], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    protected_after = _protected_hashes()
    if protected != protected_after:
        raise AssertionError("A protected source file changed during baseline training.")
    print("CONDITION A - GOLD ONLY")
    print(f"Metrics: {metrics_path}")
    print(f"Per-class metrics: {per_class_path}")
    print(f"Rare-emotion metrics: {rare_path}")
    print("WARNING: This entry point runs full baseline training and test evaluation.")
    print(
        "[CONDITION A - GOLD ONLY] COMPLETED successfully; "
        f"run directory: {run_directory.relative_to(PROJECT_ROOT)}.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[CONDITION A - GOLD ONLY] INTERRUPTED by user.", flush=True)
        raise SystemExit(130)
    except Exception as error:
        print(
            f"[CONDITION A - GOLD ONLY] ERROR: {type(error).__name__}: {error}",
            flush=True,
        )
        raise
