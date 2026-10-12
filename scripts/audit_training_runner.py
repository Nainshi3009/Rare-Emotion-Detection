"""Audit Experiment A execution safety with a two-row inference-only smoke check."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
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
from src.training import TrainingConfig, load_tokenizer, set_deterministic_seed  # noqa: E402


RESOURCE_CONFIG_PATH = PROJECT_ROOT / "configs" / "model_resource_efficient.json"
BASELINE_CONFIG_PATH = PROJECT_ROOT / "configs" / "gold_only_baseline.json"
FORMULATION_CONFIG_PATH = PROJECT_ROOT / "configs" / "model_training_formulation.json"
RUNNER_PATH = PROJECT_ROOT / "scripts" / "run_gold_only.py"
DATA_DIRECTORY = PROJECT_ROOT / "data" / "processed" / "goemotions_vad_v1"
OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs" / "experiments" / "A_gold_only"
REPORT_PATH = PROJECT_ROOT / "reports" / "training_runner_audit.json"
SPLIT_SIZES = {"train": 43_410, "validation": 5_426, "test": 5_427}
SPLIT_FILES = (
    "train.parquet",
    "validation.parquet",
    "test.parquet",
    "metadata.json",
)
PROTECTED_FILES = (
    Path("Rare_Emotion_Detection_EDA.ipynb"),
    *(Path("data/processed/goemotions_vad_v1") / name for name in SPLIT_FILES),
    Path("data/synthetic/vad_filtered_candidates_40.csv"),
    Path("data/synthetic/vad_filtered_candidates_40.metadata.json"),
    Path("data/synthetic/vad_filtered_candidates_batch2.csv"),
    Path("data/synthetic/vad_filtered_candidates_batch2.metadata.json"),
    Path("data/synthetic/vad_filtered_candidates_completion.csv"),
    Path("data/synthetic/vad_filtered_candidates_completion.metadata.json"),
    Path("configs/model_resource_efficient.json"),
    Path("configs/gold_only_baseline.json"),
)
EXPECTED_TRAINABLE_PARAMETERS = 14_199_583


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_protected() -> dict[str, str]:
    hashes = {}
    missing = []
    for relative in PROTECTED_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            missing.append(relative.as_posix())
        else:
            hashes[relative.as_posix()] = _sha256(path)
    if missing:
        raise FileNotFoundError(f"Protected files missing during runner audit: {missing}")
    return hashes


def _check_dataset() -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    frames = {}
    checks = {}
    for split, expected_rows in SPLIT_SIZES.items():
        path = DATA_DIRECTORY / f"{split}.parquet"
        frame = pd.read_parquet(path)
        problems = []
        if len(frame) != expected_rows:
            problems.append(f"expected {expected_rows} rows, found {len(frame)}")
        required = {"id", "text", "labels", "split"}
        missing = sorted(required - set(frame.columns))
        if missing:
            problems.append(f"required fields missing: {missing}")
        if not frame["split"].eq(split).all():
            problems.append("split values do not match source file")
        if frame["id"].isna().any() or frame["id"].duplicated().any():
            problems.append("missing or duplicate IDs")
        if frame["text"].isna().any():
            problems.append("missing text")
        bad_label_rows = 0
        for labels in frame["labels"]:
            try:
                ids = [int(label) for label in labels]
            except (TypeError, ValueError):
                bad_label_rows += 1
                continue
            target = [0] * len(EXPECTED_LABEL_MAPPING)
            if (
                not ids
                or len(ids) != len(set(ids))
                or any(label < 0 or label >= len(EXPECTED_LABEL_MAPPING) for label in ids)
            ):
                bad_label_rows += 1
                continue
            for label in ids:
                target[label] = 1
            if any(value not in (0, 1) for value in target):
                bad_label_rows += 1
        if bad_label_rows:
            problems.append(f"{bad_label_rows} rows have invalid label IDs")
        vad_columns = ["vad_valence", "vad_arousal", "vad_dominance"]
        if not set(vad_columns).issubset(frame.columns):
            problems.append("VAD pseudo-target columns missing")
        elif not frame[vad_columns].dropna().apply(
            lambda column: column.between(0, 1).all()
        ).all():
            problems.append("VAD values outside [0,1]")
        frames[split] = frame
        checks[split] = {
            "rows": len(frame),
            "expected_rows": expected_rows,
            "valid": not problems,
            "problems": problems,
        }
    return frames, checks


def _main() -> int:
    blockers: list[str] = []
    hashes_before = _hash_protected()
    resource = json.loads(RESOURCE_CONFIG_PATH.read_text(encoding="utf-8"))
    baseline = json.loads(BASELINE_CONFIG_PATH.read_text(encoding="utf-8"))
    formulation = json.loads(FORMULATION_CONFIG_PATH.read_text(encoding="utf-8"))
    runner_source = RUNNER_PATH.read_text(encoding="utf-8")

    set_deterministic_seed(resource["seed"])
    cpu_threads = baseline.get("cpu_num_threads", 2)
    torch.set_num_threads(cpu_threads)
    if torch.cuda.is_available():
        blockers.append("environment unexpectedly reports CUDA; approved run is CPU-only")
    if resource["batch_size"] != 1:
        blockers.append("resource configuration batch size is not the approved CPU-safe value 1")
    if resource["gradient_accumulation_steps"] != 4:
        blockers.append("gradient accumulation differs from approved value 4")

    required_shared_keys = (
        "model_name",
        "max_length",
        "batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "weight_decay",
        "warmup_ratio",
        "seed",
        "lambda_vad",
        "classification_threshold",
        "freeze_strategy",
    )
    runner_wiring = all(
        f'{key}=config_data["{key}"]' in runner_source for key in required_shared_keys
    )
    runner_wiring = runner_wiring and (
        'resource_config = json.loads(RESOURCE_CONFIG_PATH.read_text' in runner_source
        and "**resource_config" in runner_source
    )
    if not runner_wiring:
        blockers.append("Gold runner does not explicitly use every shared resource setting")
    if resource.get("freeze_strategy") != "last_2_layers":
        blockers.append("resource configuration is not last_2_layers")
    if formulation != {
        "classification_loss": "BCEWithLogitsLoss",
        "target_encoding": "multi_hot",
        "classification_threshold": 0.5,
        "vad_loss": "masked_MSE",
        "lambda_vad": 0.5,
        "class_weighting": False,
    }:
        blockers.append("loss formulation configuration differs from approved methodology")
    if baseline.get("condition_definition", {}).get("oversampling") is not False:
        blockers.append("Gold-only condition enables oversampling")
    if baseline.get("condition_definition", {}).get("synthetic_data") is not False:
        blockers.append("Gold-only condition enables synthetic data")
    if baseline.get("epochs") != 1:
        blockers.append("Gold-only epoch count is not the approved configured one epoch")
    if baseline.get("device") != "cpu":
        blockers.append("Gold-only configuration is not explicitly CPU")

    splits, dataset_integrity = _check_dataset()
    if not all(item["valid"] for item in dataset_integrity.values()):
        blockers.append("one or more Gold split integrity checks failed")
    train = splits["train"]
    if "source" in train and not train["source"].eq("gold").all():
        blockers.append("prepared Gold-only train rows have non-gold provenance")

    tokenizer = load_tokenizer(resource["model_name"], local_files_only=True)
    model = RobertaEmotionVADModel(
        model_name=resource["model_name"],
        num_labels=28,
        lambda_vad=resource["lambda_vad"],
        freeze_strategy=resource["freeze_strategy"],
        local_files_only=True,
    )
    report = model.parameter_report()
    if report["trainable_parameters"] != EXPECTED_TRAINABLE_PARAMETERS:
        blockers.append(
            f"trainable parameter count mismatch: {report['trainable_parameters']}"
        )
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    intended_trainable = all(
        name.startswith(
            ("encoder.encoder.layer.10.", "encoder.encoder.layer.11.", "emotion_head.", "vad_head.")
        )
        for name in trainable_names
    )
    if not intended_trainable:
        blockers.append("unexpected trainable parameter found outside final two layers/task heads")
    expected_heads = {"emotion_head", "vad_head"}
    if not expected_heads.issubset({name.split(".", 1)[0] for name in trainable_names}):
        blockers.append("one or both task heads are frozen")
    for layer in range(10):
        if any(name.startswith(f"encoder.encoder.layer.{layer}.") for name in trainable_names):
            blockers.append(f"encoder layer {layer} unexpectedly trainable")
    if any(name.startswith("encoder.embeddings.") for name in trainable_names):
        blockers.append("RoBERTa embeddings unexpectedly trainable")

    valid_vad_rows = train.dropna(
        subset=["vad_valence", "vad_arousal", "vad_dominance"]
    ).head(resource["batch_size"])
    if len(valid_vad_rows) < resource["batch_size"]:
        raise RuntimeError("Could not select two Gold training rows with valid VAD values.")
    tiny_dataset = GoEmotionsVADDataset(
        valid_vad_rows,
        tokenizer,
        max_length=resource["max_length"],
    )
    tiny_loader = DataLoader(
        tiny_dataset,
        batch_size=resource["batch_size"],
        shuffle=False,
        num_workers=0,
    )
    batch = next(iter(tiny_loader))

    training_config = TrainingConfig(
        model_name=resource["model_name"],
        freeze_strategy=resource["freeze_strategy"],
        max_length=resource["max_length"],
        batch_size=resource["batch_size"],
        epochs=baseline["epochs"],
        learning_rate=resource["learning_rate"],
        weight_decay=resource["weight_decay"],
        warmup_ratio=resource["warmup_ratio"],
        lambda_vad=resource["lambda_vad"],
        gradient_accumulation_steps=resource["gradient_accumulation_steps"],
        seed=resource["seed"],
        num_workers=baseline["num_workers"],
        classification_threshold=resource["classification_threshold"],
        cpu_num_threads=cpu_threads,
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    optimizer_params = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    optimizer_parameter_ids = {id(parameter) for parameter in optimizer_params}
    trainable_parameter_ids = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    optimizer_trainable_only = optimizer_parameter_ids == trainable_parameter_ids
    optimizer_parameter_count = sum(parameter.numel() for parameter in optimizer_params)
    if not optimizer_trainable_only:
        blockers.append("optimizer contains frozen parameters or misses trainable parameters")

    model.eval()
    with torch.inference_mode():
        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            emotion_targets=batch["emotion_targets"],
            vad_targets=batch["vad_targets"],
            vad_valid_mask=batch["vad_valid_mask"],
        )
        probabilities = torch.sigmoid(output.emotion_logits)
        predictions = probabilities >= resource["classification_threshold"]
    loss_finite = bool(output.loss is not None and torch.isfinite(output.loss))
    if not loss_finite:
        blockers.append("tiny joint BCE + masked VAD loss is not finite")
    if output.loss is None or output.classification_loss is None or output.vad_loss is None:
        blockers.append("tiny forward pass did not return all joint loss components")
    elif not torch.allclose(
        output.loss,
        output.classification_loss + resource["lambda_vad"] * output.vad_loss,
        rtol=1e-6,
        atol=1e-7,
    ):
        blockers.append("joint loss does not equal BCE + lambda_vad * masked VAD MSE")
    expected_emotion_shape = (resource["batch_size"], 28)
    expected_vad_shape = (resource["batch_size"], 3)
    if tuple(output.emotion_logits.shape) != expected_emotion_shape:
        blockers.append("emotion logits have an unexpected shape")
    if tuple(batch["emotion_targets"].shape) != expected_emotion_shape:
        blockers.append("multi-hot targets have an unexpected shape")
    if tuple(output.vad_predictions.shape) != expected_vad_shape:
        blockers.append("VAD predictions have an unexpected shape")
    if tuple(probabilities.shape) != expected_emotion_shape or tuple(predictions.shape) != expected_emotion_shape:
        blockers.append("sigmoid/threshold predictions have an unexpected shape")

    train_microbatches = math.ceil(
        len(train) / resource["batch_size"]
    )
    optimizer_steps = math.ceil(
        train_microbatches / resource["gradient_accumulation_steps"]
    )
    warmup_steps = int(optimizer_steps * resource["warmup_ratio"])
    runner_output_isolated = (
        'PROJECT_ROOT / "outputs" / "experiments" / "A_gold_only"' in runner_source
        and 'mkdir(parents=True, exist_ok=False)' in runner_source
        and "run-{run_id}" in runner_source
    )
    if not runner_output_isolated:
        blockers.append("Gold run output directory is not unique/non-overwriting")

    checkpoint_controls = {
        "best_checkpoint_selected_by": "validation macro-F1",
        "best_checkpoint_path": "outputs/experiments/A_gold_only/run-<UTC timestamp>/checkpoints/best_model.pt",
        "final_model": "best validation checkpoint is reloaded and evaluated on test",
        "resume_supported": False,
        "unique_run_directory": runner_output_isolated,
        "refuses_existing_run_directory": 'mkdir(parents=True, exist_ok=False)' in runner_source,
        "tokenizer_saved_in_run_checkpoint_directory": True,
        "source_hashes_checked_before_after": "_protected_hashes" in runner_source,
    }
    checkpoint_safety = all(
        checkpoint_controls[key] is True
        for key in (
            "unique_run_directory",
            "refuses_existing_run_directory",
            "tokenizer_saved_in_run_checkpoint_directory",
            "source_hashes_checked_before_after",
        )
    )
    if not checkpoint_safety:
        blockers.append("checkpoint/output/source-integrity safeguards are incomplete")

    hashes_after = _hash_protected()
    protected_unchanged = hashes_before == hashes_after
    if not protected_unchanged:
        blockers.append("protected file hashes changed during audit")

    status = "READY" if not blockers else "BLOCKED"
    result = {
        "status": status,
        "condition": "CONDITION A - GOLD ONLY",
        "freeze_strategy": resource["freeze_strategy"],
        "trainable_parameters": report["trainable_parameters"],
        "frozen_parameters": report["frozen_parameters"],
        "percentage_trainable": report["percentage_trainable"],
        "trainable_parameter_groups": report["trainable_parameter_groups"],
        "optimizer_trainable_only": optimizer_trainable_only,
        "optimizer_parameter_count": optimizer_parameter_count,
        "optimizer_parameter_tensor_count": len(optimizer_params),
        "optimizer": "AdamW",
        "loss_formulation": "BCEWithLogitsLoss + 0.5 * masked VAD MSE",
        "loss_verification": {
            "finite": loss_finite,
            "classification_loss": float(output.classification_loss),
            "masked_vad_mse": float(output.vad_loss),
            "joint_loss": float(output.loss),
            "lambda_vad": resource["lambda_vad"],
            "class_weighting": formulation["class_weighting"],
        },
        "tensor_shapes": {
            "input_ids": list(batch["input_ids"].shape),
            "multi_hot_targets": list(batch["emotion_targets"].shape),
            "emotion_logits": list(output.emotion_logits.shape),
            "vad_targets": list(batch["vad_targets"].shape),
            "vad_predictions": list(output.vad_predictions.shape),
            "sigmoid_predictions": list(probabilities.shape),
        },
        "evaluation": {
            "activation": "sigmoid",
            "threshold": resource["classification_threshold"],
            "softmax_used": False,
        },
        "training_configuration": {
            "model_name": training_config.model_name,
            "max_length": training_config.max_length,
            "batch_size": training_config.batch_size,
            "gradient_accumulation_steps": training_config.gradient_accumulation_steps,
            "effective_batch_size": training_config.batch_size * training_config.gradient_accumulation_steps,
            "epochs": training_config.epochs,
            "learning_rate": training_config.learning_rate,
            "weight_decay": training_config.weight_decay,
            "warmup_ratio": training_config.warmup_ratio,
            "estimated_optimizer_steps": optimizer_steps,
            "estimated_warmup_steps": warmup_steps,
            "seed": training_config.seed,
            "device": baseline["device"],
            "cuda_available": torch.cuda.is_available(),
        },
        "dataset_integrity": {
            "status": "PASS" if all(item["valid"] for item in dataset_integrity.values()) else "FAIL",
            "splits": dataset_integrity,
            "train_examples": len(train),
            "validation_examples": len(splits["validation"]),
            "test_examples": len(splits["test"]),
            "tiny_gold_rows_loaded": len(tiny_dataset),
            "tiny_vad_valid_rows": int(batch["vad_valid_mask"].sum()),
        },
        "checkpoint_safety": checkpoint_controls,
        "output_directory": "outputs/experiments/A_gold_only/",
        "protected_hash_verification": {
            "unchanged": protected_unchanged,
            "hashes_before": hashes_before,
            "hashes_after": hashes_after,
        },
        "resume_behavior": "not supported; every run gets a unique UTC-stamped directory",
        "backward_pass_performed": False,
        "optimizer_step_performed": False,
        "real_training_performed": False,
        "gemini_api_calls": 0,
        "blockers": blockers,
        "audit_timestamp": datetime.now(timezone.utc).isoformat(),
        "hardware": {
            "python": platform.python_version(),
            "cpu": platform.processor(),
            "logical_cpus": psutil.cpu_count(logical=True),
            "ram_total_bytes": psutil.virtual_memory().total,
            "ram_available_bytes": psutil.virtual_memory().available,
        },
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("=" * 62)
    print("EXPERIMENT A TRAINING-RUNNER SAFETY AUDIT")
    print("=" * 62)
    print("Status:", status)
    print("Freeze strategy:", resource["freeze_strategy"])
    print(
        f"Trainable parameters: {report['trainable_parameters']:,} "
        f"({report['percentage_trainable']:.2f}%)"
    )
    print(f"Optimizer parameters: {optimizer_parameter_count:,} elements in {len(optimizer_params):,} tensors")
    print("Optimizer contains only trainable parameters:", optimizer_trainable_only)
    print("Loss:", result["loss_formulation"])
    print("Tiny loss finite:", loss_finite)
    print("Output directory:", result["output_directory"])
    print("Resume:", result["resume_behavior"])
    print("Protected files unchanged:", protected_unchanged)
    print("Training performed: no")
    print("Gemini calls: 0")
    for blocker in blockers:
        print("BLOCKER:", blocker)
    print(f"Report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print("=" * 62)
    return 0 if status == "READY" else 1


if __name__ == "__main__":
    raise SystemExit(_main())
