"""Bounded inference/loss throughput benchmark for the approved CPU model."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import psutil
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import GoEmotionsVADDataset  # noqa: E402
from src.model import RobertaEmotionVADModel  # noqa: E402
from src.training import load_tokenizer, set_deterministic_seed  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "configs" / "model_resource_efficient.json"
GOLD_CONFIG_PATH = PROJECT_ROOT / "configs" / "gold_only_baseline.json"
FORMULATION_PATH = PROJECT_ROOT / "configs" / "model_training_formulation.json"
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "goemotions_vad_v1" / "train.parquet"
REPORT_PATH = PROJECT_ROOT / "reports" / "training_throughput_benchmark.json"
ITERATIONS = 3
EXAMPLE_COUNTS = (1_000, 10_000, 43_410)
PROTECTED_FILES = (
    Path("Rare_Emotion_Detection_EDA.ipynb"),
    Path("data/processed/goemotions_vad_v1/train.parquet"),
    Path("data/processed/goemotions_vad_v1/validation.parquet"),
    Path("data/processed/goemotions_vad_v1/test.parquet"),
    Path("data/processed/goemotions_vad_v1/metadata.json"),
    Path("data/synthetic/vad_filtered_candidates_40.csv"),
    Path("data/synthetic/vad_filtered_candidates_40.metadata.json"),
    Path("data/synthetic/vad_filtered_candidates_batch2.csv"),
    Path("data/synthetic/vad_filtered_candidates_batch2.metadata.json"),
    Path("configs/model_resource_efficient.json"),
    Path("configs/gold_only_baseline.json"),
)


def _hash_files() -> dict[str, str]:
    hashes = {}
    for relative in PROTECTED_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Protected file missing: {relative}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[relative.as_posix()] = digest.hexdigest()
    return hashes


def _make_batch(dataset: GoEmotionsVADDataset, batch_size: int) -> dict[str, torch.Tensor]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    if len(batch["input_ids"]) != batch_size:
        raise RuntimeError(f"Could not construct a full batch of size {batch_size}.")
    return batch


def _time_forward_losses(
    model: RobertaEmotionVADModel,
    batch: dict[str, torch.Tensor],
    batch_size: int,
) -> dict[str, Any]:
    times = []
    losses = []
    model.eval()
    for _ in range(ITERATIONS):
        started = time.perf_counter()
        with torch.inference_mode():
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                emotion_targets=batch["emotion_targets"],
                vad_targets=batch["vad_targets"],
                vad_valid_mask=batch["vad_valid_mask"],
            )
        elapsed = time.perf_counter() - started
        if output.loss is None or output.classification_loss is None or output.vad_loss is None:
            raise RuntimeError("Model did not return all expected loss values.")
        if not torch.isfinite(output.loss):
            raise RuntimeError("Combined loss is non-finite.")
        times.append(elapsed)
        losses.append(
            {
                "total": float(output.loss),
                "bce": float(output.classification_loss),
                "masked_vad_mse": float(output.vad_loss),
            }
        )

    average_seconds = sum(times) / len(times)
    steady_state_seconds = sum(times[1:]) / max(1, len(times) - 1)
    throughput = batch_size / average_seconds
    return {
        "batch_size": batch_size,
        "iterations": ITERATIONS,
        "iteration_seconds": times,
        "first_forward_loss_seconds": times[0],
        "average_forward_loss_seconds": average_seconds,
        "average_subsequent_forward_loss_seconds": steady_state_seconds,
        "examples_per_second": throughput,
        "estimated_seconds": {
            str(count): count / throughput for count in EXAMPLE_COUNTS
        },
        "losses": losses,
        "input_shape": list(batch["input_ids"].shape),
        "target_shape": list(batch["emotion_targets"].shape),
        "vad_target_shape": list(batch["vad_targets"].shape),
    }


def main() -> int:
    if torch.cuda.is_available():
        raise RuntimeError("This benchmark is specified for the current CPU-only machine.")

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    gold_config = json.loads(GOLD_CONFIG_PATH.read_text(encoding="utf-8"))
    formulation = json.loads(FORMULATION_PATH.read_text(encoding="utf-8"))
    if config["model_name"] != "roberta-base":
        raise ValueError("Expected roberta-base.")
    if config["freeze_strategy"] != "last_2_layers":
        raise ValueError("Expected last_2_layers freeze strategy.")
    if config["max_length"] != 128:
        raise ValueError("Expected configured max_length=128.")
    if formulation.get("classification_loss") != "BCEWithLogitsLoss":
        raise ValueError("Expected the approved BCEWithLogitsLoss formulation.")
    if formulation.get("vad_loss") != "masked_MSE" or config["lambda_vad"] != 0.5:
        raise ValueError("Expected masked VAD MSE with lambda_vad=0.5.")
    if formulation.get("class_weighting") is not False:
        raise ValueError("The approved benchmark uses unweighted BCE.")
    if not TRAIN_PATH.is_file():
        raise FileNotFoundError(f"Gold training Parquet is missing: {TRAIN_PATH}")

    protected_before = _hash_files()
    set_deterministic_seed(config["seed"])
    torch.set_num_threads(max(1, int(gold_config.get("cpu_num_threads", 2))))

    train_frame = pd.read_parquet(TRAIN_PATH)
    if len(train_frame) != 43_410 or not train_frame["split"].eq("train").all():
        raise ValueError("Expected only the original 43,410-row GoEmotions train split.")
    vad_columns = ["vad_valence", "vad_arousal", "vad_dominance"]
    tiny_frame = train_frame.dropna(subset=vad_columns).head(2).copy()
    if len(tiny_frame) != 2:
        raise ValueError("Could not select two Gold rows with valid VAD targets.")

    tokenizer_started = time.perf_counter()
    tokenizer = load_tokenizer(config["model_name"], local_files_only=True)
    tokenizer_seconds = time.perf_counter() - tokenizer_started
    tiny_dataset = GoEmotionsVADDataset(
        tiny_frame,
        tokenizer,
        max_length=config["max_length"],
    )

    model_started = time.perf_counter()
    model = RobertaEmotionVADModel(
        model_name=config["model_name"],
        num_labels=28,
        lambda_vad=config["lambda_vad"],
        freeze_strategy=config["freeze_strategy"],
        local_files_only=True,
    )
    model.to("cpu")
    model_initialization_seconds = time.perf_counter() - model_started
    parameter_report = model.parameter_report()
    if parameter_report["trainable_parameters"] != 14_199_583:
        raise AssertionError(
            "Unexpected trainable parameter count: "
            f"{parameter_report['trainable_parameters']}"
        )

    results = {}
    for batch_size in (2, 1):
        batch = _make_batch(tiny_dataset, batch_size)
        results[str(batch_size)] = _time_forward_losses(model, batch, batch_size)

    protected_after = _hash_files()
    if protected_before != protected_after:
        raise AssertionError("Protected data/config/notebook hashes changed during benchmark.")

    report = {
        "benchmark": "RoBERTa Gold-only forward/loss throughput; resource estimate only",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "name": config["model_name"],
            "revision": getattr(model.encoder.config, "_commit_hash", None),
            "freeze_strategy": config["freeze_strategy"],
            "total_parameters": parameter_report["total_parameters"],
            "trainable_parameters": parameter_report["trainable_parameters"],
            "trainable_percentage": parameter_report["percentage_trainable"],
        },
        "data": {
            "source": "Original GoEmotions train split with existing VAD pseudo-targets",
            "train_rows": len(train_frame),
            "benchmark_rows": len(tiny_dataset),
            "max_length": config["max_length"],
            "selected_rows_have_valid_vad": bool(
                tiny_dataset.vad_valid_mask.all()
            ),
        },
        "loss": {
            "classification": "unweighted BCEWithLogitsLoss on binary multi-hot targets",
            "vad": "masked MSE on NRC-VAD V1 lexicon-derived pseudo-targets",
            "lambda_vad": config["lambda_vad"],
            "joint": "BCEWithLogitsLoss + 0.5 * masked VAD MSE",
        },
        "device": "cpu",
        "cuda_available": torch.cuda.is_available(),
        "iterations_per_batch_size": ITERATIONS,
        "tokenizer_initialization_seconds": tokenizer_seconds,
        "model_initialization_seconds": model_initialization_seconds,
        "batch_size_results": results,
        "protected_hashes_unchanged": True,
        "protected_hashes_before": protected_before,
        "protected_hashes_after": protected_after,
        "backward_pass_performed": False,
        "optimizer_step_performed": False,
        "checkpoint_saved": False,
        "training_performed": False,
        "gemini_api_calls": 0,
        "interpretation": (
            "Timing covers forward plus both losses in inference mode only. "
            "It excludes backward, optimizer work, tokenization of the full corpus, "
            "validation, and checkpoint I/O. Extrapolations are rough estimates, not results."
        ),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("TRAINING THROUGHPUT BENCHMARK - FORWARD/LOSS ONLY")
    print(f"Model initialization: {model_initialization_seconds:.2f} sec")
    print(f"Tokenizer initialization: {tokenizer_seconds:.2f} sec")
    print(f"Model: {config['model_name']} ({config['freeze_strategy']}); max_length={config['max_length']}")
    for batch_size in (2, 1):
        result = results[str(batch_size)]
        print(f"\nBatch size {batch_size}")
        print(f"  Iterations: {result['iterations']}")
        print(f"  First forward/loss: {result['first_forward_loss_seconds']:.2f} sec")
        print(f"  Average forward/loss: {result['average_forward_loss_seconds']:.2f} sec")
        print(f"  Average subsequent: {result['average_subsequent_forward_loss_seconds']:.2f} sec")
        print(f"  Throughput: {result['examples_per_second']:.4f} examples/sec")
        for count in EXAMPLE_COUNTS:
            seconds = result["estimated_seconds"][str(count)]
            print(f"  Estimated {count:,} examples: {seconds / 3600:.2f} hours")
    print("Backward: no; optimizer step: no; checkpoint: no; Gemini calls: 0")
    print(f"Protected hashes unchanged: {report['protected_hashes_unchanged']}")
    print(f"Report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
