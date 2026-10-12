"""One-step CPU resource estimates for RoBERTa encoder-freezing strategies.

This script benchmarks only original GoEmotions training examples. It does not
train an experiment, run an epoch, or use synthetic data.
"""

from __future__ import annotations

import gc
import math
import sys
import threading
import time
from pathlib import Path

import pandas as pd
import psutil
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import GoEmotionsVADDataset  # noqa: E402
from src.model import RobertaEmotionVADModel  # noqa: E402
from src.training import load_tokenizer, set_deterministic_seed  # noqa: E402


TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "goemotions_vad_v1" / "train.parquet"
MODEL_NAME = "roberta-base"
MAX_LENGTH = 128
BATCH_SIZE = 2
SEED = 20261008
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
TRAIN_ROWS = 43_410
TRAIN_MICROBATCHES = math.ceil(TRAIN_ROWS / BATCH_SIZE)


def _set_strategy(model: RobertaEmotionVADModel, strategy: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(True)

    if strategy == "full":
        return
    if strategy == "frozen":
        for parameter in model.encoder.parameters():
            parameter.requires_grad_(False)
        return
    if strategy != "last_two":
        raise ValueError(f"Unknown strategy: {strategy}")

    for parameter in model.encoder.embeddings.parameters():
        parameter.requires_grad_(False)
    layers = model.encoder.encoder.layer
    if len(layers) < 2:
        raise RuntimeError("Expected at least two RoBERTa transformer layers.")
    for layer in layers[:-2]:
        for parameter in layer.parameters():
            parameter.requires_grad_(False)


def _sample_batch(frame: pd.DataFrame, tokenizer) -> dict[str, torch.Tensor]:
    dataset = GoEmotionsVADDataset(
        frame.head(BATCH_SIZE),
        tokenizer,
        max_length=MAX_LENGTH,
    )
    return {
        key: torch.stack([dataset[i][key] for i in range(BATCH_SIZE)])
        for key in dataset[0]
    }


def _benchmark(model: RobertaEmotionVADModel, batch: dict[str, torch.Tensor]) -> dict[str, float]:
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    process = psutil.Process()
    stop_sampling = threading.Event()
    peak_rss = [process.memory_info().rss]

    def sample_memory() -> None:
        while not stop_sampling.wait(0.05):
            try:
                peak_rss[0] = max(peak_rss[0], process.memory_info().rss)
            except psutil.Error:
                return

    rss_before = process.memory_info().rss
    available_before = psutil.virtual_memory().available
    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    output = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        emotion_targets=batch["emotion_targets"],
        vad_targets=batch["vad_targets"],
        vad_valid_mask=batch["vad_valid_mask"],
    )
    if output.loss is None or not torch.isfinite(output.loss):
        raise RuntimeError("Combined benchmark loss is missing or non-finite.")
    output.loss.backward()
    optimizer.step()
    step_seconds = time.perf_counter() - started
    peak_rss[0] = max(peak_rss[0], process.memory_info().rss)
    stop_sampling.set()
    sampler.join()
    rss_after = process.memory_info().rss
    available_after = psutil.virtual_memory().available
    result = {
        "step_seconds": step_seconds,
        "rss_before_gib": rss_before / 1024**3,
        "rss_after_gib": rss_after / 1024**3,
        "peak_rss_gib": peak_rss[0] / 1024**3,
        "peak_rss_delta_gib": max(0, peak_rss[0] - rss_before) / 1024**3,
        "available_before_gib": available_before / 1024**3,
        "available_after_gib": available_after / 1024**3,
    }
    del optimizer, output
    model.zero_grad(set_to_none=True)
    gc.collect()
    return result


def main() -> int:
    set_deterministic_seed(SEED)
    torch.set_num_threads(2)
    tokenizer = load_tokenizer(MODEL_NAME, local_files_only=True)
    train_frame = pd.read_parquet(TRAIN_PATH)
    if len(train_frame) != TRAIN_ROWS or not train_frame["split"].eq("train").all():
        raise ValueError("Expected the original GoEmotions train split only.")
    batch = _sample_batch(train_frame, tokenizer)
    del train_frame
    gc.collect()

    model = RobertaEmotionVADModel(
        model_name=MODEL_NAME,
        num_labels=28,
        lambda_vad=0.5,
        local_files_only=True,
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    checkpoint_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    ) + sum(buffer.numel() * buffer.element_size() for buffer in model.buffers())

    print("FREEZING STRATEGY ESTIMATES - ONE CPU STEP ONLY")
    print(f"Model: {MODEL_NAME} (revision {model.encoder.config._commit_hash})")
    print(f"Batch: {BATCH_SIZE}; max_length: {MAX_LENGTH}; training examples: {TRAIN_ROWS:,}")
    print("Benchmark: exactly one forward/backward/AdamW step per strategy; no epoch or validation pass.")
    print("Checkpoint estimate includes the full joint model state, independent of freeze strategy.")

    for strategy, label in (
        ("full", "Full fine-tuning"),
        ("frozen", "Frozen encoder"),
        ("last_two", "Final 2 encoder layers"),
    ):
        _set_strategy(model, strategy)
        trainable_parameters = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        optimizer_state_bytes = trainable_parameters * 8
        _set_strategy(model, strategy)
        result = _benchmark(model, batch)
        estimated_train_hours = (
            result["step_seconds"] * TRAIN_MICROBATCHES / 3600
        )
        print(f"\n{label}")
        print(f"  Total parameters: {total_parameters:,}")
        print(
            f"  Trainable parameters: {trainable_parameters:,} "
            f"({100 * trainable_parameters / total_parameters:.2f}%)"
        )
        print(
            f"  Approximate AdamW state: {optimizer_state_bytes / 1024**2:.2f} MiB "
            f"({optimizer_state_bytes / 1024**3:.4f} GiB)"
        )
        print(f"  Approximate checkpoint: {checkpoint_bytes / 1024**2:.1f} MiB")
        print(f"  Forward + backward + optimizer step: {result['step_seconds']:.2f} sec")
        print(
            f"  Process RSS before/after/peak: {result['rss_before_gib']:.2f} / "
            f"{result['rss_after_gib']:.2f} / {result['peak_rss_gib']:.2f} GiB"
        )
        print(f"  Observed peak RSS increase: {result['peak_rss_delta_gib']:.2f} GiB")
        print(
            f"  System available RAM before/after: {result['available_before_gib']:.2f} / "
            f"{result['available_after_gib']:.2f} GiB"
        )
        print(
            f"  Rough train-epoch estimate ({TRAIN_MICROBATCHES:,} microbatches): "
            f"{estimated_train_hours:.2f} hours"
        )
        if result["available_after_gib"] < 0.25:
            print("  WARNING: very low system-available RAM after this benchmark step.")

    print("\nEstimates only; no full training or scientific evaluation was performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
