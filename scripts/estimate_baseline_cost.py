from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import psutil
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import GoEmotionsVADDataset  # noqa: E402
from src.model import RobertaEmotionVADModel  # noqa: E402
from src.training import load_tokenizer, set_deterministic_seed  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "configs" / "gold_only_baseline.json"
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "goemotions_vad_v1" / "train.parquet"
VALIDATION_PATH = (
    PROJECT_ROOT / "data" / "processed" / "goemotions_vad_v1" / "validation.parquet"
)


def main() -> int:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    seed = config["seed"]
    set_deterministic_seed(seed)
    torch.set_num_threads(config["cpu_num_threads"])

    train_frame = __import__("pandas").read_parquet(TRAIN_PATH)
    validation_frame = __import__("pandas").read_parquet(VALIDATION_PATH)
    train_rows = len(train_frame)
    validation_rows = len(validation_frame)
    microbatches = math.ceil(train_rows / config["batch_size"])
    optimizer_steps_per_epoch = math.ceil(
        microbatches / config["gradient_accumulation_steps"]
    )
    effective_batch_size = (
        config["batch_size"] * config["gradient_accumulation_steps"]
    )

    memory_before = psutil.virtual_memory().available
    tokenizer = load_tokenizer(config["model_name"], local_files_only=True)
    model = RobertaEmotionVADModel(
        model_name=config["model_name"],
        num_labels=config["num_labels"],
        lambda_vad=config["lambda_vad"],
        local_files_only=True,
    )
    model.train()

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )
    parameter_checkpoint_mib = parameter_bytes / (1024**2)
    state_checkpoint_mib = (
        parameter_bytes + sum(buffer.numel() * buffer.element_size() for buffer in model.buffers())
    ) / (1024**2)
    estimated_adamw_state_gib = total_parameters * 8 / (1024**3)
    estimated_training_resident_gib = total_parameters * 16 / (1024**3)

    tiny_train = train_frame.head(config["batch_size"] * 2).copy()
    tiny_validation = validation_frame.head(config["batch_size"]).copy()
    train_dataset = GoEmotionsVADDataset(
        tiny_train,
        tokenizer,
        max_length=config["max_length"],
    )
    validation_dataset = GoEmotionsVADDataset(
        tiny_validation,
        tokenizer,
        max_length=config["max_length"],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    train_batch_times = []
    optimizer_state_initialized = False
    for batch in train_loader:
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
            raise RuntimeError("Benchmark combined loss is missing or non-finite.")
        output.loss.backward()
        optimizer.step()
        train_batch_times.append(time.perf_counter() - started)
        optimizer_state_initialized = bool(optimizer.state)
        break

    model.eval()
    eval_batch_times = []
    with torch.inference_mode():
        for batch in validation_loader:
            started = time.perf_counter()
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                emotion_targets=batch["emotion_targets"],
                vad_targets=batch["vad_targets"],
                vad_valid_mask=batch["vad_valid_mask"],
            )
            if not torch.isfinite(output.loss):
                raise RuntimeError("Benchmark evaluation loss is non-finite.")
            eval_batch_times.append(time.perf_counter() - started)
            break

    train_batch_seconds = train_batch_times[0]
    eval_batch_seconds = eval_batch_times[0]
    train_examples_benchmarked = min(len(tiny_train), config["batch_size"])
    validation_examples_benchmarked = min(
        len(tiny_validation), config["batch_size"]
    )
    full_train_batches = math.ceil(train_rows / config["batch_size"])
    full_validation_batches = math.ceil(validation_rows / config["batch_size"])
    estimated_train_seconds = train_batch_seconds * full_train_batches
    estimated_validation_seconds = eval_batch_seconds * full_validation_batches
    estimated_epoch_seconds = estimated_train_seconds + estimated_validation_seconds
    epoch_count = config["epochs"]
    total_estimated_seconds = estimated_epoch_seconds * epoch_count
    memory_after = psutil.virtual_memory().available

    print("CONDITION A - GOLD ONLY COST ESTIMATE (NO FULL TRAINING)")
    print(f"Model: {config['model_name']}")
    print(f"Total parameters: {total_parameters:,}")
    print(f"Trainable parameters: {trainable_parameters:,}")
    print(f"Parameter-only checkpoint estimate: {parameter_checkpoint_mib:.1f} MiB")
    print(f"State-dict checkpoint estimate: {state_checkpoint_mib:.1f} MiB")
    print("Optimizer: torch.optim.AdamW")
    print(f"Optimizer state after one step: {'initialized' if optimizer_state_initialized else 'not initialized'}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Gradient accumulation: {config['gradient_accumulation_steps']}")
    print(f"Effective batch size: {effective_batch_size}")
    print(f"Max sequence length: {config['max_length']}")
    print(f"Train examples: {train_rows:,}")
    print(f"Optimizer steps per epoch: {optimizer_steps_per_epoch:,}")
    print(f"Train microbatches per epoch: {full_train_batches:,}")
    print(f"Measured one train batch (forward/backward/AdamW): {train_batch_seconds:.2f} sec")
    print(f"Measured one validation batch (forward): {eval_batch_seconds:.2f} sec")
    print(f"Estimated full train epoch: {estimated_train_seconds / 3600:.2f} hours")
    print(f"Estimated validation pass: {estimated_validation_seconds / 3600:.2f} hours")
    print(f"Estimated total for {epoch_count} epoch(s): {total_estimated_seconds / 3600:.2f} hours")
    print(f"Estimated AdamW state: {estimated_adamw_state_gib:.2f} GiB")
    print(
        "Approximate parameter+gradient+AdamW-state memory before activations: "
        f"{estimated_training_resident_gib:.2f} GiB"
    )
    print(
        f"Available RAM before benchmark: {memory_before / (1024**3):.2f} GiB; "
        f"after benchmark: {memory_after / (1024**3):.2f} GiB"
    )
    print(
        "Runtime projection is a rough linear extrapolation from one full-batch CPU train "
        "step and one validation batch; it is hardware/load dependent and not a training result."
    )
    print("Benchmark batches: 1 train + 1 validation")
    print(f"Training examples benchmarked: {train_examples_benchmarked}")
    print(f"Validation examples benchmarked: {validation_examples_benchmarked}")
    print("Full-epoch training started: no")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
