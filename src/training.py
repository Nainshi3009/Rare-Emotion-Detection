"""CPU-compatible training and evaluation for the joint emotion/VAD model."""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.data_preparation import EXPECTED_LABEL_MAPPING, TARGET_LABEL_IDS
from src.dataset import GoEmotionsVADDataset
from src.model import RobertaEmotionVADModel


@dataclass
class TrainingConfig:
    model_name: str = "roberta-base"
    freeze_strategy: str = "full"
    max_length: int = 128
    batch_size: int = 2
    epochs: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lambda_vad: float = 1.0
    gradient_accumulation_steps: int = 1
    max_train_steps: int | None = None
    seed: int = 20261008
    num_workers: int = 0
    classification_threshold: float = 0.5
    cpu_num_threads: int = 4


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_tokenizer(
    model_name: str = "roberta-base",
    *,
    local_files_only: bool = False,
):
    return AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        local_files_only=local_files_only,
    )


def _metrics_from_outputs(
    targets: np.ndarray,
    probabilities: np.ndarray,
    vad_targets: np.ndarray,
    vad_predictions: np.ndarray,
    vad_valid_mask: np.ndarray,
    *,
    threshold: float,
) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(np.int32)
    per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
        targets,
        predictions,
        average=None,
        labels=list(range(len(EXPECTED_LABEL_MAPPING))),
        zero_division=0,
    )
    valid_vad = vad_valid_mask.astype(bool)
    vad_mae = (
        np.abs(vad_predictions[valid_vad] - vad_targets[valid_vad]).mean(axis=0)
        if valid_vad.any()
        else np.array([np.nan, np.nan, np.nan])
    )
    return {
        "macro_f1": float(
            f1_score(targets, predictions, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(targets, predictions, average="weighted", zero_division=0)
        ),
        "per_class_f1": {
            name: float(per_class_f1[index])
            for index, name in enumerate(EXPECTED_LABEL_MAPPING)
        },
        "per_class_precision": {
            name: float(per_class_precision[index])
            for index, name in enumerate(EXPECTED_LABEL_MAPPING)
        },
        "per_class_recall": {
            name: float(per_class_recall[index])
            for index, name in enumerate(EXPECTED_LABEL_MAPPING)
        },
        "rare_target_f1": {
            emotion: float(per_class_f1[label_id])
            for emotion, label_id in TARGET_LABEL_IDS.items()
        },
        "vad_mae": {
            "valence": float(vad_mae[0]),
            "arousal": float(vad_mae[1]),
            "dominance": float(vad_mae[2]),
            "valid_rows": int(valid_vad.sum()),
        },
        "evaluated_rows": int(targets.shape[0]),
        "classification_threshold": threshold,
        "classification_metric_note": (
            "Independent multi-label predictions: sigmoid(logits) thresholded at the configured "
            "threshold; macro/weighted F1 average the original 28 label-wise F1 values."
        ),
        "vad_metric_note": "MAE against NRC-VAD V1 lexicon-derived pseudo-targets, not human ground truth.",
    }


def evaluate_model(
    model: RobertaEmotionVADModel,
    dataloader: DataLoader,
    *,
    device: torch.device | str = "cpu",
    classification_threshold: float = 0.5,
) -> dict[str, Any]:
    device = torch.device(device)
    model.eval()
    all_targets = []
    all_probabilities = []
    all_vad_targets = []
    all_vad_predictions = []
    all_vad_masks = []
    loss_sums = {"total_loss": 0.0, "classification_loss": 0.0, "vad_loss": 0.0}
    loss_batches = 0

    with torch.inference_mode():
        for batch in dataloader:
            tensors = {
                key: value.to(device)
                for key, value in batch.items()
            }
            output = model(
                input_ids=tensors["input_ids"],
                attention_mask=tensors["attention_mask"],
                emotion_targets=tensors["emotion_targets"],
                vad_targets=tensors["vad_targets"],
                vad_valid_mask=tensors["vad_valid_mask"],
            )
            if output.loss is not None:
                loss_sums["total_loss"] += float(output.loss)
                loss_sums["classification_loss"] += float(output.classification_loss)
                loss_sums["vad_loss"] += float(output.vad_loss)
                loss_batches += 1
            all_targets.append(tensors["emotion_labels"].cpu().numpy())
            all_probabilities.append(torch.sigmoid(output.emotion_logits).cpu().numpy())
            all_vad_targets.append(tensors["vad_targets"].cpu().numpy())
            all_vad_predictions.append(output.vad_predictions.cpu().numpy())
            all_vad_masks.append(tensors["vad_valid_mask"].cpu().numpy())

    if not all_targets:
        raise ValueError("Cannot evaluate an empty dataset.")
    metrics = _metrics_from_outputs(
        np.concatenate(all_targets),
        np.concatenate(all_probabilities),
        np.concatenate(all_vad_targets),
        np.concatenate(all_vad_predictions),
        np.concatenate(all_vad_masks),
        threshold=classification_threshold,
    )
    metrics["loss"] = {
        key: value / loss_batches if loss_batches else None
        for key, value in loss_sums.items()
    }
    return metrics


def save_checkpoint(
    model: RobertaEmotionVADModel,
    tokenizer: Any,
    path: str | Path,
    *,
    config: TrainingConfig,
    epoch: int,
    validation_macro_f1: float,
) -> Path:
    checkpoint_path = Path(path).resolve()
    if ".venv" in checkpoint_path.parts:
        raise ValueError("Model checkpoints must not be saved inside .venv.")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "model_name": model.model_name,
        "num_labels": model.num_labels,
        "lambda_vad": model.lambda_vad,
        "label_mapping": list(EXPECTED_LABEL_MAPPING),
        "epoch": epoch,
        "validation_macro_f1": validation_macro_f1,
        "training_config": asdict(config),
        "vad_target_interpretation": (
            "NRC-VAD V1 lexicon-derived pseudo-targets; not human-annotated ground truth."
        ),
    }
    temporary_path = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, checkpoint_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    if tokenizer is not None:
        tokenizer.save_pretrained(checkpoint_path.parent / "tokenizer")
    return checkpoint_path


def load_checkpoint(
    model: RobertaEmotionVADModel,
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if checkpoint.get("label_mapping") != list(EXPECTED_LABEL_MAPPING):
        raise ValueError("Checkpoint label mapping differs from GoEmotions.")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return checkpoint


RECOVERY_CHECKPOINT_VERSION = 1


def _frame_fingerprint(frame) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(list(frame.columns), ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    for row in frame.itertuples(index=False, name=None):
        digest.update(
            json.dumps(
                row,
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _capture_rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [cuda_state.cpu() for cuda_state in state["cuda"]]
        )


def _save_recovery_checkpoint(
    path: str | Path,
    model: RobertaEmotionVADModel,
    optimizer: torch.optim.Optimizer,
    scheduler,
    *,
    config: TrainingConfig,
    train_fingerprint: str,
    validation_fingerprint: str,
    epoch_index: int,
    next_microbatch_position: int,
    global_optimizer_step: int,
    best_macro_f1: float,
    best_checkpoint: str | None,
    epoch_loader_generator_state: torch.Tensor,
    epoch_loss: float,
    epoch_classification_loss: float,
    epoch_vad_loss: float,
    epoch_microbatches: int,
    accumulation_position: int,
    history: list[dict[str, Any]],
    generator: torch.Generator,
) -> Path:
    checkpoint_path = Path(path).resolve()
    if ".venv" in checkpoint_path.parts:
        raise ValueError("Recovery checkpoints must not be saved inside .venv.")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    named_parameters = dict(model.named_parameters())
    payload = {
        "format_version": RECOVERY_CHECKPOINT_VERSION,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch_index": epoch_index,
        "next_microbatch_position": next_microbatch_position,
        "global_optimizer_step": global_optimizer_step,
        "best_validation_macro_f1": best_macro_f1,
        "best_checkpoint": best_checkpoint,
        "training_config": asdict(config),
        "train_fingerprint": train_fingerprint,
        "validation_fingerprint": validation_fingerprint,
        "epoch_loader_generator_state": epoch_loader_generator_state,
        "loader_generator_state": generator.get_state(),
        "epoch_loss": epoch_loss,
        "epoch_classification_loss": epoch_classification_loss,
        "epoch_vad_loss": epoch_vad_loss,
        "epoch_microbatches": epoch_microbatches,
        "accumulation_position": accumulation_position,
        "history": history,
        "gradients": {
            name: parameter.grad.detach().cpu()
            for name, parameter in named_parameters.items()
            if parameter.grad is not None
        },
        "rng_state": _capture_rng_state(),
    }
    temporary_path = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, checkpoint_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return checkpoint_path


def _load_recovery_checkpoint(
    path: str | Path,
    model: RobertaEmotionVADModel,
    optimizer: torch.optim.Optimizer,
    scheduler,
    *,
    config: TrainingConfig,
    train_fingerprint: str,
    validation_fingerprint: str,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("format_version") != RECOVERY_CHECKPOINT_VERSION:
        raise ValueError("Unsupported recovery checkpoint format.")
    if checkpoint.get("training_config") != asdict(config):
        raise ValueError("Recovery checkpoint training configuration does not match.")
    if checkpoint.get("train_fingerprint") != train_fingerprint:
        raise ValueError("Recovery checkpoint training data does not match.")
    if checkpoint.get("validation_fingerprint") != validation_fingerprint:
        raise ValueError("Recovery checkpoint validation data does not match.")

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    named_parameters = dict(model.named_parameters())
    for name, gradient in checkpoint["gradients"].items():
        if name not in named_parameters:
            raise ValueError(f"Recovery checkpoint contains an unknown gradient: {name}")
        named_parameters[name].grad = gradient.to(device)
    _restore_rng_state(checkpoint["rng_state"])
    return checkpoint


def train_model(
    model: RobertaEmotionVADModel,
    tokenizer: Any,
    train_frame,
    validation_frame,
    *,
    config: TrainingConfig | None = None,
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
    experiment_name: str = "training",
    recovery_checkpoint_path: str | Path | None = None,
    resume_from_recovery: bool = False,
    recovery_interval_steps: int = 100,
) -> dict[str, Any]:
    """Train and retain the checkpoint with best validation macro-F1."""
    config = config or TrainingConfig()
    if config.epochs < 1 or config.batch_size < 1 or config.gradient_accumulation_steps < 1:
        raise ValueError("epochs, batch_size, and gradient_accumulation_steps must be positive.")
    if not 0.0 <= config.warmup_ratio < 1.0:
        raise ValueError("warmup_ratio must be between 0.0 (inclusive) and 1.0 (exclusive).")
    if recovery_interval_steps < 1:
        raise ValueError("recovery_interval_steps must be positive.")
    if resume_from_recovery and recovery_checkpoint_path is None:
        raise ValueError("A recovery checkpoint path is required to resume.")
    set_deterministic_seed(config.seed)
    torch.set_num_threads(max(1, config.cpu_num_threads))
    model.set_freeze_strategy(config.freeze_strategy)
    device = torch.device(device)
    if device.type != "cpu" and not torch.cuda.is_available():
        raise ValueError(f"Requested device {device} is unavailable.")
    model.to(device)

    print(
        f"[{experiment_name}] Preparing tokenized train and validation datasets "
        f"(max_length={config.max_length}).",
        flush=True,
    )
    train_dataset = GoEmotionsVADDataset(
        train_frame,
        tokenizer,
        max_length=config.max_length,
    )
    validation_dataset = GoEmotionsVADDataset(
        validation_frame,
        tokenizer,
        max_length=config.max_length,
    )
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=config.num_workers,
        pin_memory=False,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=False,
    )

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    microbatches_per_epoch = len(train_loader)
    optimizer_steps_per_epoch = (
        microbatches_per_epoch + config.gradient_accumulation_steps - 1
    ) // config.gradient_accumulation_steps
    if config.max_train_steps is not None:
        optimizer_steps_per_epoch = min(optimizer_steps_per_epoch, config.max_train_steps)
    total_optimizer_steps = optimizer_steps_per_epoch * config.epochs
    warmup_steps = int(total_optimizer_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(1, total_optimizer_steps),
    )
    recovery_path = (
        Path(recovery_checkpoint_path).resolve()
        if recovery_checkpoint_path is not None
        else None
    )
    if resume_from_recovery and (recovery_path is None or not recovery_path.is_file()):
        raise FileNotFoundError(f"Recovery checkpoint not found: {recovery_path}")
    train_fingerprint = (
        _frame_fingerprint(train_frame) if recovery_path is not None else ""
    )
    validation_fingerprint = (
        _frame_fingerprint(validation_frame) if recovery_path is not None else ""
    )

    best_macro_f1 = float("-inf")
    best_checkpoint = None
    history: list[dict[str, Any]] = []
    global_optimizer_step = 0
    start_epoch_index = 0
    resume_state = None
    if resume_from_recovery:
        resume_state = _load_recovery_checkpoint(
            recovery_path,
            model,
            optimizer,
            scheduler,
            config=config,
            train_fingerprint=train_fingerprint,
            validation_fingerprint=validation_fingerprint,
            device=device,
        )
        start_epoch_index = int(resume_state["epoch_index"])
        best_macro_f1 = float(resume_state["best_validation_macro_f1"])
        best_checkpoint = resume_state["best_checkpoint"]
        if best_checkpoint is not None:
            if Path(best_checkpoint).resolve() != Path(checkpoint_path).resolve():
                raise ValueError(
                    "Recovery checkpoint best-model path does not match this run."
                )
            if not Path(best_checkpoint).is_file():
                raise FileNotFoundError(
                    f"Best-validation checkpoint referenced by recovery state is missing: "
                    f"{best_checkpoint}"
                )
        global_optimizer_step = int(resume_state["global_optimizer_step"])
        history = resume_state["history"]
        generator.set_state(resume_state["loader_generator_state"])

    if resume_state is None:
        optimizer.zero_grad(set_to_none=True)

    started = time.perf_counter()
    print(
        f"[{experiment_name}] {'Resuming' if resume_from_recovery else 'Training'}: "
        f"epochs={config.epochs}, "
        f"total_optimizer_steps={total_optimizer_steps}, "
        f"batch_size={config.batch_size}, "
        f"gradient_accumulation_steps={config.gradient_accumulation_steps}, "
        f"global_optimizer_step={global_optimizer_step}",
        flush=True,
    )

    first_training_batch_logged = False
    safe_to_checkpoint = True
    recovery_epoch_index = start_epoch_index
    recovery_batch_position = 0
    recovery_epoch_loader_state = generator.get_state()
    recovery_epoch_loss = 0.0
    recovery_epoch_cls_loss = 0.0
    recovery_epoch_vad_loss = 0.0
    recovery_epoch_microbatches = 0
    recovery_accumulation_position = 0

    def save_recovery(
        *,
        epoch_index: int,
        next_microbatch_position: int,
        epoch_loader_state: torch.Tensor,
        epoch_loss: float,
        epoch_cls_loss: float,
        epoch_vad_loss: float,
        epoch_microbatches: int,
        accumulation_position: int,
    ) -> None:
        if recovery_path is None:
            return
        display_epoch = min(epoch_index + 1, config.epochs)
        print(
            f"[{experiment_name}] Saving recovery checkpoint: "
            f"epoch={display_epoch}/{config.epochs}, "
            f"microbatch={next_microbatch_position}/{len(train_loader)}, "
            f"optimizer_step={global_optimizer_step}/{total_optimizer_steps}, "
            f"path={recovery_path}.",
            flush=True,
        )
        _save_recovery_checkpoint(
            recovery_path,
            model,
            optimizer,
            scheduler,
            config=config,
            train_fingerprint=train_fingerprint,
            validation_fingerprint=validation_fingerprint,
            epoch_index=epoch_index,
            next_microbatch_position=next_microbatch_position,
            global_optimizer_step=global_optimizer_step,
            best_macro_f1=best_macro_f1,
            best_checkpoint=str(best_checkpoint) if best_checkpoint else None,
            epoch_loader_generator_state=epoch_loader_state,
            epoch_loss=epoch_loss,
            epoch_classification_loss=epoch_cls_loss,
            epoch_vad_loss=epoch_vad_loss,
            epoch_microbatches=epoch_microbatches,
            accumulation_position=accumulation_position,
            history=history,
            generator=generator,
        )
        print(
            f"[{experiment_name}] Recovery checkpoint saved.",
            flush=True,
        )

    try:
        for epoch in range(start_epoch_index, config.epochs):
            is_resuming_epoch = (
                resume_state is not None
                and epoch == start_epoch_index
                and int(resume_state["epoch_index"]) < config.epochs
            )
            if is_resuming_epoch:
                recovery_epoch_loader_state = resume_state[
                    "epoch_loader_generator_state"
                ]
                generator.set_state(recovery_epoch_loader_state)
                recovery_batch_position = int(
                    resume_state["next_microbatch_position"]
                )
                recovery_epoch_loss = float(resume_state["epoch_loss"])
                recovery_epoch_cls_loss = float(
                    resume_state["epoch_classification_loss"]
                )
                recovery_epoch_vad_loss = float(resume_state["epoch_vad_loss"])
                recovery_epoch_microbatches = int(
                    resume_state["epoch_microbatches"]
                )
                recovery_accumulation_position = int(
                    resume_state["accumulation_position"]
                )
                optimizer_steps = global_optimizer_step - sum(
                    int(record["optimizer_steps"]) for record in history
                )
            else:
                recovery_batch_position = 0
                recovery_epoch_loss = 0.0
                recovery_epoch_cls_loss = 0.0
                recovery_epoch_vad_loss = 0.0
                recovery_epoch_microbatches = 0
                recovery_accumulation_position = 0
                optimizer_steps = 0
                optimizer.zero_grad(set_to_none=True)
                recovery_epoch_loader_state = generator.get_state()
                save_recovery(
                    epoch_index=epoch,
                    next_microbatch_position=0,
                    epoch_loader_state=recovery_epoch_loader_state,
                    epoch_loss=0.0,
                    epoch_cls_loss=0.0,
                    epoch_vad_loss=0.0,
                    epoch_microbatches=0,
                    accumulation_position=0,
                )

            recovery_epoch_index = epoch
            model.train()
            epoch_microbatch_limit = (
                min(len(train_loader), config.max_train_steps)
                if config.max_train_steps is not None
                else len(train_loader)
            )
            iterator = iter(train_loader)
            for _ in range(recovery_batch_position):
                next(iterator)

            step = recovery_batch_position
            while step < epoch_microbatch_limit:
                safe_to_checkpoint = False
                batch = next(iterator)
                step += 1
                if not first_training_batch_logged:
                    print(
                        f"[{experiment_name}] First training batch: "
                        f"epoch={epoch + 1}/{config.epochs}, "
                        f"microbatch={step}/{len(train_loader)}.",
                        flush=True,
                    )
                    first_training_batch_logged = True
                tensors = {key: value.to(device) for key, value in batch.items()}
                output = model(
                    input_ids=tensors["input_ids"],
                    attention_mask=tensors["attention_mask"],
                    emotion_targets=tensors["emotion_targets"],
                    vad_targets=tensors["vad_targets"],
                    vad_valid_mask=tensors["vad_valid_mask"],
                )
                if (
                    output.loss is None
                    or output.classification_loss is None
                    or output.vad_loss is None
                ):
                    raise RuntimeError(
                        "Joint model did not return all required loss components."
                    )
                accumulation_group_start = (
                    (step - 1) // config.gradient_accumulation_steps
                ) * config.gradient_accumulation_steps
                accumulation_group_size = min(
                    config.gradient_accumulation_steps,
                    len(train_loader) - accumulation_group_start,
                )
                if config.max_train_steps is not None:
                    accumulation_group_size = min(
                        accumulation_group_size,
                        config.max_train_steps - accumulation_group_start,
                    )
                (output.loss / accumulation_group_size).backward()
                recovery_epoch_loss += float(output.loss.detach())
                recovery_epoch_cls_loss += float(output.classification_loss.detach())
                recovery_epoch_vad_loss += float(output.vad_loss.detach())
                recovery_epoch_microbatches += 1
                recovery_accumulation_position += 1
                recovery_batch_position = step

                optimizer_updated = False
                if step % config.gradient_accumulation_steps == 0:
                    safe_to_checkpoint = False
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += 1
                    global_optimizer_step += 1
                    recovery_accumulation_position = 0
                    optimizer_updated = True
                safe_to_checkpoint = True
                if config.max_train_steps is not None and step >= config.max_train_steps:
                    break

                if optimizer_updated:
                    if (
                        global_optimizer_step % 10 == 0
                        or global_optimizer_step == total_optimizer_steps
                    ):
                        print(
                            f"[{experiment_name}] Progress: "
                            f"epoch={epoch + 1}/{config.epochs}, "
                            f"total_optimizer_steps={total_optimizer_steps}, "
                            f"microbatch={step}/{len(train_loader)}, "
                            f"optimizer_step={global_optimizer_step}/{total_optimizer_steps}, "
                            f"elapsed={time.perf_counter() - started:.1f}s, "
                            f"loss={float(output.loss.detach()):.6f}",
                            flush=True,
                        )
                    if (
                        recovery_path is not None
                        and global_optimizer_step % recovery_interval_steps == 0
                    ):
                        save_recovery(
                            epoch_index=epoch,
                            next_microbatch_position=recovery_batch_position,
                            epoch_loader_state=recovery_epoch_loader_state,
                            epoch_loss=recovery_epoch_loss,
                            epoch_cls_loss=recovery_epoch_cls_loss,
                            epoch_vad_loss=recovery_epoch_vad_loss,
                            epoch_microbatches=recovery_epoch_microbatches,
                            accumulation_position=recovery_accumulation_position,
                        )

            if recovery_accumulation_position:
                safe_to_checkpoint = False
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                global_optimizer_step += 1
                recovery_accumulation_position = 0
                safe_to_checkpoint = True

            save_recovery(
                epoch_index=epoch,
                next_microbatch_position=recovery_batch_position,
                epoch_loader_state=recovery_epoch_loader_state,
                epoch_loss=recovery_epoch_loss,
                epoch_cls_loss=recovery_epoch_cls_loss,
                epoch_vad_loss=recovery_epoch_vad_loss,
                epoch_microbatches=recovery_epoch_microbatches,
                accumulation_position=recovery_accumulation_position,
            )
            print(
                f"[{experiment_name}] Validation starting for epoch "
                f"{epoch + 1}/{config.epochs}.",
                flush=True,
            )
            validation_metrics = evaluate_model(
                model,
                validation_loader,
                device=device,
                classification_threshold=config.classification_threshold,
            )
            train_loss = {
                "total_loss": recovery_epoch_loss
                / max(1, recovery_epoch_microbatches),
                "classification_loss": recovery_epoch_cls_loss
                / max(1, recovery_epoch_microbatches),
                "vad_loss": recovery_epoch_vad_loss
                / max(1, recovery_epoch_microbatches),
            }
            epoch_record = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "validation": validation_metrics,
                "optimizer_steps": optimizer_steps,
            }
            history.append(epoch_record)

            rare_f1 = validation_metrics["rare_target_f1"]
            print(
                f"Epoch {epoch + 1}/{config.epochs} "
                f"train_loss={train_loss['total_loss']:.6f} "
                f"emotion_loss={train_loss['classification_loss']:.6f} "
                f"vad_loss={train_loss['vad_loss']:.6f} "
                f"val_macro_f1={validation_metrics['macro_f1']:.6f} "
                f"val_weighted_f1={validation_metrics['weighted_f1']:.6f} "
                f"rare_f1={rare_f1}",
                flush=True,
            )

            if validation_metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = validation_metrics["macro_f1"]
                print(
                    f"[{experiment_name}] Saving best checkpoint for epoch "
                    f"{epoch + 1}/{config.epochs} to {checkpoint_path}.",
                    flush=True,
                )
                best_checkpoint = save_checkpoint(
                    model,
                    tokenizer,
                    checkpoint_path,
                    config=config,
                    epoch=epoch + 1,
                    validation_macro_f1=best_macro_f1,
                )

            recovery_epoch_index = epoch + 1
            recovery_batch_position = 0
            recovery_epoch_loss = 0.0
            recovery_epoch_cls_loss = 0.0
            recovery_epoch_vad_loss = 0.0
            recovery_epoch_microbatches = 0
            recovery_accumulation_position = 0
            recovery_epoch_loader_state = generator.get_state()
            safe_to_checkpoint = True
            save_recovery(
                epoch_index=recovery_epoch_index,
                next_microbatch_position=0,
                epoch_loader_state=recovery_epoch_loader_state,
                epoch_loss=0.0,
                epoch_cls_loss=0.0,
                epoch_vad_loss=0.0,
                epoch_microbatches=0,
                accumulation_position=0,
            )
            resume_state = None
    except KeyboardInterrupt:
        if recovery_path is not None and safe_to_checkpoint:
            save_recovery(
                epoch_index=recovery_epoch_index,
                next_microbatch_position=recovery_batch_position,
                epoch_loader_state=recovery_epoch_loader_state,
                epoch_loss=recovery_epoch_loss,
                epoch_cls_loss=recovery_epoch_cls_loss,
                epoch_vad_loss=recovery_epoch_vad_loss,
                epoch_microbatches=recovery_epoch_microbatches,
                accumulation_position=recovery_accumulation_position,
            )
            print(
                f"[{experiment_name}] Recovery checkpoint saved to {recovery_path}.",
                flush=True,
            )
        elif recovery_path is not None:
            print(
                f"[{experiment_name}] Interrupted during an unsafe update; "
                f"the last atomic recovery checkpoint remains at {recovery_path}.",
                flush=True,
            )
        raise

    return {
        "history": history,
        "best_validation_macro_f1": best_macro_f1,
        "best_checkpoint": str(best_checkpoint) if best_checkpoint else None,
        "runtime_seconds": time.perf_counter() - started,
        "device": str(device),
        "train_rows": len(train_dataset),
        "validation_rows": len(validation_dataset),
        "training_config": asdict(config),
        "global_optimizer_step": global_optimizer_step,
        "recovery_checkpoint": str(recovery_path) if recovery_path else None,
        "resumed_from_recovery": resume_from_recovery,
    }
