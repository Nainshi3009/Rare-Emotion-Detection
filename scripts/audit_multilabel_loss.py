"""Verify binary multi-hot targets and BCE/sigmoid behavior without training."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import default_collate

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import GoEmotionsVADDataset  # noqa: E402
from src.model import RobertaEmotionVADModel  # noqa: E402
from src.training import load_tokenizer, set_deterministic_seed  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "configs" / "model_resource_efficient.json"
FORMULATION_PATH = PROJECT_ROOT / "configs" / "model_training_formulation.json"
TRAIN_PATH = (
    PROJECT_ROOT / "data" / "processed" / "goemotions_vad_v1" / "train.parquet"
)


def main() -> int:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    formulation = json.loads(FORMULATION_PATH.read_text(encoding="utf-8"))
    if formulation != {
        "classification_loss": "BCEWithLogitsLoss",
        "target_encoding": "multi_hot",
        "classification_threshold": 0.5,
        "vad_loss": "masked_MSE",
        "lambda_vad": 0.5,
        "class_weighting": False,
    }:
        raise ValueError("Training formulation config does not match the selected method.")
    set_deterministic_seed(config["seed"])
    torch.set_num_threads(2)

    frame = pd.read_parquet(TRAIN_PATH)
    if len(frame) != 43_410 or not frame["split"].eq("train").all():
        raise ValueError("Expected the original 43,410-row GoEmotions train split.")
    label_counts = frame["labels"].map(len)
    multilabel_indices = frame.index[label_counts > 1].tolist()
    singlelabel_indices = frame.index[label_counts == 1].tolist()
    if not multilabel_indices or not singlelabel_indices:
        raise RuntimeError("Expected both multi-label and single-label training rows.")

    sample = frame.loc[
        [multilabel_indices[0], singlelabel_indices[0]]
    ].reset_index(drop=True)
    tokenizer = load_tokenizer(config["model_name"], local_files_only=True)
    dataset = GoEmotionsVADDataset(
        sample,
        tokenizer,
        max_length=config["max_length"],
    )
    batch = default_collate([dataset[index] for index in range(len(dataset))])

    targets = batch["emotion_targets"]
    if targets.shape != (len(dataset), 28):
        raise AssertionError(f"Unexpected target shape: {tuple(targets.shape)}")
    if not torch.all((targets == 0) | (targets == 1)):
        raise AssertionError("Targets must contain exactly binary 0/1 values.")
    if not torch.equal(targets, batch["emotion_labels"]):
        raise AssertionError("Emotion targets must preserve unnormalized multi-hot labels.")
    label_counts_in_batch = targets.sum(dim=1)
    if int(label_counts_in_batch[0]) < 2 or int(label_counts_in_batch[1]) != 1:
        raise AssertionError("Audit sample does not include multi-label and single-label rows.")

    model = RobertaEmotionVADModel(
        model_name=config["model_name"],
        num_labels=28,
        lambda_vad=formulation["lambda_vad"],
        freeze_strategy=config["freeze_strategy"],
        local_files_only=True,
    )
    if not isinstance(model.classification_loss_fn, torch.nn.BCEWithLogitsLoss):
        raise AssertionError("Model classification loss is not BCEWithLogitsLoss.")
    if model.classification_loss_fn.weight is not None:
        raise AssertionError("Class weighting must be disabled.")
    if model.classification_loss_fn.pos_weight is not None:
        raise AssertionError("Positive class weighting must be disabled.")

    model.eval()
    with torch.inference_mode():
        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            emotion_targets=targets,
        )
        probabilities = torch.sigmoid(output.emotion_logits)

    expected_shape = (len(dataset), 28)
    if output.emotion_logits.shape != expected_shape:
        raise AssertionError(f"Unexpected logits shape: {tuple(output.emotion_logits.shape)}")
    if probabilities.shape != expected_shape:
        raise AssertionError(f"Unexpected sigmoid prediction shape: {tuple(probabilities.shape)}")
    if output.classification_loss is None or not torch.isfinite(output.classification_loss):
        raise AssertionError("BCEWithLogitsLoss did not produce a finite loss.")

    print("MULTI-LABEL FORMULATION AUDIT — INFERENCE ONLY; NO TRAINING")
    print("Target shape:", tuple(targets.shape))
    print("Target values are exclusively 0/1:", True)
    print("First original multi-hot target:", targets[0].tolist())
    print("Number of positive labels per example:", label_counts_in_batch.tolist())
    print("Logits shape:", tuple(output.emotion_logits.shape))
    print("Loss:", type(model.classification_loss_fn).__name__)
    print("BCEWithLogitsLoss value:", float(output.classification_loss))
    print("Sigmoid prediction shape:", tuple(probabilities.shape))
    print("Evaluation threshold:", formulation["classification_threshold"])
    print("Softmax used:", False)
    print("Class weighting enabled:", False)
    print("lambda_vad:", model.lambda_vad)
    print("Backward pass/optimizer step performed: no")
    print("Full or partial training performed: no")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
