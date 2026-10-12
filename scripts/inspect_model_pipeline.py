from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preparation import EXPECTED_LABEL_MAPPING  # noqa: E402
from src.dataset import GoEmotionsVADDataset  # noqa: E402
from src.model import RobertaEmotionVADModel  # noqa: E402
from src.training import load_tokenizer  # noqa: E402


def main() -> int:
    prepared_dir = PROJECT_ROOT / "data" / "processed" / "goemotions_vad_v1"
    metadata = json.loads((prepared_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata["synthetic_files_sha256_before"] != metadata["synthetic_files_sha256_after"]:
        raise AssertionError("Synthetic files were changed since VAD-target preparation.")

    validation = __import__("pandas").read_parquet(prepared_dir / "validation.parquet").head(2)
    tokenizer = load_tokenizer("roberta-base")
    print(f"Tokenizer loaded: {tokenizer.__class__.__name__}")
    model = RobertaEmotionVADModel("roberta-base", num_labels=28, lambda_vad=1.0)
    model.eval()
    print(f"Model initialized: {model.__class__.__name__}")
    print(f"Emotion labels: {model.num_labels}")
    print(f"VAD output dimension: {model.vad_head.out_features}")

    dataset = GoEmotionsVADDataset(validation, tokenizer, max_length=128)
    batch = {key: value.unsqueeze(0) for key, value in dataset[0].items()}
    with torch.inference_mode():
        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            emotion_targets=batch["emotion_targets"],
            vad_targets=batch["vad_targets"],
            vad_valid_mask=batch["vad_valid_mask"],
        )
    expected_logits = (1, len(EXPECTED_LABEL_MAPPING))
    expected_vad = (1, 3)
    print(f"Input IDs shape: {tuple(batch['input_ids'].shape)}")
    print(f"Emotion logits shape: {tuple(output.emotion_logits.shape)}")
    print(f"VAD output shape: {tuple(output.vad_predictions.shape)}")
    print(f"Classification loss: {float(output.classification_loss):.6f}")
    print(f"VAD loss: {float(output.vad_loss):.6f}")
    print(f"Combined loss: {float(output.loss):.6f}")
    if tuple(output.emotion_logits.shape) != expected_logits:
        raise AssertionError(f"Expected logits shape {expected_logits}.")
    if tuple(output.vad_predictions.shape) != expected_vad:
        raise AssertionError(f"Expected VAD shape {expected_vad}.")
    if output.loss is None or not torch.isfinite(output.loss):
        raise AssertionError("Joint loss is missing or non-finite.")
    print("Pipeline inspection passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
