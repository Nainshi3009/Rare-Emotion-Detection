"""Verify the configured RoBERTa final-two-layer freeze policy."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import RobertaEmotionVADModel  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "configs" / "model_resource_efficient.json"


def main() -> int:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("freeze_strategy") != "last_2_layers":
        raise ValueError("Expected freeze_strategy='last_2_layers'.")

    model = RobertaEmotionVADModel(
        model_name=config["model_name"],
        lambda_vad=config["lambda_vad"],
        freeze_strategy=config["freeze_strategy"],
        local_files_only=True,
    )
    layers = model.encoder.encoder.layer
    if len(layers) != 12:
        raise RuntimeError(
            f"Expected roberta-base to contain 12 transformer layers; found {len(layers)}."
        )

    unexpected_trainable = []
    missing_trainable = []
    for name, parameter in model.named_parameters():
        if name.startswith("encoder.embeddings."):
            expected = False
        elif name.startswith("encoder.encoder.layer."):
            layer_index = int(name.split(".")[3])
            expected = layer_index in {10, 11}
        elif name.startswith(("emotion_head.", "vad_head.")):
            expected = True
        else:
            expected = False

        if parameter.requires_grad and not expected:
            unexpected_trainable.append(name)
        if expected and not parameter.requires_grad:
            missing_trainable.append(name)

    if unexpected_trainable or missing_trainable:
        raise AssertionError(
            "Freeze policy mismatch. "
            f"Unexpected trainable: {unexpected_trainable}; "
            f"expected but frozen: {missing_trainable}"
        )

    report = model.parameter_report()
    print(f"Model: {config['model_name']}")
    print(f"Freeze strategy: {model.freeze_strategy}")
    print(f"Transformer layers: {len(layers)} (indices 0-11)")
    print("Trainable layers: encoder.encoder.layer.10 and encoder.encoder.layer.11")
    print("Trainable task heads: emotion_head and vad_head")
    print("Trainable parameter groups:")
    for group, parameter_count in report["trainable_parameter_groups"].items():
        print(f"  {group}: {parameter_count:,}")
    print(f"Total parameters: {report['total_parameters']:,}")
    print(f"Trainable parameters: {report['trainable_parameters']:,}")
    print(f"Frozen parameters: {report['frozen_parameters']:,}")
    print(f"Percentage trainable: {report['percentage_trainable']:.2f}%")
    print("Freeze configuration verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
