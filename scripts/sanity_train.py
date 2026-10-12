from __future__ import annotations

import hashlib
import json
import random
import sys
import time
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preparation import TARGET_LABEL_IDS  # noqa: E402
from src.model import RobertaEmotionVADModel  # noqa: E402
from src.training import (  # noqa: E402
    TrainingConfig,
    load_checkpoint,
    load_tokenizer,
    train_model,
)


PREPARED_DIRECTORY = PROJECT_ROOT / "data" / "processed" / "goemotions_vad_v1"
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "sanity_check" / "best_checkpoint.pt"
PROTECTED_FILES = (
    PROJECT_ROOT / "Rare_Emotion_Detection_EDA.ipynb",
    PROJECT_ROOT / "data" / "synthetic" / "vad_filtered_candidates_40.csv",
    PROJECT_ROOT / "data" / "synthetic" / "vad_filtered_candidates_40.metadata.json",
    PROJECT_ROOT / "data" / "synthetic" / "vad_filtered_candidates_batch2.csv",
    PROJECT_ROOT / "data" / "synthetic" / "vad_filtered_candidates_batch2.metadata.json",
    PREPARED_DIRECTORY / "train.parquet",
    PREPARED_DIRECTORY / "validation.parquet",
    PREPARED_DIRECTORY / "test.parquet",
    PREPARED_DIRECTORY / "metadata.json",
)


def _fingerprints() -> dict[str, str]:
    return {
        str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in PROTECTED_FILES
    }


def _subset_with_target_examples(
    frame: pd.DataFrame,
    *,
    size: int,
    seed: int,
    minimum_per_target: int,
) -> pd.DataFrame:
    rng = random.Random(seed)
    selected = []
    selected_indices = set()
    for emotion, label_id in TARGET_LABEL_IDS.items():
        positions = [
            index
            for index, labels in enumerate(frame["labels"])
            if label_id in labels
        ]
        rng.shuffle(positions)
        for index in positions[:minimum_per_target]:
            if index not in selected_indices:
                selected.append(index)
                selected_indices.add(index)
    remaining = [index for index in range(len(frame)) if index not in selected_indices]
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, size - len(selected))])
    return frame.iloc[selected].reset_index(drop=True)


def main() -> int:
    print("SANITY CHECK ONLY — NOT A FINAL EXPERIMENTAL RESULT")
    started = time.perf_counter()
    protected_before = _fingerprints()
    train_frame = pd.read_parquet(PREPARED_DIRECTORY / "train.parquet")
    validation_frame = pd.read_parquet(PREPARED_DIRECTORY / "validation.parquet")
    train_small = _subset_with_target_examples(
        train_frame,
        size=500,
        seed=20261008,
        minimum_per_target=15,
    )
    validation_small = _subset_with_target_examples(
        validation_frame,
        size=200,
        seed=20261009,
        minimum_per_target=5,
    )
    tokenizer = load_tokenizer("roberta-base")
    config = TrainingConfig(
        model_name="roberta-base",
        max_length=128,
        batch_size=2,
        epochs=1,
        learning_rate=2e-5,
        lambda_vad=1.0,
        gradient_accumulation_steps=1,
        max_train_steps=2,
        seed=20261008,
        num_workers=0,
        cpu_num_threads=2,
    )
    model = RobertaEmotionVADModel(
        model_name="roberta-base",
        num_labels=28,
        lambda_vad=config.lambda_vad,
        local_files_only=True,
    )
    result = train_model(
        model,
        tokenizer,
        train_small,
        validation_small,
        config=config,
        checkpoint_path=CHECKPOINT_PATH,
        device="cpu",
    )

    saved = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    reloaded_model = RobertaEmotionVADModel(
        model_name="roberta-base",
        num_labels=28,
        lambda_vad=config.lambda_vad,
        local_files_only=True,
    )
    loaded = load_checkpoint(reloaded_model, CHECKPOINT_PATH)
    checkpoint_roundtrip = (
        saved["epoch"] == loaded["epoch"]
        and saved["validation_macro_f1"] == loaded["validation_macro_f1"]
        and all(
            torch.equal(value, reloaded_model.state_dict()[key])
            for key, value in saved["model_state_dict"].items()
        )
    )
    if not checkpoint_roundtrip:
        raise AssertionError("Checkpoint save/load roundtrip failed.")

    protected_after = _fingerprints()
    if protected_before != protected_after:
        raise AssertionError("A protected notebook, prepared split, or synthetic file changed.")
    if result["best_checkpoint"] is None:
        raise AssertionError("No best checkpoint was saved.")

    print("SANITY CHECK ONLY — NOT A FINAL EXPERIMENTAL RESULT")
    print(f"Training subset rows: {len(train_small)}")
    print(f"Validation subset rows: {len(validation_small)}")
    print(f"Train steps: {config.max_train_steps} (bounded sanity pass)")
    print(f"Best validation macro-F1 (verification only): {result['best_validation_macro_f1']:.6f}")
    print(f"Checkpoint save/load: {'passed' if checkpoint_roundtrip else 'failed'}")
    print(f"Checkpoint path: {CHECKPOINT_PATH}")
    print(f"Runtime seconds: {time.perf_counter() - started:.2f}")
    print("Protected notebook, prepared Parquet, and Batch 1/Batch 2 hashes unchanged: yes")
    print("No synthetic condition was loaded or trained.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
