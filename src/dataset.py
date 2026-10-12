"""Tokenized PyTorch dataset wrapper for prepared GoEmotions V1 VAD Parquet."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data_preparation import EXPECTED_LABEL_MAPPING


VAD_COLUMNS = ("vad_valence", "vad_arousal", "vad_dominance")


class GoEmotionsVADDataset(Dataset):
    """Tokenize rows while preserving every GoEmotions label.

    ``emotion_labels`` and ``emotion_targets`` are identical 28-dimensional
    binary multi-hot targets for BCEWithLogitsLoss. Values are never normalized
    across positive labels. VAD values are V1 lexicon-derived pseudo-targets.
    """

    def __init__(
        self,
        data: pd.DataFrame | str | Path,
        tokenizer: Any,
        *,
        max_length: int = 128,
    ) -> None:
        if max_length < 2:
            raise ValueError("max_length must be at least 2.")
        self.frame = (
            pd.read_parquet(data).reset_index(drop=True)
            if isinstance(data, (str, Path))
            else data.reset_index(drop=True).copy()
        )
        required = {"text", "labels", *VAD_COLUMNS}
        missing = required.difference(self.frame.columns)
        if missing:
            raise ValueError(f"Prepared data is missing columns: {sorted(missing)}")

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encodings = tokenizer(
            self.frame["text"].fillna("").astype(str).tolist(),
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

        label_targets = np.zeros((len(self.frame), len(EXPECTED_LABEL_MAPPING)), dtype=np.float32)
        for row_index, labels in enumerate(self.frame["labels"]):
            normalized_labels = [int(label) for label in labels]
            if not normalized_labels:
                raise ValueError(f"Row {row_index} has no GoEmotions label.")
            if len(set(normalized_labels)) != len(normalized_labels):
                raise ValueError(f"Row {row_index} contains duplicate emotion labels.")
            if any(label < 0 or label >= len(EXPECTED_LABEL_MAPPING) for label in normalized_labels):
                raise ValueError(f"Row {row_index} has a label outside the original 0-27 range.")
            label_targets[row_index, normalized_labels] = 1.0
        raw_vad = self.frame.loc[:, VAD_COLUMNS].to_numpy(dtype=np.float32)
        vad_valid = np.isfinite(raw_vad).all(axis=1)
        vad_targets = np.where(vad_valid[:, None], raw_vad, 0.0)
        self.emotion_labels = torch.from_numpy(label_targets)
        self.emotion_targets = self.emotion_labels
        self.vad_targets = torch.from_numpy(vad_targets.astype(np.float32, copy=False))
        self.vad_valid_mask = torch.from_numpy(vad_valid)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.encodings["input_ids"][index],
            "attention_mask": self.encodings["attention_mask"][index],
            "emotion_labels": self.emotion_labels[index],
            "emotion_targets": self.emotion_targets[index],
            "vad_targets": self.vad_targets[index],
            "vad_valid_mask": self.vad_valid_mask[index],
        }
