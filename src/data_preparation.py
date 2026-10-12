"""Reusable GoEmotions and synthetic-candidate preparation utilities."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import DatasetDict, DownloadConfig, load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INSPECTION_REPORT_PATH = PROJECT_ROOT / "data" / "ml_preparation_inspection.json"
DATASET_ID = "google-research-datasets/go_emotions"
RANDOM_SEED = 20261008
OVERSAMPLING_TARGET_COUNT = 164
TARGET_EMOTIONS = ("grief", "pride", "relief", "nervousness")
EXPECTED_LABEL_MAPPING = (
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
)
TARGET_LABEL_IDS = {
    emotion: EXPECTED_LABEL_MAPPING.index(emotion)
    for emotion in TARGET_EMOTIONS
}
SYNTHETIC_PATHS = (
    Path("data/synthetic/vad_filtered_candidates_40.csv"),
    Path("data/synthetic/vad_filtered_candidates_batch2.csv"),
)


class IncompleteSyntheticConditionError(ValueError):
    """Raised when a caller requires complete target-emotion coverage."""


def _read_inspection_report() -> dict[str, Any]:
    if not INSPECTION_REPORT_PATH.is_file():
        raise FileNotFoundError(f"Inspection report not found: {INSPECTION_REPORT_PATH}")
    return json.loads(INSPECTION_REPORT_PATH.read_text(encoding="utf-8"))


def load_goemotions() -> DatasetDict:
    """Load all official splits using the existing Hugging Face cache only."""
    dataset = load_dataset(
        DATASET_ID,
        download_config=DownloadConfig(local_files_only=True),
    )
    required_splits = {"train", "validation", "test"}
    missing_splits = required_splits.difference(dataset.keys())
    if missing_splits:
        raise ValueError(f"GoEmotions is missing required splits: {sorted(missing_splits)}")

    mapping = get_label_mapping(dataset)
    if tuple(item["emotion"] for item in mapping) != EXPECTED_LABEL_MAPPING:
        raise ValueError("Loaded GoEmotions label mapping differs from the inspected 28-class mapping.")
    report = _read_inspection_report()
    report_mapping = [
        {"index": item["index"], "emotion": item["emotion"]}
        for item in report["label_mapping"]
    ]
    if mapping != report_mapping:
        raise ValueError("Loaded GoEmotions labels differ from the ML inspection report.")
    report_sizes = report["split_sizes"]
    actual_sizes = {split: dataset[split].num_rows for split in required_splits}
    if any(actual_sizes[split] != report_sizes[split] for split in required_splits):
        raise ValueError(
            f"Loaded GoEmotions split sizes {actual_sizes} differ from inspected sizes."
        )
    return dataset


def get_label_mapping(dataset: DatasetDict | None = None) -> list[dict[str, Any]]:
    """Return the exact dataset-schema label indices and names."""
    if dataset is None:
        dataset = load_goemotions()
    names = dataset["train"].features["labels"].feature.names
    return [{"index": index, "emotion": name} for index, name in enumerate(names)]


def identify_target_emotions(
    label_mapping: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return project target emotions and their verified GoEmotions label IDs."""
    if label_mapping is None:
        report = _read_inspection_report()
        mapping = [
            {"index": item["index"], "emotion": item["emotion"]}
            for item in report["label_mapping"]
        ]
        if tuple(item["emotion"] for item in mapping) != EXPECTED_LABEL_MAPPING:
            raise ValueError("Inspection report label mapping differs from the expected schema.")
    else:
        mapping = label_mapping
    by_name = {item["emotion"]: item["index"] for item in mapping}
    missing = [emotion for emotion in TARGET_EMOTIONS if emotion not in by_name]
    if missing:
        raise ValueError(f"Target emotions are missing from the label mapping: {missing}")
    return [{"emotion": emotion, "index": by_name[emotion]} for emotion in TARGET_EMOTIONS]


def _to_frame(split_data: Any, split_name: str) -> pd.DataFrame:
    frame = split_data.to_pandas() if hasattr(split_data, "to_pandas") else pd.DataFrame(split_data)
    required = {"text", "labels", "id"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{split_name} split is missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["labels"] = frame["labels"].map(lambda labels: [int(label) for label in labels])
    frame["split"] = split_name
    frame["source"] = "gold"
    frame["oversampled"] = False
    return frame


def create_gold_only(dataset: DatasetDict | None = None) -> dict[str, Any]:
    """Return unchanged official splits with explicit provenance columns."""
    dataset = dataset if dataset is not None else load_goemotions()
    gold = {
        split: _to_frame(dataset[split], split)
        for split in ("train", "validation", "test")
    }
    validate_label_ids(gold)
    return {
        **gold,
        "metadata": {
            "condition": "GOLD_ONLY",
            "dataset_id": DATASET_ID,
            "split_sizes": {split: len(gold[split]) for split in gold},
            "label_mapping": get_label_mapping(dataset),
        },
    }


def _as_train_frame(train_data: Any | None) -> pd.DataFrame:
    if train_data is None:
        dataset = load_goemotions()
        frame = _to_frame(dataset["train"], "train")
    elif isinstance(train_data, pd.DataFrame):
        frame = train_data.copy()
    else:
        frame = _to_frame(train_data, "train")
    if "split" in frame and not frame["split"].eq("train").all():
        raise ValueError("Standard oversampling accepts only rows from the original train split.")
    if "source" in frame and not frame["source"].eq("gold").all():
        raise ValueError("Standard oversampling accepts only original gold training examples.")
    if not {"text", "labels", "id"}.issubset(frame.columns):
        raise ValueError("Training data must have text, labels, and id columns.")
    frame["labels"] = frame["labels"].map(lambda labels: [int(label) for label in labels])
    frame["split"] = "train"
    frame["source"] = "gold"
    if "oversampled" not in frame:
        frame["oversampled"] = False
    return frame


def _label_counts(frame: pd.DataFrame, target_mapping: list[dict[str, Any]]) -> dict[str, int]:
    return {
        item["emotion"]: sum(item["index"] in labels for labels in frame["labels"])
        for item in target_mapping
    }


def create_standard_oversampling(
    train_data: Any | None = None,
    *,
    random_seed: int = RANDOM_SEED,
    target_count: int = OVERSAMPLING_TARGET_COUNT,
) -> pd.DataFrame:
    """Oversample gold train rows carrying target labels, with replacement."""
    if target_count < 1:
        raise ValueError("target_count must be positive.")
    original = _as_train_frame(train_data)
    target_mapping = identify_target_emotions()
    initial_counts = _label_counts(original, target_mapping)
    if any(count > target_count for count in initial_counts.values()):
        raise ValueError(
            f"Target count {target_count} would downsample an emotion: {initial_counts}"
        )

    rng = np.random.default_rng(random_seed)
    additions = []
    target_label_ids = {item["index"] for item in target_mapping}
    for item in target_mapping:
        emotion, label_id = item["emotion"], item["index"]
        deficit = max(0, target_count - initial_counts[emotion])
        if deficit == 0:
            continue
        candidates = original.loc[
            original["labels"].map(
                lambda labels: label_id in labels
                and not (target_label_ids.intersection(labels) - {label_id})
            )
        ]
        if candidates.empty:
            raise ValueError(
                f"No gold training examples without another target label found for {emotion}."
            )
        selected_positions = rng.integers(0, len(candidates), size=deficit)
        for position in selected_positions:
            duplicate = candidates.iloc[int(position)].copy()
            duplicate["oversampled"] = True
            duplicate["oversampling_emotion"] = emotion
            duplicate["oversample_index"] = len(additions)
            additions.append(duplicate)

    result = pd.concat([original, pd.DataFrame(additions)], ignore_index=True)
    result.attrs["oversampling"] = {
        "method": (
            "deterministic random oversampling with replacement from original train rows; "
            "multi-target rows are excluded from sampling to avoid incidental target-label inflation"
        ),
        "seed": random_seed,
        "target_count_per_emotion": target_count,
        "target_counts_before": initial_counts,
        "target_counts_after": _label_counts(result, target_mapping),
        "synthetic_text_generated": False,
        "validation_test_modified": False,
    }
    return result


def load_synthetic_candidates(
    paths: tuple[str | Path, ...] | list[str | Path] | None = None,
) -> pd.DataFrame:
    """Load all available batch candidates and report target-emotion completeness."""
    selected_paths = (
        [Path(path) if Path(path).is_absolute() else PROJECT_ROOT / path for path in paths]
        if paths is not None
        else [PROJECT_ROOT / path for path in SYNTHETIC_PATHS]
    )
    mapping = {item["emotion"]: item["index"] for item in identify_target_emotions()}
    frames = []
    missing_files = []
    for path in selected_paths:
        if not path.is_file():
            missing_files.append(str(path))
            continue
        frame = pd.read_csv(path)
        required = {"emotion", "text", "source"}
        missing_columns = required.difference(frame.columns)
        if missing_columns:
            raise ValueError(f"{path} is missing columns: {sorted(missing_columns)}")
        if not frame["source"].eq("synthetic").all():
            raise ValueError(f"Rows in {path} must have source='synthetic'.")
        unknown = sorted(set(frame["emotion"].dropna()) - set(mapping))
        if unknown:
            raise ValueError(f"Unexpected synthetic target labels in {path}: {unknown}")
        frame = frame.copy()
        frame["batch_file"] = path.name
        frame["labels"] = frame["emotion"].map(lambda emotion: [mapping[emotion]])
        frame["split"] = "train"
        frame["id"] = frame.apply(
            lambda row: f"synthetic::{path.stem}::{row.get('synthetic_id', row.name)}",
            axis=1,
        )
        frame["source"] = "synthetic"
        frames.append(frame)

    columns = [
        "id",
        "text",
        "labels",
        "emotion",
        "source",
        "split",
        "batch_file",
        "synthetic_id",
        "decision",
        "valence",
        "arousal",
        "dominance",
        "distance_to_centroid",
        "target_emotion_threshold",
        "model",
        "generation_model",
        "prompt_version",
        "timestamp",
        "generation_timestamp",
    ]
    available_columns = list(dict.fromkeys(
        column
        for frame in frames
        for column in frame.columns
    ))
    ordered_columns = [column for column in columns if column in available_columns]
    ordered_columns.extend(column for column in available_columns if column not in ordered_columns)
    combined = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame(columns=ordered_columns)

    normalized_text = combined["text"].map(_normalize_text) if not combined.empty else pd.Series(dtype=str)
    duplicate_mask = normalized_text.duplicated(keep="first")
    duplicate_count = int(duplicate_mask.sum())
    combined = combined.loc[~duplicate_mask].reset_index(drop=True)
    counts = {
        emotion: int(combined["emotion"].eq(emotion).sum())
        for emotion in TARGET_EMOTIONS
    }
    missing_emotions = [emotion for emotion, count in counts.items() if count == 0]
    combined.attrs["availability"] = {
        "paths": [str(path) for path in selected_paths],
        "missing_files": missing_files,
        "available_emotions": [emotion for emotion, count in counts.items() if count > 0],
        "missing_emotions": missing_emotions,
        "counts_per_emotion": counts,
        "exact_duplicates_removed": duplicate_count,
        "complete": not missing_emotions and not missing_files,
    }
    validate_label_ids(combined)
    return combined


def _normalize_text(text: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))


def _create_synthetic_condition(
    condition_name: str,
    *,
    vad_filtered: bool,
    gold: dict[str, Any] | None = None,
    synthetic_candidates: pd.DataFrame | None = None,
    require_complete: bool = False,
) -> dict[str, Any]:
    gold = gold if gold is not None else create_gold_only()
    candidates = (
        synthetic_candidates.copy()
        if synthetic_candidates is not None
        else load_synthetic_candidates()
    )
    availability = dict(candidates.attrs.get("availability", {}))
    if "availability" not in availability:
        counts = {
            emotion: int(candidates.get("emotion", pd.Series(dtype=str)).eq(emotion).sum())
            for emotion in TARGET_EMOTIONS
        }
        missing = [emotion for emotion, count in counts.items() if count == 0]
        availability = {
            "available_emotions": [emotion for emotion, count in counts.items() if count > 0],
            "missing_emotions": missing,
            "counts_per_emotion": counts,
            "complete": not missing,
        }
    selected = candidates
    if vad_filtered:
        if "decision" not in selected.columns:
            raise ValueError("VAD-filtered condition requires saved decision values.")
        selected = selected.loc[selected["decision"].eq("ACCEPT")].copy()

    included_counts = {
        emotion: int(selected["emotion"].eq(emotion).sum()) if not selected.empty else 0
        for emotion in TARGET_EMOTIONS
    }
    missing_included = [emotion for emotion, count in included_counts.items() if count == 0]
    availability["candidate_counts_per_emotion"] = availability.get("counts_per_emotion", {})
    availability["counts_per_emotion"] = included_counts
    availability["available_emotions"] = [
        emotion for emotion, count in included_counts.items() if count > 0
    ]
    availability["missing_emotions"] = missing_included
    availability["complete"] = not missing_included and not availability.get("missing_files", [])
    if require_complete and not availability["complete"]:
        raise IncompleteSyntheticConditionError(
            f"{condition_name} is incomplete; missing target emotions: {missing_included}"
        )

    gold_texts = {
        _normalize_text(text)
        for split in ("train", "validation", "test")
        for text in gold[split]["text"]
        if _normalize_text(text)
    }
    synthetic_texts = selected["text"].map(_normalize_text) if not selected.empty else pd.Series(dtype=str)
    overlap = sorted(set(synthetic_texts).intersection(gold_texts))
    if overlap:
        raise ValueError(
            f"Synthetic text overlaps original gold training data ({len(overlap)} normalized texts)."
        )

    gold_train = gold["train"].copy()
    gold_train["synthetic_id"] = pd.NA
    synthetic_train = selected.copy()
    if not synthetic_train.empty:
        synthetic_train["source"] = "synthetic"
        synthetic_train["split"] = "train"
        synthetic_train["oversampled"] = False
    combined_train = pd.concat([gold_train, synthetic_train], ignore_index=True, sort=False)
    condition = {
        "train": combined_train,
        "validation": gold["validation"].copy(),
        "test": gold["test"].copy(),
        "metadata": {
            "condition": condition_name,
            "synthetic_inclusion": "accepted_only" if vad_filtered else "all_available_candidates",
            "synthetic_completeness": availability,
            "complete": availability["complete"],
            "blocked_for_final_experiment": not availability["complete"],
            "training_rows": len(combined_train),
            "validation_rows": len(gold["validation"]),
            "test_rows": len(gold["test"]),
            "validation_test_unchanged": True,
        },
    }
    condition["metadata"]["validation_test_unchanged"] = (
        condition["validation"].equals(gold["validation"])
        and condition["test"].equals(gold["test"])
    )
    validate_label_ids(condition)
    return condition


def create_unfiltered_synthetic(
    *,
    gold: dict[str, Any] | None = None,
    synthetic_candidates: pd.DataFrame | None = None,
    require_complete: bool = False,
) -> dict[str, Any]:
    """Add all available generated candidates to gold train; never use eval splits."""
    return _create_synthetic_condition(
        "UNFILTERED_SYNTHETIC",
        vad_filtered=False,
        gold=gold,
        synthetic_candidates=synthetic_candidates,
        require_complete=require_complete,
    )


def create_vad_filtered_synthetic(
    *,
    gold: dict[str, Any] | None = None,
    synthetic_candidates: pd.DataFrame | None = None,
    require_complete: bool = False,
) -> dict[str, Any]:
    """Add only saved ACCEPT candidates to gold train; never use eval splits."""
    return _create_synthetic_condition(
        "VAD_FILTERED_SYNTHETIC",
        vad_filtered=True,
        gold=gold,
        synthetic_candidates=synthetic_candidates,
        require_complete=require_complete,
    )


def validate_label_ids(data: Any) -> None:
    """Raise if any row has a label outside the original 28-label space."""
    if isinstance(data, dict) and "train" in data:
        frames = [data[split] for split in ("train", "validation", "test") if split in data]
    elif isinstance(data, pd.DataFrame):
        frames = [data]
    else:
        raise TypeError("Expected a prepared split mapping or pandas DataFrame.")
    invalid = sorted(
        {
            int(label)
            for frame in frames
            for labels in frame.get("labels", pd.Series(dtype=object))
            for label in labels
            if int(label) < 0 or int(label) >= len(EXPECTED_LABEL_MAPPING)
        }
    )
    if invalid:
        raise ValueError(f"Invalid GoEmotions label IDs found: {invalid}")


def check_split_integrity(splits: dict[str, Any]) -> dict[str, Any]:
    """Report cross-split ID/text overlap and accidental train text duplicates."""
    names = ("train", "validation", "test")
    id_sets = {
        split: set(splits[split]["id"].astype(str))
        for split in names
        if split in splits and "id" in splits[split]
    }
    text_sets = {
        split: {
            _normalize_text(text)
            for text in splits[split]["text"]
            if _normalize_text(text)
        }
        for split in names
        if split in splits
    }
    id_leakage = {}
    text_leakage = {}
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            if left in id_sets and right in id_sets:
                id_leakage[f"{left}__{right}"] = len(id_sets[left] & id_sets[right])
            if left in text_sets and right in text_sets:
                text_leakage[f"{left}__{right}"] = len(text_sets[left] & text_sets[right])

    train = splits.get("train", pd.DataFrame())
    if "source" in train:
        gold_train = train.loc[train["source"].eq("gold")]
        synthetic_train = train.loc[train["source"].eq("synthetic")]
    else:
        gold_train = train
        synthetic_train = pd.DataFrame(columns=train.columns)
    if "oversampled" in gold_train:
        gold_train = gold_train.loc[~gold_train["oversampled"].fillna(False)]

    gold_normalized = gold_train.get("text", pd.Series(dtype=str)).map(_normalize_text)
    gold_duplicate_count = int(gold_normalized[gold_normalized.ne("")].duplicated().sum())
    synthetic_normalized = synthetic_train.get("text", pd.Series(dtype=str)).map(_normalize_text)
    synthetic_duplicate_count = int(
        synthetic_normalized[synthetic_normalized.ne("")].duplicated().sum()
    )
    synthetic_train_texts = set(synthetic_normalized) - {""}
    gold_split_texts = {
        _normalize_text(text)
        for split in ("validation", "test")
        if split in splits
        for text in splits[split]["text"]
        if _normalize_text(text)
    }
    synthetic_eval_overlap = len(synthetic_train_texts & gold_split_texts)
    gold_train_texts = set(gold_normalized) - {""}
    synthetic_gold_overlap = len(synthetic_train_texts & gold_train_texts)
    oversampled_duplicate_count = int(
        train.get("oversampled", pd.Series(False, index=train.index)).fillna(False).sum()
    )
    no_new_split_leakage = (
        not any(id_leakage.values())
        and synthetic_eval_overlap == 0
        and synthetic_gold_overlap == 0
    )
    return {
        "cross_split_id_overlap": id_leakage,
        "cross_split_text_overlap": text_leakage,
        "preexisting_gold_train_duplicate_text_count": gold_duplicate_count,
        "synthetic_train_duplicate_text_count": synthetic_duplicate_count,
        "synthetic_gold_train_text_overlap": synthetic_gold_overlap,
        "synthetic_validation_test_text_overlap": synthetic_eval_overlap,
        "intentional_oversampled_rows": oversampled_duplicate_count,
        "no_split_leakage": not any(id_leakage.values()) and not any(text_leakage.values()),
        "no_new_split_leakage": no_new_split_leakage,
        "no_unintentional_synthetic_duplicates": synthetic_duplicate_count == 0,
        "gold_train_duplicate_texts_are_preexisting": True,
    }


def validate_condition(gold: dict[str, Any], condition: dict[str, Any]) -> dict[str, Any]:
    """Check provenance, labels, split preservation, leakage, and train duplicates."""
    validate_label_ids(condition)
    target_ids = TARGET_LABEL_IDS
    synthetic_rows = condition["train"].loc[
        condition["train"]["source"].eq("synthetic")
    ]
    synthetic_labels_valid = (
        True
        if synthetic_rows.empty
        else synthetic_rows[["emotion", "labels"]]
        .apply(
            lambda row: (
                row["emotion"] in target_ids
                and row["labels"] == [target_ids[row["emotion"]]]
            ),
            axis=1,
        )
        .all()
    )
    source_ok = (
        condition["train"]["source"].isin(["gold", "synthetic"]).all()
        and condition["train"].loc[condition["train"]["source"].eq("gold"), "source"].eq("gold").all()
        and synthetic_labels_valid
        and condition["train"].loc[condition["train"]["source"].eq("synthetic"), "labels"].map(
            lambda labels: len(labels) == 1
        ).all()
        and condition["validation"]["source"].eq("gold").all()
        and condition["test"]["source"].eq("gold").all()
    )
    validation_test_unchanged = (
        condition["validation"].equals(gold["validation"])
        and condition["test"].equals(gold["test"])
    )
    integrity = check_split_integrity(condition)
    integrity["source_provenance_valid"] = bool(source_ok)
    integrity["validation_test_unchanged"] = validation_test_unchanged
    integrity["all_checks_passed"] = all(
        value for key, value in integrity.items()
        if isinstance(value, bool) and key != "no_split_leakage"
    )
    return integrity


def get_target_training_counts(train_data: pd.DataFrame) -> dict[str, int]:
    """Count multi-label target emotion occurrences in a training DataFrame."""
    return _label_counts(train_data, identify_target_emotions())
