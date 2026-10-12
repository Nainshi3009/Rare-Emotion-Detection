from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preparation import (  # noqa: E402
    TARGET_EMOTIONS,
    get_label_mapping,
    identify_target_emotions,
    load_goemotions,
)
from src.vad_targets import add_vad_targets, load_vad_lexicon  # noqa: E402


OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "processed" / "goemotions_vad_v1"
METADATA_PATH = OUTPUT_DIRECTORY / "metadata.json"
SYNTHETIC_FILES = (
    PROJECT_ROOT / "data" / "synthetic" / "vad_filtered_candidates_40.csv",
    PROJECT_ROOT / "data" / "synthetic" / "vad_filtered_candidates_40.metadata.json",
    PROJECT_ROOT / "data" / "synthetic" / "vad_filtered_candidates_batch2.csv",
    PROJECT_ROOT / "data" / "synthetic" / "vad_filtered_candidates_batch2.metadata.json",
)
ORIGINAL_COLUMNS = ("id", "text", "labels")
VAD_COLUMNS = (
    "vad_valence",
    "vad_arousal",
    "vad_dominance",
    "vad_coverage",
    "vad_matched_terms",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _label_rows(frame: pd.DataFrame) -> list[list[int]]:
    return [[int(label) for label in labels] for labels in frame["labels"]]


def _valid_vad(frame: pd.DataFrame) -> pd.Series:
    return frame[["vad_valence", "vad_arousal", "vad_dominance"]].notna().all(axis=1)


def _summarize(frame: pd.DataFrame) -> dict:
    valid = _valid_vad(frame)
    means = frame.loc[valid, ["vad_valence", "vad_arousal", "vad_dominance"]].mean()
    return {
        "rows": int(len(frame)),
        "valid_vad_rows": int(valid.sum()),
        "valid_vad_percentage": float(valid.mean() * 100) if len(frame) else 0.0,
        "zero_lexicon_coverage_rows": int(frame["vad_matched_terms"].eq(0).sum()),
        "mean_vad": {
            "valence": float(means["vad_valence"]) if valid.any() else None,
            "arousal": float(means["vad_arousal"]) if valid.any() else None,
            "dominance": float(means["vad_dominance"]) if valid.any() else None,
        },
    }


def main() -> int:
    synthetic_hashes_before = {
        path.name: _sha256(path)
        for path in SYNTHETIC_FILES
        if path.is_file()
    }
    dataset = load_goemotions()
    label_mapping = get_label_mapping(dataset)
    targets = identify_target_emotions(label_mapping)
    lexicon = load_vad_lexicon()

    original_frames = {}
    prepared_frames = {}
    for split in ("train", "validation", "test"):
        original = dataset[split].to_pandas()
        if tuple(column for column in ORIGINAL_COLUMNS if column in original.columns) != ORIGINAL_COLUMNS:
            raise ValueError(f"Unexpected original column schema in {split}.")
        prepared = add_vad_targets(original, lexicon=lexicon)
        prepared["split"] = split

        if len(prepared) != len(original):
            raise AssertionError(f"{split} row count changed during VAD preparation.")
        if prepared["id"].tolist() != original["id"].tolist():
            raise AssertionError(f"{split} IDs or row order changed during VAD preparation.")
        if prepared["text"].tolist() != original["text"].tolist():
            raise AssertionError(f"{split} text changed during VAD preparation.")
        if _label_rows(prepared) != _label_rows(original):
            raise AssertionError(f"{split} label IDs changed during VAD preparation.")
        labels = {label for row in _label_rows(prepared) for label in row}
        if not labels.issubset(set(range(28))):
            raise AssertionError(f"Invalid label IDs found in {split}: {sorted(labels - set(range(28)))}")

        original_frames[split] = original
        prepared_frames[split] = prepared

    split_id_sets = {
        split: set(frame["id"].astype(str))
        for split, frame in original_frames.items()
    }
    id_overlap = {}
    split_names = ("train", "validation", "test")
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            id_overlap[f"{left}__{right}"] = len(
                split_id_sets[left].intersection(split_id_sets[right])
            )
    if any(id_overlap.values()):
        raise AssertionError(f"GoEmotions IDs overlap across splits: {id_overlap}")

    target_statistics = {}
    for target in targets:
        emotion, label_id = target["emotion"], target["index"]
        split_statistics = {}
        target_frames = []
        for split, frame in prepared_frames.items():
            mask = frame["labels"].map(lambda labels: label_id in labels)
            target_split = frame.loc[mask]
            split_statistics[split] = _summarize(target_split)
            target_frames.append(target_split)
        target_frame = pd.concat(target_frames, ignore_index=True)
        target_statistics[emotion] = {
            "label_id": label_id,
            "all_splits": _summarize(target_frame),
            "by_split": split_statistics,
        }

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_files = {}
    for split, frame in prepared_frames.items():
        output_path = OUTPUT_DIRECTORY / f"{split}.parquet"
        frame.to_parquet(output_path, index=False, compression="snappy")
        output_files[split] = str(output_path.relative_to(PROJECT_ROOT))

        saved = pd.read_parquet(output_path)
        if len(saved) != len(original_frames[split]):
            raise AssertionError(f"Saved {split} row count is incorrect.")
        if saved["id"].tolist() != original_frames[split]["id"].tolist():
            raise AssertionError(f"Saved {split} IDs or row order changed.")
        if saved["text"].tolist() != original_frames[split]["text"].tolist():
            raise AssertionError(f"Saved {split} text differs from the original.")
        if _label_rows(saved) != _label_rows(original_frames[split]):
            raise AssertionError(f"Saved {split} labels differ from the original.")
        saved_label_ids = {label for row in _label_rows(saved) for label in row}
        if saved_label_ids != set(range(28)):
            raise AssertionError(f"Saved {split} label space is not the original 28 classes.")

    metadata = {
        "dataset_id": "google-research-datasets/go_emotions",
        "dataset_source": "Existing Hugging Face cache; loaded with local_files_only=True.",
        "label_mapping": label_mapping,
        "target_emotions": targets,
        "vad": {
            "version": "NRC-VAD V1 (2018)",
            "lexicon_path": "data/NRC-VAD-v1-2018/NRC-VAD-Lexicon.txt",
            "method": "Existing notebook tokenization; arithmetic mean of matched single-token V/A/D scores; unmatched tokens ignored.",
            "coverage_definition": "Matched V1 terms divided by tokenizer output token count; zero for nonempty token sequences without lexicon matches.",
            "columns": list(VAD_COLUMNS),
            "interpretation": "Lexicon-derived pseudo-targets; NOT human-annotated sentence-level VAD ground truth.",
        },
        "deterministic": True,
        "random_seed": 20261008,
        "outputs": output_files,
        "split_statistics": {
            split: _summarize(frame)
            for split, frame in prepared_frames.items()
        },
        "target_emotion_statistics_all_splits": target_statistics,
        "integrity": {
            "split_sizes": {split: len(frame) for split, frame in prepared_frames.items()},
            "cross_split_id_overlap": id_overlap,
            "all_original_ids_text_and_labels_preserved_in_order": True,
            "all_28_label_ids_preserved": True,
        },
        "synthetic_files_sha256_before": synthetic_hashes_before,
    }

    synthetic_hashes_after = {
        path.name: _sha256(path)
        for path in SYNTHETIC_FILES
        if path.is_file()
    }
    if synthetic_hashes_before != synthetic_hashes_after:
        raise AssertionError("One or more Batch 1/Batch 2 synthetic files changed.")
    metadata["synthetic_files_sha256_after"] = synthetic_hashes_after
    metadata["integrity"]["synthetic_files_unchanged"] = (
        synthetic_hashes_before == synthetic_hashes_after
    )
    METADATA_PATH.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Prepared V1 pseudo-target splits:")
    for split, stats in metadata["split_statistics"].items():
        print(
            f"  {split}: rows={stats['rows']}, valid_vad={stats['valid_vad_rows']} "
            f"({stats['valid_vad_percentage']:.2f}%), "
            f"zero_coverage={stats['zero_lexicon_coverage_rows']}, "
            f"mean_vad={stats['mean_vad']}"
        )
    print("Target-emotion statistics:")
    for emotion, stats in target_statistics.items():
        train_stats = stats["by_split"]["train"]
        print(
            f"  {emotion} (label {stats['label_id']}) train: rows={train_stats['rows']}, "
            f"valid_vad={train_stats['valid_vad_rows']} "
            f"({train_stats['valid_vad_percentage']:.2f}%), "
            f"zero_coverage={train_stats['zero_lexicon_coverage_rows']}, "
            f"mean_vad={train_stats['mean_vad']}"
        )
    print(f"Cross-split ID overlaps: {id_overlap}")
    print("Original text/label/ID order preserved in every split: yes")
    print("All 28 label IDs preserved: yes")
    print("Synthetic files unchanged: yes")
    for path in (*[OUTPUT_DIRECTORY / f"{split}.parquet" for split in prepared_frames], METADATA_PATH):
        print(f"Output: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
