"""Audit data and shared-model readiness for GoEmotions conditions A-D.

This checker is read-only with respect to source, synthetic, and configuration
inputs. It may write only reports/experiment_readiness.json.
"""

from __future__ import annotations

import html
import json
import math
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preparation import (  # noqa: E402
    EXPECTED_LABEL_MAPPING,
    OVERSAMPLING_TARGET_COUNT,
    RANDOM_SEED,
    SYNTHETIC_PATHS,
    TARGET_EMOTIONS,
    TARGET_LABEL_IDS,
    create_standard_oversampling,
)


DATA_DIRECTORY = PROJECT_ROOT / "data" / "processed" / "goemotions_vad_v1"
SYNTHETIC_DIRECTORY = PROJECT_ROOT / "data" / "synthetic"
NOTEBOOK_PATH = PROJECT_ROOT / "Rare_Emotion_Detection_EDA.ipynb"
RESOURCE_CONFIG_PATH = PROJECT_ROOT / "configs" / "model_resource_efficient.json"
BASELINE_CONFIG_PATH = PROJECT_ROOT / "configs" / "gold_only_baseline.json"
FORMULATION_CONFIG_PATH = PROJECT_ROOT / "configs" / "model_training_formulation.json"
BASELINE_RUNNER_PATH = PROJECT_ROOT / "scripts" / "run_gold_only.py"
REPORT_PATH = PROJECT_ROOT / "reports" / "experiment_readiness.json"
SPLIT_SIZES = {"train": 43_410, "validation": 5_426, "test": 5_427}
VAD_COLUMNS = ("valence", "arousal", "dominance")
SPLIT_VAD_COLUMNS = ("vad_valence", "vad_arousal", "vad_dominance")


class _SavedTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self._table: list[list[str]] | None = None
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            self._table = []
            self.tables.append(self._table)
        elif tag == "tr" and self._table is not None:
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell = []

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._cell is not None and self._row is not None:
            self._row.append(html.unescape("".join(self._cell)).strip())
            self._cell = None
        elif tag == "tr" and self._row is not None and self._table is not None:
            if self._row:
                self._table.append(self._row)
            self._row = None
        elif tag == "table":
            self._table = None


def _normalize_text(value: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value).lower()))


def _notebook_table(cell_index: int, display_index: int) -> list[dict[str, str]]:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cell = notebook["cells"][cell_index]
    if cell.get("cell_type") != "code" or cell.get("execution_count") is None:
        raise ValueError(f"Notebook cell {cell_index} is not a saved executed cell.")
    displays = [
        output.get("data", {}).get("text/html")
        for output in cell.get("outputs", [])
        if output.get("output_type") == "display_data"
        and output.get("data", {}).get("text/html")
    ]
    if len(displays) <= display_index:
        raise ValueError(f"Notebook table {cell_index}:{display_index} is missing.")
    source = displays[display_index]
    if isinstance(source, list):
        source = "".join(source)
    parser = _SavedTableParser()
    parser.feed(source)
    if not parser.tables or not parser.tables[0]:
        raise ValueError(f"Could not parse saved notebook table {cell_index}:{display_index}.")
    rows = parser.tables[0]
    headers = rows[0]
    if headers and not headers[0]:
        headers = headers[1:]
    records = []
    for row in rows[1:]:
        if len(row) == len(headers) + 1 and row[0].isdigit():
            row = row[1:]
        if len(row) == len(headers):
            records.append(dict(zip(headers, row)))
    if not records:
        raise ValueError(f"Saved notebook table {cell_index}:{display_index} has no rows.")
    return records


def _load_vad_filter_parameters() -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    centroid_rows = _notebook_table(29, 0)
    threshold_rows = _notebook_table(36, 1)
    centroids = {
        row["emotion"]: {
            dimension: float(row[dimension]) for dimension in VAD_COLUMNS
        }
        for row in centroid_rows
    }
    thresholds = {}
    for row in threshold_rows:
        if int(row["calibration_percentile"]) != 95:
            raise ValueError("Saved VAD thresholds are not the existing P95 thresholds.")
        thresholds[row["emotion"]] = float(row["threshold"])
    if set(centroids) != set(TARGET_EMOTIONS) or set(thresholds) != set(TARGET_EMOTIONS):
        raise ValueError("Saved V1 centroids/thresholds do not cover all target emotions.")
    return centroids, thresholds


def _read_splits() -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    splits: dict[str, pd.DataFrame] = {}
    checks: dict[str, Any] = {}
    for split, expected_size in SPLIT_SIZES.items():
        path = DATA_DIRECTORY / f"{split}.parquet"
        if not path.is_file():
            checks[split] = {"exists": False, "valid": False, "error": "file missing"}
            continue
        frame = pd.read_parquet(path)
        problems = []
        if len(frame) != expected_size:
            problems.append(f"expected {expected_size} rows, found {len(frame)}")
        required = {"id", "text", "labels", "split", *SPLIT_VAD_COLUMNS}
        missing = sorted(required - set(frame.columns))
        if missing:
            problems.append(f"missing columns: {missing}")
        if "split" in frame and not frame["split"].eq(split).all():
            problems.append("split column does not match file")

        bad_labels = []
        if "labels" in frame:
            for row_number, labels in enumerate(frame["labels"]):
                try:
                    ids = [int(label) for label in labels]
                except (TypeError, ValueError):
                    bad_labels.append(row_number)
                    continue
                if not ids or len(ids) != len(set(ids)) or any(
                    label < 0 or label >= len(EXPECTED_LABEL_MAPPING) for label in ids
                ):
                    bad_labels.append(row_number)
        if bad_labels:
            problems.append(f"invalid label-ID rows: {len(bad_labels)}")

        vad_valid_rows = 0
        vad_columns_valid = False
        if not missing:
            vad_values = frame.loc[:, SPLIT_VAD_COLUMNS].apply(
                pd.to_numeric, errors="coerce"
            ).to_numpy(dtype=float)
            vad_mask = np.isfinite(vad_values).all(axis=1)
            vad_valid_rows = int(vad_mask.sum())
            vad_columns_valid = bool(
                ((vad_values[vad_mask] >= 0) & (vad_values[vad_mask] <= 1)).all()
            )
            if not vad_columns_valid:
                problems.append("valid VAD values fall outside [0,1]")
        splits[split] = frame
        checks[split] = {
            "exists": True,
            "rows": len(frame),
            "expected_rows": expected_size,
            "valid": not problems,
            "bad_label_rows": len(bad_labels),
            "valid_vad_rows": vad_valid_rows,
            "vad_values_in_range": vad_columns_valid,
            "problems": problems,
        }
    return splits, checks


def _check_multihot(splits: dict[str, pd.DataFrame]) -> dict[str, Any]:
    all_valid = True
    rows_checked = 0
    multi_label_rows = 0
    invalid_rows = 0
    rows_by_split = {}
    multilabel_by_split = {}
    for split_name, frame in splits.items():
        split_multilabel = 0
        split_invalid = 0
        if "labels" not in frame:
            all_valid = False
            continue
        rows_checked += len(frame)
        for labels in frame["labels"]:
            ids = [int(label) for label in labels]
            target = np.zeros(len(EXPECTED_LABEL_MAPPING), dtype=np.float32)
            if not ids or len(ids) != len(set(ids)) or any(i < 0 or i >= 28 for i in ids):
                invalid_rows += 1
                split_invalid += 1
                continue
            target[ids] = 1.0
            if not np.isin(target, [0.0, 1.0]).all() or int(target.sum()) != len(ids):
                invalid_rows += 1
                split_invalid += 1
            if len(ids) > 1:
                multi_label_rows += 1
                split_multilabel += 1
        rows_by_split[split_name] = len(frame)
        multilabel_by_split[split_name] = split_multilabel
        if split_invalid:
            all_valid = False
    all_valid = all_valid and invalid_rows == 0
    return {
        "valid": all_valid,
        "rows_checked": rows_checked,
        "invalid_rows": invalid_rows,
        "multi_label_rows": multi_label_rows,
        "rows_by_split": rows_by_split,
        "multi_label_rows_by_split": multilabel_by_split,
        "target_shape": [len(EXPECTED_LABEL_MAPPING)],
        "target_values": [0, 1],
    }


def _source_split_overlap(splits: dict[str, pd.DataFrame]) -> dict[str, int]:
    normalized = {
        split: {
            _normalize_text(text)
            for text in frame.get("text", pd.Series(dtype=str))
            if _normalize_text(text)
        }
        for split, frame in splits.items()
    }
    pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
    return {
        f"{left}__{right}": len(normalized.get(left, set()) & normalized.get(right, set()))
        for left, right in pairs
    }


def _inspect_synthetic(
    splits: dict[str, pd.DataFrame],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    centroids, thresholds = _load_vad_filter_parameters()
    frames: list[pd.DataFrame] = []
    file_diagnostics = []
    missing_candidate_files = []
    for relative_path in SYNTHETIC_PATHS:
        path = PROJECT_ROOT / relative_path
        if not path.is_file():
            missing_candidate_files.append(str(relative_path))
            continue
        frame = pd.read_csv(path)
        frame["__batch_file"] = path.name
        frames.append(frame)
        meta_path = path.with_suffix(".metadata.json")
        metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        file_diagnostics.append({
            "file": path.name,
            "rows": len(frame),
            "metadata_file": meta_path.name if meta_path.is_file() else None,
            "model": metadata.get("model", metadata.get("generation_model")),
            "prompt_version": metadata.get("prompt_version"),
        })
    for path in sorted(SYNTHETIC_DIRECTORY.glob("*.csv")):
        if path not in [PROJECT_ROOT / rel for rel in SYNTHETIC_PATHS]:
            frame = pd.read_csv(path)
            frame["__batch_file"] = path.name
            frames.append(frame)
            meta_path = path.with_suffix(".metadata.json")
            metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
            file_diagnostics.append({
                "file": path.name,
                "rows": len(frame),
                "metadata_file": meta_path.name if meta_path.is_file() else None,
                "model": metadata.get("model", metadata.get("generation_model")),
                "prompt_version": metadata.get("prompt_version"),
            })

    candidates = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    by_emotion = {emotion: 0 for emotion in TARGET_EMOTIONS}
    accepted_by_emotion = {emotion: 0 for emotion in TARGET_EMOTIONS}
    rejected_by_emotion = {emotion: 0 for emotion in TARGET_EMOTIONS}
    data_problems: list[str] = []
    decision_mismatches = 0
    distance_mismatches = 0
    valid_row_mask: list[bool] = []
    synthetic_texts: list[str] = []
    row_detail: list[dict[str, Any]] = []

    for index, row in candidates.iterrows():
        emotion = str(row.get("emotion", "")).strip().lower()
        text = row.get("text")
        text_ok = isinstance(text, str) and bool(text.strip())
        source_ok = str(row.get("source", "")).strip().lower() == "synthetic"
        emotion_ok = emotion in TARGET_EMOTIONS
        if emotion_ok:
            by_emotion[emotion] += 1
        else:
            data_problems.append(f"row {index}: unknown target emotion {emotion!r}")
        if not text_ok:
            data_problems.append(f"row {index}: missing or empty text")
        if not source_ok:
            data_problems.append(f"row {index}: source marker is not 'synthetic'")
        if text_ok:
            synthetic_texts.append(_normalize_text(text))

        vad_values = []
        vad_ok = True
        for column in VAD_COLUMNS:
            value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
            if pd.isna(value) or not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
                vad_ok = False
            vad_values.append(float(value) if not pd.isna(value) else float("nan"))

        distance = pd.to_numeric(
            pd.Series([row.get("distance_to_centroid")]), errors="coerce"
        ).iloc[0]
        threshold = pd.to_numeric(
            pd.Series([row.get("target_emotion_threshold")]), errors="coerce"
        ).iloc[0]
        distance_ok = not pd.isna(distance) and math.isfinite(float(distance)) and float(distance) >= 0
        threshold_ok = (
            not pd.isna(threshold)
            and math.isfinite(float(threshold))
            and float(threshold) > 0
        )
        if emotion_ok and vad_ok and distance_ok:
            expected_distance = math.sqrt(
                sum(
                    (vad_values[i] - centroids[emotion][dimension]) ** 2
                    for i, dimension in enumerate(VAD_COLUMNS)
                )
            )
            if not math.isclose(float(distance), expected_distance, rel_tol=1e-4, abs_tol=1e-5):
                distance_mismatches += 1
        if not vad_ok:
            data_problems.append(f"row {index}: missing, nonnumeric, or out-of-range VAD")
        if not distance_ok:
            data_problems.append(f"row {index}: distance is missing or invalid")
        if not threshold_ok:
            data_problems.append(f"row {index}: threshold is missing or invalid")

        decision = str(row.get("decision", "")).strip().upper()
        decision_ok = decision in {"ACCEPT", "REJECT"}
        if not decision_ok:
            data_problems.append(f"row {index}: invalid decision {decision!r}")
        rule_ok = emotion_ok and distance_ok and threshold_ok and decision_ok
        if rule_ok:
            expected_decision = "ACCEPT" if float(distance) <= float(threshold) else "REJECT"
            if decision != expected_decision:
                decision_mismatches += 1
                data_problems.append(f"row {index}: decision does not match saved distance/threshold")
        if emotion_ok and decision_ok:
            if decision == "ACCEPT":
                accepted_by_emotion[emotion] += 1
            else:
                rejected_by_emotion[emotion] += 1

        valid_row_mask.append(
            text_ok and source_ok and emotion_ok and vad_ok and distance_ok
            and threshold_ok and decision_ok and (rule_ok and decision == expected_decision)
        )
        row_detail.append({
            "emotion": emotion,
            "decision": decision,
            "model": row.get("generation_model", row.get("model")),
            "file": row.get("__batch_file"),
        })

    normalized_nonempty = [text for text in synthetic_texts if text]
    synthetic_duplicate_count = len(normalized_nonempty) - len(set(normalized_nonempty))
    split_texts = {
        split: {
            _normalize_text(text)
            for text in frame.get("text", pd.Series(dtype=str))
            if _normalize_text(text)
        }
        for split, frame in splits.items()
    }
    synthetic_text_set = set(normalized_nonempty)
    leakage_by_split = {
        split: len(synthetic_text_set & text_set)
        for split, text_set in split_texts.items()
    }

    total = len(candidates)
    accepted_total = sum(accepted_by_emotion.values())
    rejected_total = sum(rejected_by_emotion.values())
    acceptance_rates = {
        emotion: (
            accepted_by_emotion[emotion] / by_emotion[emotion]
            if by_emotion[emotion]
            else None
        )
        for emotion in TARGET_EMOTIONS
    }
    integrity = {
        "valid": bool(total == len(valid_row_mask) and all(valid_row_mask)),
        "problems": data_problems,
        "duplicate_normalized_text_count": synthetic_duplicate_count,
        "source_marker_valid": all(
            str(row.get("source", "")).strip().lower() == "synthetic"
            for _, row in candidates.iterrows()
        ) if total else True,
        "distance_recalculation_mismatches": distance_mismatches,
        "decision_rule_mismatches": decision_mismatches,
        "missing_candidate_files": missing_candidate_files,
    }
    summary = {
        "total": total,
        "by_emotion": by_emotion,
        "accepted_by_emotion": accepted_by_emotion,
        "rejected_by_emotion": rejected_by_emotion,
        "acceptance_rate_by_emotion": acceptance_rates,
        "accepted_total": accepted_total,
        "rejected_total": rejected_total,
        "overall_acceptance_rate": accepted_total / total if total else None,
        "models": sorted({str(item["model"]) for item in file_diagnostics if item["model"]}),
        "files": file_diagnostics,
    }
    leakage = {
        "synthetic_vs_synthetic_duplicate_count": synthetic_duplicate_count,
        "synthetic_vs_train": leakage_by_split.get("train", 0),
        "synthetic_vs_validation": leakage_by_split.get("validation", 0),
        "synthetic_vs_test": leakage_by_split.get("test", 0),
        "synthetic_leakage_zero": (
            synthetic_duplicate_count == 0 and not any(leakage_by_split.values())
        ),
    }
    return row_detail, summary | {"integrity": integrity}, leakage


def _check_oversampling(
    splits: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    if "train" not in splits or not {"id", "labels", "text"}.issubset(splits["train"].columns):
        return {"valid": False, "error": "train split unavailable or missing required fields"}
    train = splits["train"].copy()
    train["source"] = "gold"
    train["oversampled"] = False
    validation = splits.get("validation", pd.DataFrame())
    test = splits.get("test", pd.DataFrame())
    before = {
        emotion: int(train["labels"].map(lambda labels: TARGET_LABEL_IDS[emotion] in labels).sum())
        for emotion in TARGET_EMOTIONS
    }
    expected_additions = {emotion: max(0, OVERSAMPLING_TARGET_COUNT - before[emotion]) for emotion in TARGET_EMOTIONS}
    try:
        oversampled = create_standard_oversampling(
            train,
            random_seed=RANDOM_SEED,
            target_count=OVERSAMPLING_TARGET_COUNT,
        )
    except Exception as error:
        return {
            "valid": False,
            "error": f"{type(error).__name__}: {error}",
            "counts_before": before,
            "expected_additions": expected_additions,
        }

    additions = oversampled.loc[oversampled["oversampled"].fillna(False)]
    after = {
        emotion: int(
            oversampled["labels"].map(lambda labels: TARGET_LABEL_IDS[emotion] in labels).sum()
        )
        for emotion in TARGET_EMOTIONS
    }
    addition_counts = {
        emotion: int(
            additions["oversampling_emotion"].eq(emotion).sum()
        ) if "oversampling_emotion" in additions else 0
        for emotion in TARGET_EMOTIONS
    }
    eval_texts = {
        _normalize_text(text)
        for frame in (validation, test)
        for text in frame.get("text", pd.Series(dtype=str))
        if _normalize_text(text)
    }
    addition_eval_overlap = sum(
        _normalize_text(text) in eval_texts for text in additions.get("text", [])
    )
    valid_labels = True
    for labels in oversampled["labels"]:
        target = np.zeros(28, dtype=np.uint8)
        for label in labels:
            target[int(label)] = 1
        if not np.isin(target, [0, 1]).all():
            valid_labels = False
            break
    train_only = bool(
        oversampled["split"].eq("train").all()
        and oversampled["source"].eq("gold").all()
    )
    unchanged_eval = True
    properties = {
        "train_only": train_only,
        "validation_test_not_passed_to_constructor": unchanged_eval,
        "target_counts_before": before,
        "target_counts_after": after,
        "expected_additions": expected_additions,
        "actual_additions_by_emotion": addition_counts,
        "added_rows": len(additions),
        "expected_added_rows": sum(expected_additions.values()),
        "evaluation_text_overlap_from_additions": addition_eval_overlap,
        "no_synthetic_rows": bool(oversampled["source"].eq("gold").all()),
        "binary_multihot_labels": valid_labels,
    }
    properties["valid"] = bool(
        train_only
        and unchanged_eval
        and after == {emotion: OVERSAMPLING_TARGET_COUNT for emotion in TARGET_EMOTIONS}
        and addition_counts == expected_additions
        and len(additions) == sum(expected_additions.values())
        and addition_eval_overlap == 0
        and properties["no_synthetic_rows"]
        and valid_labels
    )
    return properties


def _configuration_consistency() -> dict[str, Any]:
    resource = json.loads(RESOURCE_CONFIG_PATH.read_text(encoding="utf-8"))
    baseline = json.loads(BASELINE_CONFIG_PATH.read_text(encoding="utf-8"))
    formulation = json.loads(FORMULATION_CONFIG_PATH.read_text(encoding="utf-8"))
    runner_source = (
        BASELINE_RUNNER_PATH.read_text(encoding="utf-8")
        if BASELINE_RUNNER_PATH.is_file()
        else ""
    )
    mismatches: dict[str, Any] = {}
    resource_fields = (
        "model_name",
        "max_length",
        "batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "weight_decay",
        "warmup_ratio",
        "seed",
        "lambda_vad",
        "classification_threshold",
        "freeze_strategy",
    )
    missing_wiring = [
        key for key in resource_fields
        if f'{key}=config_data["{key}"]' not in runner_source
    ]
    if "resource_config = json.loads(RESOURCE_CONFIG_PATH.read_text" not in runner_source:
        missing_wiring.append("resource_config_loading")
    if "**resource_config" not in runner_source:
        missing_wiring.append("resource_config_merge")
    if missing_wiring:
        mismatches["baseline_runner_shared_config_wiring"] = {
            "expected": "load shared resource config and explicitly pass its values",
            "missing": missing_wiring,
        }
    if resource.get("freeze_strategy") != "last_2_layers":
        mismatches["freeze_strategy"] = {
            "expected": "last_2_layers",
            "configured": resource.get("freeze_strategy"),
        }
    if "epochs" not in resource and (
        baseline.get("epochs") != 1
        or 'config_data["epochs"] = condition_config["epochs"]' not in runner_source
    ):
        mismatches["epochs"] = {
            "expected": "one epoch from Gold-only condition config",
            "configured": baseline.get("epochs"),
        }
    expected_formulation = {
        "classification_loss": "BCEWithLogitsLoss",
        "target_encoding": "multi_hot",
        "classification_threshold": resource.get("classification_threshold"),
        "vad_loss": "masked_MSE",
        "lambda_vad": resource.get("lambda_vad"),
        "class_weighting": False,
    }
    for key, expected in expected_formulation.items():
        if formulation.get(key) != expected:
            mismatches[key] = {
                "expected": expected,
                "configured": formulation.get(key),
            }
    return {
        "status": "PASS" if not mismatches else "BLOCKER",
        "shared_resource_configuration": resource,
        "selected_loss_formulation": formulation,
        "gold_only_epochs": baseline.get("epochs"),
        "gold_only_epoch_source": (
            "gold_only_baseline.json; model_resource_efficient.json does not define epochs"
        ),
        "conditions_intended_to_share_settings": ["A", "B", "C", "D"],
        "only_intended_difference": "training data",
        "mismatches": mismatches,
        "all_conditions_have_runners": {
            "A": BASELINE_RUNNER_PATH.is_file(),
            "B": False,
            "C": False,
            "D": False,
        },
    }


def _status(blockers: list[str], ready: str) -> dict[str, Any]:
    return {"status": "BLOCKED" if blockers else ready, "blockers": blockers}


def build_report() -> dict[str, Any]:
    splits, split_checks = _read_splits()
    multihot = _check_multihot(splits)
    target_mapping = {
        emotion: TARGET_LABEL_IDS.get(emotion) for emotion in TARGET_EMOTIONS
    }
    vad_files_exist = all((DATA_DIRECTORY / f"{split}.parquet").is_file() for split in SPLIT_SIZES)
    core_blockers = []
    if any(not check.get("valid", False) for check in split_checks.values()):
        core_blockers.append("one or more official GoEmotions VAD split files are missing or invalid")
    if not multihot["valid"]:
        core_blockers.append("the 28-label binary multi-hot representation is invalid")
    if not vad_files_exist:
        core_blockers.append("one or more VAD target split files are missing")
    if any(value is None for value in target_mapping.values()):
        core_blockers.append("one or more target emotions are missing from the project label mapping")

    oversampling = _check_oversampling(splits)
    b_blockers = []
    if not oversampling.get("valid", False):
        b_blockers.append(oversampling.get("error", "deterministic oversampling validation failed"))

    _, synthetic_summary, leakage = _inspect_synthetic(splits)
    summary_counts = synthetic_summary["by_emotion"]
    c_blockers = []
    for emotion in TARGET_EMOTIONS:
        if summary_counts[emotion] == 0:
            c_blockers.append(f"{emotion} synthetic examples missing")
    if synthetic_summary["integrity"]["problems"]:
        c_blockers.append("one or more synthetic candidates fail data integrity checks")
    if not leakage["synthetic_leakage_zero"]:
        c_blockers.append("synthetic duplicate or GoEmotions split leakage is nonzero")

    d_blockers = []
    for emotion in TARGET_EMOTIONS:
        if synthetic_summary["accepted_by_emotion"][emotion] == 0:
            d_blockers.append(f"{emotion} accepted synthetic examples missing")
    if synthetic_summary["integrity"]["problems"]:
        d_blockers.append("one or more synthetic candidate VAD/filter records are invalid")
    if not leakage["synthetic_leakage_zero"]:
        d_blockers.append("synthetic duplicate or GoEmotions split leakage is nonzero")

    all_accepted = (
        synthetic_summary["total"] > 0
        and synthetic_summary["accepted_total"] == synthetic_summary["total"]
    )
    c_d_difference = bool(synthetic_summary["rejected_total"] > 0)
    c_d_diagnostic = {
        "different": c_d_difference,
        "status": "DEMONSTRATED" if c_d_difference else "NOT YET DEMONSTRATED",
        "explanation": (
            "At least one generated candidate is rejected, so C includes candidates excluded from D."
            if c_d_difference
            else (
                "Current generated candidates do not contain rejected examples, so the unfiltered and "
                "VAD-filtered synthetic conditions would currently be identical for the available "
                "synthetic data."
                if all_accepted
                else "No rejected candidates are currently available to distinguish C from D."
            )
        ),
    }
    config = _configuration_consistency()

    condition_a = _status(core_blockers, "READY")
    condition_b_blockers = list(core_blockers) + b_blockers
    condition_b = _status(
        condition_b_blockers,
        "READY-CONSTRUCTIBLE",
    )
    condition_c_blockers = list(core_blockers) + c_blockers
    condition_d_blockers = list(core_blockers) + d_blockers
    condition_c = _status(condition_c_blockers, "READY")
    condition_d = _status(condition_d_blockers, "READY")

    source_overlap = _source_split_overlap(splits)
    overall_blockers = []
    for name, condition in (
        ("A", condition_a),
        ("B", condition_b),
        ("C", condition_c),
        ("D", condition_d),
    ):
        if condition["status"] == "BLOCKED":
            overall_blockers.append(f"Condition {name} is blocked")
    if config["status"] != "PASS":
        overall_blockers.append("configuration consistency has blockers")

    report = {
        "project": "Rare Emotion Detection",
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "core_dataset": {
            "split_files": split_checks,
            "multi_hot_representation": multihot,
            "target_emotion_mapping": target_mapping,
            "vad_target_files_exist": vad_files_exist,
            "source_split_text_overlap": source_overlap,
        },
        "condition_A": condition_a,
        "condition_B": condition_b | {
            "data_status": "READY-CONSTRUCTIBLE" if oversampling.get("valid") else "BLOCKED",
            "oversampling_validation": oversampling,
        },
        "condition_C": condition_c,
        "condition_D": condition_d,
        "synthetic_summary": synthetic_summary,
        "leakage": leakage | {"existing_source_split_text_overlap": source_overlap},
        "configuration_consistency": config,
        "experiment_execution_status": "BLOCKED" if config["status"] != "PASS" else "READY",
        "experiment_execution_blockers": (
            ["shared model configuration does not match the selected experiment methodology"]
            if config["status"] != "PASS"
            else []
        ),
        "c_vs_d_difference": c_d_difference,
        "c_vs_d_diagnostic": c_d_diagnostic,
        "overall_status": "BLOCKED" if overall_blockers else "READY",
        "overall_blockers": overall_blockers,
        "next_required_action": (
            "Obtain genuine synthetic examples for missing target emotions and naturally occurring "
            "VAD-rejected candidates before constructing C/D."
            if config["status"] == "PASS"
            else "Resolve shared training-config wiring before running any condition."
        ),
        "training_performed": False,
        "synthetic_generation_performed": False,
        "synthetic_files_modified": False,
    }
    return report


def _print_report(report: dict[str, Any]) -> None:
    def mark(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    print("=" * 60)
    print("RARE EMOTION DETECTION - EXPERIMENT READINESS")
    print("=" * 60)
    core = report["core_dataset"]
    print("\nCore dataset")
    print(f"[{mark(all(x['valid'] for x in core['split_files'].values()))}] GoEmotions splits")
    print(f"[{mark(core['multi_hot_representation']['valid'])}] 28-label multi-hot representation")
    print(f"[{mark(core['vad_target_files_exist'])}] VAD pseudo-targets")
    print(f"[{mark(all(core['target_emotion_mapping'].values()))}] Target emotion mapping")

    for key, title in (
        ("condition_A", "Condition A - GOLD ONLY"),
        ("condition_B", "Condition B - GOLD + STANDARD OVERSAMPLING"),
        ("condition_C", "Condition C - GOLD + UNFILTERED SYNTHETIC"),
        ("condition_D", "Condition D - GOLD + VAD-FILTERED SYNTHETIC"),
    ):
        item = report[key]
        print(f"\n{title}\n[{item['status']}]")
        for blocker in item["blockers"]:
            print(f"  - {blocker}")
    summary = report["synthetic_summary"]
    print("\nSynthetic diagnostics")
    print(f"Total candidates: {summary['total']}")
    print(f"Models: {', '.join(summary['models']) or 'unknown'}")
    for emotion in TARGET_EMOTIONS:
        print(
            f"{emotion}: {summary['by_emotion'][emotion]} "
            f"(accepted {summary['accepted_by_emotion'][emotion]}, "
            f"rejected {summary['rejected_by_emotion'][emotion]})"
        )
    print(
        f"Accepted total: {summary['accepted_total']}; "
        f"rejected total: {summary['rejected_total']}; "
        f"overall acceptance rate: {summary['overall_acceptance_rate']}"
    )
    print("\nC vs D")
    print(f"[{report['c_vs_d_diagnostic']['status']}]")
    print(report["c_vs_d_diagnostic"]["explanation"])
    print("\nLeakage")
    print(f"[{mark(report['leakage']['synthetic_leakage_zero'])}] Synthetic duplicates/split leakage")
    print("Existing source split text overlap:", report["leakage"]["existing_source_split_text_overlap"])
    print("\nModel configuration consistency")
    print(f"[{report['configuration_consistency']['status']}]")
    for key, mismatch in report["configuration_consistency"]["mismatches"].items():
        print(f"  - {key}: {mismatch}")
    print("\nOVERALL STATUS")
    print(report["overall_status"])
    print("Experiment execution status:", report["experiment_execution_status"])
    for blocker in report["overall_blockers"]:
        print(f"  - {blocker}")
    print("Next required action:", report["next_required_action"])
    print(f"Machine-readable report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print("=" * 60)


def main() -> int:
    report = build_report()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
