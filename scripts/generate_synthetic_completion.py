"""Generate genuine Gemini completion candidates; defaults to a no-call dry-run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import generate_synthetic_candidates as stage1  # noqa: E402
from src.data_preparation import TARGET_EMOTIONS  # noqa: E402


OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "synthetic"
CSV_PATH = OUTPUT_DIRECTORY / "vad_filtered_candidates_completion.csv"
METADATA_PATH = OUTPUT_DIRECTORY / "vad_filtered_candidates_completion.metadata.json"
DATA_DIRECTORY = PROJECT_ROOT / "data" / "processed" / "goemotions_vad_v1"
MODEL = "gemini-3.8-flash"
TARGETS = ("pride", "nervousness")
CANDIDATES_PER_EMOTION = 25
PROMPT_VERSION = "rare-emotion-completion-v1.0"
RANDOM_SEED = 20261008
MAX_ATTEMPTS = 3
BACKOFF_BASE_SECONDS = 2
GENERATION_SETTINGS = {
    "temperature": 0.7,
    "max_output_tokens": 4200,
    "response_mime_type": "application/json",
}
DISTANCE_METRIC = "3D Euclidean distance in NRC-VAD V1 valence/arousal/dominance space"
THRESHOLD_RULE = "Existing per-emotion NRC-VAD V1 95th percentile of GoEmotions training distances"
CSV_COLUMNS = (
    "synthetic_id",
    "emotion",
    "text",
    "source",
    "model",
    "prompt_version",
    "timestamp",
    "valence",
    "arousal",
    "dominance",
    "distance_to_centroid",
    "target_emotion_threshold",
    "decision",
)
PROMPT_TEMPLATE = """Generate exactly 25 distinct candidate sentences for a multi-label emotion-classification dataset.
The target label is: {emotion}

Write natural, realistic English sentences plausibly associated with this target while allowing everyday context and varied wording.
Do not use the target emotion word or its grammatical variants.
Do not explain the emotion or provide a textbook definition.
Do not produce VAD scores.
Do not copy, quote, imitate, or reproduce examples from GoEmotions.
Avoid duplicates and near-duplicates; each item must be a distinct sentence.
Return only JSON matching the requested schema, with exactly 25 objects in "candidates", each containing a "text" string.
"""
PROTECTED_FILES = (
    Path("Rare_Emotion_Detection_EDA.ipynb"),
    Path("data/processed/goemotions_vad_v1/train.parquet"),
    Path("data/processed/goemotions_vad_v1/validation.parquet"),
    Path("data/processed/goemotions_vad_v1/test.parquet"),
    Path("data/processed/goemotions_vad_v1/metadata.json"),
    Path("data/synthetic/vad_filtered_candidates_40.csv"),
    Path("data/synthetic/vad_filtered_candidates_40.metadata.json"),
    Path("data/synthetic/vad_filtered_candidates_batch2.csv"),
    Path("data/synthetic/vad_filtered_candidates_batch2.metadata.json"),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _protected_hashes() -> dict[str, str]:
    missing = [
        str(relative)
        for relative in PROTECTED_FILES
        if not (PROJECT_ROOT / relative).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Protected inputs are missing: {missing}")
    return {
        relative.as_posix(): _sha256(PROJECT_ROOT / relative)
        for relative in PROTECTED_FILES
    }


def _normalized(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _candidate_pairs(texts: list[str], *, similarity_threshold: float = 0.90) -> list[dict[str, Any]]:
    pairs = []
    normalized = [_normalized(text) for text in texts]
    for left in range(len(normalized)):
        if not normalized[left]:
            continue
        for right in range(left + 1, len(normalized)):
            if not normalized[right]:
                continue
            ratio = SequenceMatcher(None, normalized[left], normalized[right]).ratio()
            if ratio >= similarity_threshold:
                pairs.append({"index_1": left, "index_2": right, "similarity": ratio})
    return pairs


def _read_existing_texts() -> tuple[dict[str, list[str]], dict[str, int]]:
    existing: dict[str, list[str]] = {"synthetic": [], "goemotions": []}
    counts = {}
    for path in sorted((PROJECT_ROOT / "data" / "synthetic").glob("*.csv")):
        frame = pd.read_csv(path)
        if "text" not in frame:
            raise ValueError(f"Existing synthetic CSV lacks text column: {path.name}")
        batch_texts = frame["text"].dropna().astype(str).tolist()
        existing["synthetic"].extend(batch_texts)
        counts[path.name] = len(batch_texts)
    for split in ("train", "validation", "test"):
        path = DATA_DIRECTORY / f"{split}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"Prepared GoEmotions split is missing: {path}")
        frame = pd.read_parquet(path, columns=["text"])
        batch_texts = frame["text"].dropna().astype(str).tolist()
        existing["goemotions"].extend(batch_texts)
        counts[f"goemotions_{split}"] = len(batch_texts)
    return existing, counts


def _response_schema() -> dict[str, Any]:
    return {
        "type": "OBJECT",
        "properties": {
            "candidates": {
                "type": "ARRAY",
                "minItems": CANDIDATES_PER_EMOTION,
                "maxItems": CANDIDATES_PER_EMOTION,
                "items": {
                    "type": "OBJECT",
                    "properties": {"text": {"type": "STRING"}},
                    "required": ["text"],
                },
            }
        },
        "required": ["candidates"],
    }


def _parse_candidates(response_text: str | None) -> list[str]:
    if not response_text:
        raise ValueError("Gemini returned an empty response.")
    payload = json.loads(response_text)
    if not isinstance(payload, dict) or not isinstance(payload.get("candidates"), list):
        raise ValueError('Gemini response must contain a "candidates" array.')
    candidates = payload["candidates"]
    if len(candidates) != CANDIDATES_PER_EMOTION:
        raise ValueError(
            f"Expected {CANDIDATES_PER_EMOTION} candidate texts, got {len(candidates)}."
        )
    texts = []
    for index, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, dict) or not isinstance(candidate.get("text"), str):
            raise ValueError(f"Candidate {index} does not contain a text string.")
        text = candidate["text"].strip()
        if not text:
            raise ValueError(f"Candidate {index} is empty.")
        texts.append(text)
    return texts


def _redacted_error(error: Exception, api_key: str) -> str:
    message = str(error)
    return message.replace(api_key, "[REDACTED]") if api_key else message


def _is_quota_exhausted(error: Exception) -> bool:
    message = str(error).lower()
    return (
        "resource_exhausted" in message
        or ("quota exceeded" in message and "retry in" in message)
    )


def _generate_for_emotion(
    client: Any,
    emotion: str,
    *,
    request_seed: int,
    api_key: str,
    metadata: dict[str, Any],
) -> list[str] | None:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            metadata["api_request_count"] += 1
            response = client.models.generate_content(
                model=MODEL,
                contents=PROMPT_TEMPLATE.format(emotion=emotion),
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=_response_schema(),
                    temperature=GENERATION_SETTINGS["temperature"],
                    seed=request_seed,
                    max_output_tokens=GENERATION_SETTINGS["max_output_tokens"],
                ),
            )
            return _parse_candidates(response.text)
        except (APIError, httpx.HTTPError, json.JSONDecodeError, ValueError) as error:
            failure = {
                "emotion": emotion,
                "attempt": attempt,
                "error_type": type(error).__name__,
                "error": _redacted_error(error, api_key),
            }
            metadata["generation_failures"].append(failure)
            if isinstance(error, (APIError, httpx.HTTPError)):
                metadata["api_failures"].append(failure.copy())
            if _is_quota_exhausted(error):
                metadata["quota_exhausted"] = True
                return None
            if attempt < MAX_ATTEMPTS:
                delay = BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
                time.sleep(delay)
    return None


def _score_text(
    text: str,
    emotion: str,
    *,
    vad: dict[str, Any],
    centroids: dict[str, dict[str, float]],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    vad_result = vad["calculate_sentence_vad_v1"](text)
    distance = vad["calculate_vad_distance"](vad_result, centroids[emotion])
    threshold = thresholds[emotion]
    score = {
        "valence": vad_result.get("valence"),
        "arousal": vad_result.get("arousal"),
        "dominance": vad_result.get("dominance"),
        "distance_to_centroid": distance,
        "target_emotion_threshold": threshold,
    }
    if pd.isna(distance) or not math.isfinite(float(distance)):
        score["decision"] = "UNSCORABLE"
    else:
        score["decision"] = (
            "ACCEPT" if float(distance) <= threshold else "REJECT"
        )
    return score


def _write_outputs(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    METADATA_PATH.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _base_metadata(timestamp: str) -> dict[str, Any]:
    return {
        "model": MODEL,
        "prompt_version": PROMPT_VERSION,
        "target_emotions": list(TARGETS),
        "requested_candidates_per_emotion": CANDIDATES_PER_EMOTION,
        "requested_candidates_total": CANDIDATES_PER_EMOTION * len(TARGETS),
        "generation_settings": {
            **GENERATION_SETTINGS,
            "seed_base": RANDOM_SEED,
            "max_attempts_per_emotion": MAX_ATTEMPTS,
            "exponential_backoff_base_seconds": BACKOFF_BASE_SECONDS,
        },
        "timestamp": timestamp,
        "nrc_vad_version": "NRC-VAD V1 (2018)",
        "distance_metric": DISTANCE_METRIC,
        "threshold_rule": THRESHOLD_RULE,
        "api_request_count": 0,
        "api_failures": [],
        "generation_failures": [],
        "quota_exhausted": False,
        "vad_scoring_failures": [],
        "actual_candidates_generated": 0,
        "generated_count_per_emotion": {emotion: 0 for emotion in TARGETS},
        "accepted_count_per_emotion": {emotion: 0 for emotion in TARGETS},
        "rejected_count_per_emotion": {emotion: 0 for emotion in TARGETS},
        "unscorable_count_per_emotion": {emotion: 0 for emotion in TARGETS},
        "duplicate_or_near_duplicate_pairs": [],
        "synthetic_exact_overlap": 0,
        "goemotions_exact_overlap_by_split": {},
        "protected_hashes_before": {},
        "protected_hashes_after": {},
        "completion_status": "not_started",
    }


def _dry_run() -> int:
    hashes = _protected_hashes()
    existing_texts, source_counts = _read_existing_texts()
    notebook = json.loads(stage1.NOTEBOOK_PATH.read_text(encoding="utf-8"))
    centroids, thresholds = stage1._load_saved_v1_parameters(notebook)
    stage1._load_existing_v1_functions(notebook)
    if set(TARGETS) - set(centroids) or set(TARGETS) - set(thresholds):
        raise ValueError("Existing V1 centroids and thresholds do not cover completion targets.")
    print("SYNTHETIC COMPLETION DRY-RUN - NO GEMINI API CALLS")
    print(f"Model: {MODEL}")
    print(f"Target emotions: {', '.join(TARGETS)}")
    print(f"Requested total: {CANDIDATES_PER_EMOTION * len(TARGETS)} "
          f"({CANDIDATES_PER_EMOTION} per emotion)")
    print(f"Prompt version: {PROMPT_VERSION}")
    print(f"Generation retries: up to {MAX_ATTEMPTS} attempts per emotion with exponential backoff")
    print("VAD scorer: existing NRC-VAD V1 notebook implementation")
    print(f"Existing V1 centroids and P95 thresholds: verified for {', '.join(TARGETS)}")
    print(f"Existing text sources loaded for duplicate/leakage checks: {source_counts}")
    print(
        "Existing normalized source texts available: "
        f"{sum(len(values) for values in existing_texts.values())}"
    )
    print(f"Protected input files fingerprinted: {len(hashes)}")
    print(
        f"Output CSV: {CSV_PATH.relative_to(PROJECT_ROOT)} "
        f"(already exists: {CSV_PATH.exists()})"
    )
    print(
        f"Output metadata: {METADATA_PATH.relative_to(PROJECT_ROOT)} "
        f"(already exists: {METADATA_PATH.exists()})"
    )
    if CSV_PATH.exists() or METADATA_PATH.exists():
        print("Generation mode will refuse to overwrite existing completion outputs.")
    print("API requests: 0")
    print("No files were written.")
    print("To perform generation later, explicitly run with --generate.")
    return 0


def _generate() -> int:
    if CSV_PATH.exists() or METADATA_PATH.exists():
        raise FileExistsError(
            "Completion output already exists; refusing to overwrite: "
            f"{CSV_PATH if CSV_PATH.exists() else METADATA_PATH}"
        )
    hashes_before = _protected_hashes()
    existing_texts, source_counts = _read_existing_texts()
    notebook = json.loads(stage1.NOTEBOOK_PATH.read_text(encoding="utf-8"))
    centroids, thresholds = stage1._load_saved_v1_parameters(notebook)
    vad = stage1._load_existing_v1_functions(notebook)
    timestamp = datetime.now(timezone.utc).isoformat()
    metadata = _base_metadata(timestamp)
    metadata["protected_hashes_before"] = hashes_before
    metadata["existing_text_source_counts"] = source_counts

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not available in the environment or project .env.")

    client = genai.Client(api_key=api_key)
    rows: list[dict[str, Any]] = []
    all_texts: list[str] = []
    for emotion_index, emotion in enumerate(TARGETS):
        if metadata["quota_exhausted"]:
            break
        texts = _generate_for_emotion(
            client,
            emotion,
            request_seed=RANDOM_SEED + emotion_index,
            api_key=api_key,
            metadata=metadata,
        )
        if texts is None:
            continue
        start_index = len([row for row in rows if row["emotion"] == emotion]) + 1
        for local_index, text in enumerate(texts, start=start_index):
            synthetic_id = f"completion-{emotion}-{local_index:03d}"
            try:
                score = _score_text(
                    text,
                    emotion,
                    vad=vad,
                    centroids=centroids,
                    thresholds=thresholds,
                )
            except (KeyError, TypeError, ValueError, ArithmeticError) as error:
                metadata["vad_scoring_failures"].append(
                    {
                        "synthetic_id": synthetic_id,
                        "emotion": emotion,
                        "error": str(error),
                    }
                )
                score = {
                    "valence": None,
                    "arousal": None,
                    "dominance": None,
                    "distance_to_centroid": None,
                    "target_emotion_threshold": thresholds[emotion],
                    "decision": "UNSCORABLE",
                }
            rows.append(
                {
                    "synthetic_id": synthetic_id,
                    "emotion": emotion,
                    "text": text,
                    "source": "synthetic",
                    "model": MODEL,
                    "prompt_version": PROMPT_VERSION,
                    "timestamp": timestamp,
                    **score,
                }
            )
            all_texts.append(text)

    normalized_existing_synthetic = {
        _normalized(text)
        for text in existing_texts["synthetic"]
        if _normalized(text)
    }
    normalized_new = [_normalized(text) for text in all_texts if _normalized(text)]
    metadata["synthetic_exact_overlap"] = sum(
        normalized in normalized_existing_synthetic for normalized in normalized_new
    )
    metadata["goemotions_exact_overlap_by_split"] = {}
    for split in ("train", "validation", "test"):
        frame = pd.read_parquet(DATA_DIRECTORY / f"{split}.parquet", columns=["text"])
        normalized_split = {
            _normalized(text)
            for text in frame["text"].dropna().astype(str)
            if _normalized(text)
        }
        metadata["goemotions_exact_overlap_by_split"][split] = len(
            set(normalized_new) & normalized_split
        )
    metadata["duplicate_or_near_duplicate_pairs"] = _candidate_pairs(
        [str(row["text"]) for row in rows]
    )
    metadata["existing_synthetic_near_duplicate_pairs"] = _candidate_pairs(
        existing_texts["synthetic"] + [str(row["text"]) for row in rows]
    )
    metadata["actual_candidates_generated"] = len(rows)
    metadata["generated_count_per_emotion"] = {
        emotion: sum(row["emotion"] == emotion for row in rows)
        for emotion in TARGETS
    }
    metadata["accepted_count_per_emotion"] = {
        emotion: sum(
            row["emotion"] == emotion and row["decision"] == "ACCEPT" for row in rows
        )
        for emotion in TARGETS
    }
    metadata["rejected_count_per_emotion"] = {
        emotion: sum(
            row["emotion"] == emotion and row["decision"] == "REJECT" for row in rows
        )
        for emotion in TARGETS
    }
    metadata["unscorable_count_per_emotion"] = {
        emotion: sum(
            row["emotion"] == emotion and row["decision"] == "UNSCORABLE" for row in rows
        )
        for emotion in TARGETS
    }
    all_requested = all(
        metadata["generated_count_per_emotion"][emotion] == CANDIDATES_PER_EMOTION
        for emotion in TARGETS
    )
    no_scoring_failures = not metadata["vad_scoring_failures"] and not any(
        metadata["unscorable_count_per_emotion"].values()
    )
    metadata["completion_status"] = (
        "complete"
        if all_requested and no_scoring_failures
        else "partial_due_to_generation_or_scoring_failures"
    )
    metadata["protected_hashes_after"] = _protected_hashes()
    metadata["protected_files_unchanged"] = (
        metadata["protected_hashes_before"] == metadata["protected_hashes_after"]
    )
    _write_outputs(rows, metadata)

    print(f"Generated candidates: {len(rows)}")
    for emotion in TARGETS:
        print(
            f"{emotion}: generated={metadata['generated_count_per_emotion'][emotion]}, "
            f"accepted={metadata['accepted_count_per_emotion'][emotion]}, "
            f"rejected={metadata['rejected_count_per_emotion'][emotion]}, "
            f"unscorable={metadata['unscorable_count_per_emotion'][emotion]}"
        )
    print(f"Gemini API requests: {metadata['api_request_count']}")
    print(f"Generation failures: {len(metadata['generation_failures'])}")
    print(f"VAD scoring failures: {len(metadata['vad_scoring_failures'])}")
    print(f"Near-duplicate candidate pairs: {len(metadata['duplicate_or_near_duplicate_pairs'])}")
    print(f"Exact overlap with existing synthetic: {metadata['synthetic_exact_overlap']}")
    print(f"GoEmotions exact overlap by split: {metadata['goemotions_exact_overlap_by_split']}")
    print(f"Protected inputs unchanged: {metadata['protected_files_unchanged']}")
    print(f"CSV: {CSV_PATH}")
    print(f"Metadata: {METADATA_PATH}")
    output_text = CSV_PATH.read_text(encoding="utf-8") + METADATA_PATH.read_text(encoding="utf-8")
    if api_key in output_text:
        raise AssertionError("API key leakage detected in output files.")
    if not metadata["protected_files_unchanged"]:
        raise AssertionError("A protected input changed during candidate generation.")
    return 0 if all_requested and no_scoring_failures else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Make Gemini API calls and create completion output files; default is a no-call dry-run.",
    )
    args = parser.parse_args()
    if args.generate:
        return _generate()
    return _dry_run()


if __name__ == "__main__":
    raise SystemExit(main())
