import ast
import csv
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

import httpx
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError

import generate_synthetic_candidates as stage1


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "synthetic"
CSV_PATH = OUTPUT_DIRECTORY / "vad_filtered_candidates_batch2.csv"
METADATA_PATH = OUTPUT_DIRECTORY / "vad_filtered_candidates_batch2.metadata.json"
BATCH1_CSV_PATH = OUTPUT_DIRECTORY / "vad_filtered_candidates_40.csv"
BATCH1_METADATA_PATH = OUTPUT_DIRECTORY / "vad_filtered_candidates_40.metadata.json"

MODEL = "gemini-3.8-flash"
TARGET_EMOTIONS = ("grief", "pride", "relief", "nervousness")
CANDIDATES_PER_EMOTION = 10
PROMPT_VERSION = "rare-emotion-batch2-ambiguous-v1.0"
RANDOM_SEED = 20261008
DISTANCE_METRIC = "3D Euclidean distance in NRC-VAD V1 valence/arousal/dominance space"
THRESHOLD_RULE = "Existing per-emotion NRC-VAD V1 95th percentile of GoEmotions training distances"
GENERATION_SETTINGS = {
    "temperature": 0.7,
    "seed_base": RANDOM_SEED,
    "max_output_tokens": 1800,
    "response_mime_type": "application/json",
}

PROMPT_TEMPLATE = """Generate exactly 10 distinct candidate sentences for an emotion-classification dataset.
The requested target label is: {emotion}

Each sentence should be a realistic, natural English sentence that could plausibly be associated with the target label, while allowing natural ambiguity or mixed context. Keep the target plausible but do not force an extremely explicit expression.

Rules:
- Do not use the target emotion word or a grammatical form of it.
- Do not explicitly explain the emotion or give a textbook definition.
- Use varied everyday situations, experiences, and wording.
- Each candidate must be one self-contained sentence.
- Do not copy, quote, imitate, or reproduce any GoEmotions example.
- Do not repeat or lightly paraphrase another candidate.
- Do not provide VAD scores.
- Return only JSON matching the requested schema, with exactly 10 objects in "candidates", each containing a "text" string.
"""

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

FORBIDDEN_TERMS = {
    "grief": {"grief", "grieve", "grieves", "grieving", "grieved"},
    "pride": {"pride", "proud"},
    "relief": {"relief", "relieve", "relieves", "relieved", "relieving"},
    "nervousness": {"nervousness", "nervous"},
}


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _api_error_message(error, api_key):
    return str(error).replace(api_key, "[REDACTED]")


def _is_503(error):
    return (
        getattr(error, "code", None) == 503
        or getattr(error, "status_code", None) == 503
        or "503 UNAVAILABLE" in str(error)
    )


def _normalize(text):
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _find_near_duplicates(texts):
    duplicate_pairs = []
    for index, text in enumerate(texts):
        left = _normalize(text)
        for other_index in range(index + 1, len(texts)):
            right = _normalize(texts[other_index])
            if left == right or SequenceMatcher(None, left, right).ratio() >= 0.90:
                duplicate_pairs.append((index, other_index))
    return duplicate_pairs


def _parse_response(response_text):
    if not response_text:
        raise ValueError("Gemini returned an empty response.")
    payload = json.loads(response_text)
    if not isinstance(payload, dict) or not isinstance(payload.get("candidates"), list):
        raise ValueError('Response must contain a JSON "candidates" array.')
    candidates = payload["candidates"]
    if len(candidates) != CANDIDATES_PER_EMOTION:
        raise ValueError(
            f"Expected {CANDIDATES_PER_EMOTION} candidates, got {len(candidates)}."
        )
    texts = []
    for index, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, dict) or not isinstance(candidate.get("text"), str):
            raise ValueError(f"Candidate {index} has no string text field.")
        text = candidate["text"].strip()
        if not text:
            raise ValueError(f"Candidate {index} has empty text.")
        texts.append(text)
    return texts


def _validate_texts(texts, emotion, existing_texts):
    forbidden = FORBIDDEN_TERMS[emotion]
    for index, text in enumerate(texts, start=1):
        tokens = set(re.findall(r"[a-z]+", text.lower()))
        if forbidden.intersection(tokens):
            raise ValueError(f"Candidate {index} contains the target word or its variant.")
    if _find_near_duplicates(existing_texts + texts):
        raise ValueError("Candidate batch contains duplicate or near-duplicate text.")


def _make_response_schema():
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


def _write_outputs(rows, metadata):
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    METADATA_PATH.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main():
    timestamp = datetime.now(timezone.utc).isoformat()
    batch1_hashes = {}
    metadata = {
        "model": MODEL,
        "prompt_version": PROMPT_VERSION,
        "target_emotions": list(TARGET_EMOTIONS),
        "requested_candidates_per_emotion": CANDIDATES_PER_EMOTION,
        "requested_candidates_total": CANDIDATES_PER_EMOTION * len(TARGET_EMOTIONS),
        "actual_candidates_generated": 0,
        "generated_count_per_emotion": {emotion: 0 for emotion in TARGET_EMOTIONS},
        "accepted_count_per_emotion": {emotion: 0 for emotion in TARGET_EMOTIONS},
        "rejected_count_per_emotion": {emotion: 0 for emotion in TARGET_EMOTIONS},
        "api_request_count": 0,
        "api_failures": [],
        "generation_failures": [],
        "vad_scoring_failures": [],
        "timestamp": timestamp,
        "nrc_vad_version": "NRC-VAD V1 (2018)",
        "distance_metric": DISTANCE_METRIC,
        "threshold_rule": THRESHOLD_RULE,
        "generation_settings": GENERATION_SETTINGS,
        "completion_status": "not_started",
        "batch1_hashes_before": batch1_hashes,
    }
    rows = []

    try:
        if not BATCH1_CSV_PATH.is_file() or not BATCH1_METADATA_PATH.is_file():
            raise FileNotFoundError("Both Batch 1 output files must exist before running Batch 2.")
        batch1_hashes.update(
            {
                BATCH1_CSV_PATH.name: _sha256(BATCH1_CSV_PATH),
                BATCH1_METADATA_PATH.name: _sha256(BATCH1_METADATA_PATH),
            }
        )
        with BATCH1_CSV_PATH.open(encoding="utf-8", newline="") as handle:
            batch1_rows = list(csv.DictReader(handle))
        existing_texts = [row["text"] for row in batch1_rows]
        batch1_metadata_text = BATCH1_METADATA_PATH.read_text(encoding="utf-8")
        batch1_rows_text = BATCH1_CSV_PATH.read_text(encoding="utf-8")

        load_dotenv(PROJECT_ROOT / ".env")
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing from the environment and project .env file.")

        notebook = json.loads(stage1.NOTEBOOK_PATH.read_text(encoding="utf-8"))
        centroids, thresholds = stage1._load_saved_v1_parameters(notebook)
        vad = stage1._load_existing_v1_functions(notebook)
    except (OSError, json.JSONDecodeError, KeyError, ValueError, SyntaxError) as error:
        metadata["completion_status"] = "failed_pipeline_setup"
        metadata["generation_failures"].append(
            {"stage": "pipeline_setup", "error": str(error)}
        )
        if batch1_hashes:
            metadata["batch1_hashes_after"] = {
                BATCH1_CSV_PATH.name: _sha256(BATCH1_CSV_PATH),
                BATCH1_METADATA_PATH.name: _sha256(BATCH1_METADATA_PATH),
            }
        _write_outputs(rows, metadata)
        print(f"Batch 2 setup failed: {error}")
        print(f"CSV: {CSV_PATH}")
        print(f"Metadata: {METADATA_PATH}")
        return 1

    client = genai.Client(api_key=api_key)
    schema = _make_response_schema()
    batch_texts = []

    for emotion_index, emotion in enumerate(TARGET_EMOTIONS):
        response = None
        for attempt in range(2):
            try:
                metadata["api_request_count"] += 1
                response = client.models.generate_content(
                    model=MODEL,
                    contents=PROMPT_TEMPLATE.format(emotion=emotion),
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=schema,
                        temperature=GENERATION_SETTINGS["temperature"],
                        seed=RANDOM_SEED + emotion_index,
                        max_output_tokens=GENERATION_SETTINGS["max_output_tokens"],
                    ),
                )
                break
            except (APIError, httpx.HTTPError) as error:
                failure = {
                    "emotion": emotion,
                    "attempt": attempt + 1,
                    "error": _api_error_message(error, api_key),
                }
                metadata["api_failures"].append(failure)
                if attempt == 0 and _is_503(error):
                    continue
                response = None
                break

        if response is None:
            continue

        try:
            texts = _parse_response(response.text)
            _validate_texts(texts, emotion, existing_texts + batch_texts)
        except (json.JSONDecodeError, ValueError) as error:
            metadata["generation_failures"].append(
                {"emotion": emotion, "attempt": 1, "error": str(error)}
            )
            continue

        for local_index, text in enumerate(texts, start=1):
            synthetic_id = f"batch2-{emotion}-{local_index:02d}"
            vad_result = vad["calculate_sentence_vad_v1"](text)
            centroid = centroids[emotion]
            distance = vad["calculate_vad_distance"](vad_result, centroid)
            threshold = thresholds[emotion]
            if pd.isna(distance):
                metadata["vad_scoring_failures"].append(
                    {
                        "synthetic_id": synthetic_id,
                        "emotion": emotion,
                        "error": "No V1 lexicon terms matched; no filter decision was possible.",
                    }
                )
                decision = ""
            else:
                decision = eval(
                    vad["_existing_decision_expression"],
                    {"distance": distance, "threshold": threshold},
                )
            rows.append(
                {
                    "synthetic_id": synthetic_id,
                    "emotion": emotion,
                    "text": text,
                    "source": "synthetic",
                    "model": MODEL,
                    "prompt_version": PROMPT_VERSION,
                    "timestamp": timestamp,
                    "valence": vad_result["valence"],
                    "arousal": vad_result["arousal"],
                    "dominance": vad_result["dominance"],
                    "distance_to_centroid": distance,
                    "target_emotion_threshold": threshold,
                    "decision": decision,
                }
            )
            batch_texts.append(text)

    metadata["actual_candidates_generated"] = len(rows)
    metadata["generated_count_per_emotion"] = {
        emotion: sum(row["emotion"] == emotion for row in rows)
        for emotion in TARGET_EMOTIONS
    }
    metadata["accepted_count_per_emotion"] = {
        emotion: sum(row["emotion"] == emotion and row["decision"] == "ACCEPT" for row in rows)
        for emotion in TARGET_EMOTIONS
    }
    metadata["rejected_count_per_emotion"] = {
        emotion: sum(row["emotion"] == emotion and row["decision"] == "REJECT" for row in rows)
        for emotion in TARGET_EMOTIONS
    }
    metadata["unresolved_emotions"] = [
        emotion
        for emotion in TARGET_EMOTIONS
        if metadata["generated_count_per_emotion"][emotion] != CANDIDATES_PER_EMOTION
    ]
    metadata["batch1_hashes_after"] = {
        BATCH1_CSV_PATH.name: _sha256(BATCH1_CSV_PATH),
        BATCH1_METADATA_PATH.name: _sha256(BATCH1_METADATA_PATH),
    }
    metadata["batch1_unchanged"] = (
        metadata["batch1_hashes_before"] == metadata["batch1_hashes_after"]
        and BATCH1_CSV_PATH.read_text(encoding="utf-8") == batch1_rows_text
        and BATCH1_METADATA_PATH.read_text(encoding="utf-8") == batch1_metadata_text
    )
    metadata["duplicate_or_near_duplicate_pairs"] = [
        {"index_1": left, "index_2": right}
        for left, right in _find_near_duplicates([row["text"] for row in rows])
    ]
    metadata["completion_status"] = (
        "complete"
        if len(rows) == CANDIDATES_PER_EMOTION * len(TARGET_EMOTIONS)
        and not metadata["vad_scoring_failures"]
        and not metadata["duplicate_or_near_duplicate_pairs"]
        and metadata["batch1_unchanged"]
        else "partial_due_to_generation_or_validation_failures"
    )
    _write_outputs(rows, metadata)

    print(f"Generated candidates: {len(rows)}")
    for emotion in TARGET_EMOTIONS:
        print(
            f"{emotion}: generated={metadata['generated_count_per_emotion'][emotion]}, "
            f"accepted={metadata['accepted_count_per_emotion'][emotion]}, "
            f"rejected={metadata['rejected_count_per_emotion'][emotion]}"
        )
    decisions = [row["decision"] for row in rows if row["decision"] in ("ACCEPT", "REJECT")]
    accepted = sum(decision == "ACCEPT" for decision in decisions)
    rate = accepted / len(decisions) if decisions else 0.0
    print(f"Overall acceptance rate: {rate:.1%} ({accepted}/{len(decisions)})")
    print(f"API requests: {metadata['api_request_count']}")
    print(f"API failures: {len(metadata['api_failures'])}")
    print(f"Generation failures: {len(metadata['generation_failures'])}")
    print(f"VAD scoring failures: {len(metadata['vad_scoring_failures'])}")
    print(f"Batch 1 unchanged: {metadata['batch1_unchanged']}")
    print(f"CSV: {CSV_PATH}")
    print(f"Metadata: {METADATA_PATH}")

    try:
        csv_text = CSV_PATH.read_text(encoding="utf-8")
        metadata_text = METADATA_PATH.read_text(encoding="utf-8")
        if api_key in csv_text or api_key in metadata_text:
            raise AssertionError("API key found in a Batch 2 output file.")
        if not metadata["batch1_unchanged"]:
            raise AssertionError("Batch 1 files changed during Batch 2.")
        if metadata["duplicate_or_near_duplicate_pairs"]:
            raise AssertionError("Duplicate or near-duplicate candidate texts detected.")
        if any(
            not all(pd.notna(row[dimension]) for dimension in ("valence", "arousal", "dominance"))
            for row in rows
        ):
            raise AssertionError("A generated candidate is missing a VAD value.")
        for row in rows:
            emotion = row["emotion"]
            score = vad["calculate_sentence_vad_v1"](row["text"])
            distance = vad["calculate_vad_distance"](score, centroids[emotion])
            threshold = thresholds[emotion]
            expected_decision = eval(
                vad["_existing_decision_expression"],
                {"distance": distance, "threshold": threshold},
            )
            if not all(
                abs(float(row[dimension]) - score[dimension]) <= 1e-12
                for dimension in ("valence", "arousal", "dominance")
            ):
                raise AssertionError(f"Saved V1 scores failed revalidation for {row['synthetic_id']}.")
            if abs(float(row["distance_to_centroid"]) - distance) > 1e-12:
                raise AssertionError(f"Saved distance failed revalidation for {row['synthetic_id']}.")
            if abs(float(row["target_emotion_threshold"]) - threshold) > 1e-12:
                raise AssertionError(f"Saved threshold failed revalidation for {row['synthetic_id']}.")
            if row["decision"] != expected_decision:
                raise AssertionError(f"Saved decision failed revalidation for {row['synthetic_id']}.")
    except (OSError, AssertionError) as error:
        metadata["completion_status"] = "partial_due_to_generation_or_validation_failures"
        metadata.setdefault("validation_failures", []).append(str(error))
        _write_outputs(rows, metadata)
        print(f"Validation failure: {error}")
        return 1

    return 0 if metadata["completion_status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
