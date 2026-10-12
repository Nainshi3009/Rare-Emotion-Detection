import ast
import argparse
import csv
import json
import os
import re
from datetime import datetime, timezone
from difflib import SequenceMatcher
from html.parser import HTMLParser
from pathlib import Path

import httpx
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError


PROJECT_ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = PROJECT_ROOT / "Rare_Emotion_Detection_EDA.ipynb"
LEXICON_PATH = PROJECT_ROOT / "data" / "NRC-VAD-v1-2018" / "NRC-VAD-Lexicon.txt"
OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "synthetic"
CSV_PATH = OUTPUT_DIRECTORY / "vad_filtered_candidates_40.csv"
METADATA_PATH = OUTPUT_DIRECTORY / "vad_filtered_candidates_40.metadata.json"

MODEL = "gemini-3.8-flash"
TARGET_EMOTIONS = ("grief", "pride", "relief", "nervousness")
CANDIDATES_PER_EMOTION = 10
PROMPT_VERSION = "rare-emotion-candidates-v1.0"
RANDOM_SEED = 20261008
DISTANCE_METRIC = "3D Euclidean distance in NRC-VAD V1 valence/arousal/dominance space"
THRESHOLD_RULE = "Existing per-emotion NRC-VAD V1 95th percentile of GoEmotions training distances"

PROMPT_TEMPLATE = """Generate exactly 10 distinct candidate sentences for an emotion-classification dataset.
The intended target is: {emotion}

Requirements:
- Each item must be one natural, self-contained English sentence, ideally 10 to 25 words.
- Express only the requested target emotion clearly through a realistic situation.
- Do not use the target emotion name or its grammatical variants in any sentence.
- Vary the people, setting, event, sentence structure, and wording across all 10 items.
- Do not repeat or lightly paraphrase another item.
- Do not quote, imitate, or refer to any existing dataset example.
- Return only JSON matching the requested schema, with exactly 10 objects in "candidates", each containing a "text" string.
"""

CSV_COLUMNS = (
    "synthetic_id",
    "emotion",
    "text",
    "source",
    "generation_method",
    "generation_model",
    "prompt_version",
    "generation_timestamp",
    "valence",
    "arousal",
    "dominance",
    "distance_to_centroid",
    "target_emotion_threshold",
    "decision",
)


class _TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows = []
        self._row = None
        self._cell = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._row = []
        elif tag in ("th", "td") and self._row is not None:
            self._cell = []

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag):
        if tag in ("th", "td") and self._cell is not None:
            self._row.append("".join(self._cell).strip())
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None


def _notebook_cell(notebook, index):
    cell = notebook["cells"][index]
    if cell.get("cell_type") != "code" or cell.get("execution_count") is None:
        raise ValueError(f"Notebook cell {index} is not a previously executed code cell.")
    return cell


def _display_table(notebook, cell_index, display_index):
    cell = _notebook_cell(notebook, cell_index)
    displays = [
        output.get("data", {}).get("text/html")
        for output in cell.get("outputs", [])
        if output.get("output_type") == "display_data"
        and output.get("data", {}).get("text/html")
    ]
    if len(displays) <= display_index:
        raise ValueError(
            f"Notebook cell {cell_index} is missing its saved table at index {display_index}."
        )

    parser = _TableParser()
    display_html = displays[display_index]
    if isinstance(display_html, list):
        display_html = "".join(display_html)
    parser.feed(display_html)
    if not parser.rows:
        raise ValueError(f"Could not read the saved table in notebook cell {cell_index}.")

    headers = parser.rows[0]
    if headers and not headers[0]:
        headers = headers[1:]
    records = []
    for row in parser.rows[1:]:
        if row and not row[0].isdigit() and len(row) == len(headers):
            records.append(dict(zip(headers, row)))
        elif row and row[0].isdigit() and len(row[1:]) == len(headers):
            records.append(dict(zip(headers, row[1:])))
    if not records:
        raise ValueError(f"Saved table in notebook cell {cell_index} has no data rows.")
    return records


def _load_saved_v1_parameters(notebook):
    centroid_records = _display_table(notebook, 29, 0)
    threshold_records = _display_table(notebook, 36, 1)
    centroids = {}
    for row in centroid_records:
        emotion = row["emotion"]
        centroids[emotion] = {
            dimension: float(row[dimension])
            for dimension in ("valence", "arousal", "dominance")
        }

    thresholds = {}
    for row in threshold_records:
        if int(row["calibration_percentile"]) != 95:
            raise ValueError("Saved V1 threshold table is not the existing 95th-percentile table.")
        thresholds[row["emotion"]] = float(row["threshold"])

    if set(centroids) != set(TARGET_EMOTIONS) or set(thresholds) != set(TARGET_EMOTIONS):
        raise ValueError("Saved notebook centroids or thresholds do not cover the four targets.")
    return centroids, thresholds


def _find_function(tree, name):
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise ValueError(f"Required existing notebook function was not found: {name}")


def _find_assignment(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return node
    raise ValueError(f"Required existing notebook assignment was not found: {name}")


def _load_existing_v1_functions(notebook):
    scorer_tree = ast.parse("".join(_notebook_cell(notebook, 26)["source"]))
    distance_tree = ast.parse("".join(_notebook_cell(notebook, 35)["source"]))
    filter_tree = ast.parse("".join(_notebook_cell(notebook, 40)["source"]))

    score_nodes = [
        _find_function(scorer_tree, "_tokenize_english"),
        _find_assignment(scorer_tree, "_v1_lookup"),
        _find_function(scorer_tree, "calculate_sentence_vad_v1"),
    ]
    distance_function = _find_function(distance_tree, "calculate_vad_distance")
    decision_assignment = _find_assignment(filter_tree, "decision")
    if not isinstance(decision_assignment.value, ast.IfExp):
        raise ValueError("Existing notebook ACCEPT/REJECT expression has changed.")

    vad_v1_df = pd.read_csv(
        LEXICON_PATH,
        sep="\t",
        header=None,
        names=["Word", "Valence", "Arousal", "Dominance"],
    )
    namespace = {"pd": pd, "re": re, "math": __import__("math"), "vad_v1_df": vad_v1_df}
    score_module = ast.Module(body=score_nodes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(score_module), str(NOTEBOOK_PATH), "exec"), namespace)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[distance_function], type_ignores=[])),
            str(NOTEBOOK_PATH),
            "exec",
        ),
        namespace,
    )
    namespace["_existing_decision_expression"] = compile(
        ast.Expression(body=decision_assignment.value),
        str(NOTEBOOK_PATH),
        "eval",
    )
    return namespace


def _parse_generated_candidates(response_text):
    if not response_text:
        raise ValueError("Gemini returned an empty response.")
    result = json.loads(response_text)
    if not isinstance(result, dict) or not isinstance(result.get("candidates"), list):
        raise ValueError('Gemini response must be a JSON object containing "candidates".')
    candidates = result["candidates"]
    if len(candidates) != CANDIDATES_PER_EMOTION:
        raise ValueError(
            f"Expected {CANDIDATES_PER_EMOTION} candidates; Gemini returned {len(candidates)}."
        )
    texts = []
    for index, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, dict) or not isinstance(candidate.get("text"), str):
            raise ValueError(f"Candidate {index} is missing a string text field.")
        text = candidate["text"].strip()
        if not text:
            raise ValueError(f"Candidate {index} has empty text.")
        texts.append(text)
    return texts


def _normalized_text(text):
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _find_near_duplicates(rows):
    duplicates = []
    for index, row in enumerate(rows):
        left = _normalized_text(row["text"])
        for other in rows[index + 1 :]:
            right = _normalized_text(other["text"])
            if left == right or SequenceMatcher(None, left, right).ratio() >= 0.90:
                duplicates.append((row["synthetic_id"], other["synthetic_id"]))
    return duplicates


def _error_message(error, api_key):
    return str(error).replace(api_key, "[REDACTED]")


def _is_service_unavailable(error):
    return (
        getattr(error, "code", None) == 503
        or getattr(error, "status_code", None) == 503
        or "503 UNAVAILABLE" in str(error)
    )


def _write_outputs(rows, metadata):
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    METADATA_PATH.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep valid candidates already saved and request only missing target batches.",
    )
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).isoformat()
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    metadata = {
        "generation_model": MODEL,
        "number_of_candidates_requested": CANDIDATES_PER_EMOTION * len(TARGET_EMOTIONS),
        "target_emotions": list(TARGET_EMOTIONS),
        "candidates_per_emotion": CANDIDATES_PER_EMOTION,
        "prompt_version": PROMPT_VERSION,
        "timestamp": timestamp,
        "vad_lexicon_version": "NRC-VAD V1 (2018)",
        "distance_metric": DISTANCE_METRIC,
        "threshold_rule": THRESHOLD_RULE,
        "random_seed": RANDOM_SEED,
        "generation_api_requests": 0,
        "successful_target_batches": [],
        "generation_failures": [],
        "vad_scoring_failures": [],
        "near_duplicate_pairs": [],
    }
    rows = []

    try:
        if args.resume and CSV_PATH.is_file() and METADATA_PATH.is_file():
            previous_metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
            if (
                previous_metadata.get("generation_model") != MODEL
                or previous_metadata.get("prompt_version") != PROMPT_VERSION
                or previous_metadata.get("target_emotions") != list(TARGET_EMOTIONS)
                or previous_metadata.get("candidates_per_emotion") != CANDIDATES_PER_EMOTION
            ):
                raise ValueError("Existing output metadata does not match this generation run.")
            with CSV_PATH.open(encoding="utf-8", newline="") as csv_file:
                rows.extend(csv.DictReader(csv_file))
            if any(row.get("emotion") not in TARGET_EMOTIONS for row in rows):
                raise ValueError("Existing CSV contains an unexpected target emotion.")
            if any(
                sum(row["emotion"] == emotion for row in rows) > CANDIDATES_PER_EMOTION
                for emotion in TARGET_EMOTIONS
            ):
                raise ValueError("Existing CSV contains more candidates than requested for an emotion.")
            metadata.update(previous_metadata)
            metadata["timestamp"] = previous_metadata.get("timestamp", timestamp)
            metadata["generation_failures"] = previous_metadata.get("generation_failures", [])
            metadata["vad_scoring_failures"] = previous_metadata.get("vad_scoring_failures", [])
            previous_successes = previous_metadata.get("successful_target_batches")
            if previous_successes is None:
                previous_counts = previous_metadata.get("generated_per_emotion", {})
                previous_successes = [
                    emotion
                    for emotion in TARGET_EMOTIONS
                    if previous_counts.get(emotion) == CANDIDATES_PER_EMOTION
                ]
            metadata["successful_target_batches"] = previous_successes
            metadata["generation_api_requests"] = max(
                previous_metadata.get("generation_api_requests", 0),
                len(metadata["generation_failures"]) + len(previous_successes),
            )
        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing from the environment and project .env file.")
        if not NOTEBOOK_PATH.is_file() or not LEXICON_PATH.is_file():
            raise FileNotFoundError("The existing EDA notebook or NRC-VAD V1 lexicon is missing.")

        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        centroids, thresholds = _load_saved_v1_parameters(notebook)
        vad = _load_existing_v1_functions(notebook)
    except (OSError, json.JSONDecodeError, KeyError, ValueError, SyntaxError) as error:
        metadata["generation_failures"].append(
            {"stage": "pipeline_setup", "error": _error_message(error, api_key)}
        )
        _write_outputs(rows, metadata)
        print(f"Generated candidate rows: {len(rows)}")
        print(f"Generation failures: {len(metadata['generation_failures'])}")
        print(f"CSV: {CSV_PATH}")
        print(f"Metadata: {METADATA_PATH}")
        return 1

    client = genai.Client(api_key=api_key)
    response_schema = {
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

    for emotion_index, emotion in enumerate(TARGET_EMOTIONS):
        existing_count = sum(row["emotion"] == emotion for row in rows)
        if existing_count >= CANDIDATES_PER_EMOTION:
            continue
        response = None
        for attempt in range(2):
            try:
                metadata["generation_api_requests"] += 1
                response = client.models.generate_content(
                    model=MODEL,
                    contents=PROMPT_TEMPLATE.format(emotion=emotion),
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=response_schema,
                        temperature=0.7,
                        seed=RANDOM_SEED + emotion_index,
                        max_output_tokens=1800,
                    ),
                )
                break
            except (APIError, httpx.HTTPError) as error:
                metadata["generation_failures"].append(
                    {
                        "emotion": emotion,
                        "attempt": attempt + 1,
                        "error": _error_message(error, api_key),
                    }
                )
                if attempt == 0 and _is_service_unavailable(error):
                    continue
                response = None
                break

        if response is None:
            continue

        try:
            texts = _parse_generated_candidates(response.text if response else None)
            forbidden_terms = {
                "grief": {"grief", "grieve", "grieves", "grieving", "grieved"},
                "pride": {"pride", "proud"},
                "relief": {"relief", "relieve", "relieves", "relieved", "relieving"},
                "nervousness": {"nervousness", "nervous"},
            }[emotion]
            for text_index, text in enumerate(texts, start=1):
                if forbidden_terms.intersection(re.findall(r"[a-z]+", text.lower())):
                    raise ValueError(
                        f"Candidate {text_index} names the target emotion or a grammatical variant."
                    )
            if _find_near_duplicates(
                [
                    {"synthetic_id": f"{emotion}-{index:03d}", "text": text}
                    for index, text in enumerate(texts, start=existing_count + 1)
                ]
            ):
                raise ValueError("Gemini returned duplicate or near-duplicate candidates in the batch.")
            metadata["successful_target_batches"].append(emotion)
            for text in texts:
                existing_count += 1
                synthetic_id = f"{emotion}-{existing_count:03d}"

                vad_result = vad["calculate_sentence_vad_v1"](text)
                centroid = centroids[emotion]
                distance = vad["calculate_vad_distance"](vad_result, centroid)
                threshold = thresholds[emotion]
                if pd.isna(distance):
                    metadata["vad_scoring_failures"].append(
                        {"synthetic_id": synthetic_id, "emotion": emotion, "error": "No V1 lexicon matches."}
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
                        "generation_method": f"Gemini API ({PROMPT_VERSION})",
                        "generation_model": MODEL,
                        "prompt_version": PROMPT_VERSION,
                        "generation_timestamp": timestamp,
                        "valence": vad_result["valence"],
                        "arousal": vad_result["arousal"],
                        "dominance": vad_result["dominance"],
                        "distance_to_centroid": distance,
                        "target_emotion_threshold": threshold,
                        "decision": decision,
                    }
                )
        except (APIError, httpx.HTTPError, json.JSONDecodeError, ValueError) as error:
            metadata["generation_failures"].append(
                {"emotion": emotion, "attempt": 1, "error": _error_message(error, api_key)}
            )

    duplicate_pairs = _find_near_duplicates(rows)
    metadata["near_duplicate_pairs"] = [
        {"synthetic_id_1": left, "synthetic_id_2": right}
        for left, right in duplicate_pairs
    ]
    metadata["candidates_generated"] = len(rows)
    metadata["total_successful_candidates"] = len(rows)
    metadata["generated_per_emotion"] = {
        emotion: sum(row["emotion"] == emotion for row in rows)
        for emotion in TARGET_EMOTIONS
    }
    metadata["successful_target_batches"] = [
        emotion
        for emotion in TARGET_EMOTIONS
        if metadata["generated_per_emotion"][emotion] == CANDIDATES_PER_EMOTION
    ]
    metadata["generation_api_requests"] = max(
        metadata["generation_api_requests"],
        len(metadata["generation_failures"]) + len(metadata["successful_target_batches"]),
    )
    metadata["accepted_per_emotion"] = {
        emotion: sum(row["emotion"] == emotion and row["decision"] == "ACCEPT" for row in rows)
        for emotion in TARGET_EMOTIONS
    }
    metadata["rejected_per_emotion"] = {
        emotion: sum(row["emotion"] == emotion and row["decision"] == "REJECT" for row in rows)
        for emotion in TARGET_EMOTIONS
    }
    metadata["unresolved_generation_failures"] = [
        emotion
        for emotion in TARGET_EMOTIONS
        if metadata["generated_per_emotion"][emotion] != CANDIDATES_PER_EMOTION
    ]

    _write_outputs(rows, metadata)
    print(f"Generated candidate rows: {len(rows)}")
    for emotion in TARGET_EMOTIONS:
        print(
            f"{emotion}: generated={metadata['generated_per_emotion'][emotion]}, "
            f"accepted={metadata['accepted_per_emotion'][emotion]}, "
            f"rejected={metadata['rejected_per_emotion'][emotion]}"
        )
    decided = sum(row["decision"] in ("ACCEPT", "REJECT") for row in rows)
    accepted = sum(row["decision"] == "ACCEPT" for row in rows)
    acceptance_rate = accepted / decided if decided else 0.0
    print(f"Overall acceptance rate: {acceptance_rate:.1%} ({accepted}/{decided})")
    print(f"API/generation failures recorded: {len(metadata['generation_failures'])}")
    print(f"Unresolved target batches: {len(metadata['unresolved_generation_failures'])}")
    print(f"VAD scoring failures: {len(metadata['vad_scoring_failures'])}")
    print(f"Near-duplicate pairs: {len(duplicate_pairs)}")
    print(f"CSV: {CSV_PATH}")
    print(f"Metadata: {METADATA_PATH}")

    valid = (
        len(rows) == CANDIDATES_PER_EMOTION * len(TARGET_EMOTIONS)
        and all(metadata["generated_per_emotion"][emotion] == CANDIDATES_PER_EMOTION for emotion in TARGET_EMOTIONS)
        and all(row["text"].strip() and row["source"] == "synthetic" for row in rows)
        and len({_normalized_text(row["text"]) for row in rows}) == len(rows)
        and not metadata["vad_scoring_failures"]
        and not duplicate_pairs
    )
    return 0 if valid and not metadata["unresolved_generation_failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
