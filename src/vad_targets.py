"""NRC-VAD V1 lexicon scoring for lexicon-derived pseudo-targets.

The resulting sentence-level V/A/D values are lexicon-derived pseudo-targets,
not human-annotated sentence-level VAD ground truth. This module uses only
NRC-VAD V1 (2018); it does not use or numerically mix V2.1 scores.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V1_LEXICON_PATH = PROJECT_ROOT / "data" / "NRC-VAD-v1-2018" / "NRC-VAD-Lexicon.txt"
TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?")


@lru_cache(maxsize=1)
def load_vad_lexicon() -> dict[str, tuple[float, float, float]]:
    """Load the original NRC-VAD V1 (2018) word scores into a lookup map."""
    if not V1_LEXICON_PATH.is_file():
        raise FileNotFoundError(f"NRC-VAD V1 lexicon not found: {V1_LEXICON_PATH}")

    lexicon = pd.read_csv(
        V1_LEXICON_PATH,
        sep="\t",
        header=None,
        names=["Word", "Valence", "Arousal", "Dominance"],
    )
    if lexicon.shape[1] != 4:
        raise ValueError(f"Expected four NRC-VAD V1 columns, found {lexicon.shape[1]}.")

    lexicon = lexicon.loc[
        lexicon["Word"].notna() & lexicon["Word"].astype(str).str.strip().ne("")
    ].copy()
    for column in ("Valence", "Arousal", "Dominance"):
        lexicon[column] = pd.to_numeric(lexicon[column], errors="raise")

    return {
        str(row.Word).strip().lower(): (
            float(row.Valence),
            float(row.Arousal),
            float(row.Dominance),
        )
        for row in lexicon.itertuples(index=False)
    }


def score_text_vad(
    text: object,
    lexicon: Mapping[str, tuple[float, float, float]] | None = None,
) -> dict[str, float | int]:
    """Score text using the notebook's V1 tokenizer and matched-token mean.

    Tokens missing from NRC-VAD V1 are ignored. When there are no lexicon
    matches, V/A/D are NaN and coverage is zero for tokenized text (NaN for
    text that yields no tokens), matching the existing notebook behavior.
    These values are pseudo-targets, not human VAD annotations.
    """
    lookup = lexicon if lexicon is not None else load_vad_lexicon()
    tokens = TOKEN_PATTERN.findall(str(text).lower())
    scores = [lookup[token] for token in tokens if token in lookup]
    matched_terms = len(scores)

    if not scores:
        return {
            "vad_valence": float("nan"),
            "vad_arousal": float("nan"),
            "vad_dominance": float("nan"),
            "vad_coverage": 0.0 if tokens else float("nan"),
            "vad_matched_terms": 0,
        }

    return {
        "vad_valence": sum(score[0] for score in scores) / matched_terms,
        "vad_arousal": sum(score[1] for score in scores) / matched_terms,
        "vad_dominance": sum(score[2] for score in scores) / matched_terms,
        "vad_coverage": matched_terms / len(tokens),
        "vad_matched_terms": matched_terms,
    }


def add_vad_targets(
    df: pd.DataFrame,
    text_column: str = "text",
    *,
    lexicon: Mapping[str, tuple[float, float, float]] | None = None,
) -> pd.DataFrame:
    """Return a copy of ``df`` with V1 lexicon-derived pseudo-target columns."""
    if text_column not in df.columns:
        raise KeyError(f"Text column {text_column!r} is not present in the DataFrame.")
    lookup = lexicon if lexicon is not None else load_vad_lexicon()
    scores = [score_text_vad(text, lookup) for text in df[text_column]]
    result = df.copy()
    score_frame = pd.DataFrame(scores, index=df.index)
    for column in score_frame.columns:
        result[column] = score_frame[column]
    return result
