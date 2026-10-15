"""Warmth and competence word counts from Stereotype Content Model dictionaries (spec §4.4).

Expected files (NOT shipped; see QUESTIONS.md):
  config/dictionaries/scm_en.csv   columns: term,dimension[,direction]
  config/dictionaries/scm_zh.csv   same columns; Chinese translation, flagged for validation
`dimension` in {warmth, competence}; `direction` in {high, low} (optional; default high).
English terms may end in '*' (prefix match). Chinese terms are matched as substrings.
If a dictionary is missing, counts are NaN (never invented).
"""
from __future__ import annotations

import re
from functools import lru_cache

import pandas as pd

from ..common.config import CONFIG_DIR

_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")


@lru_cache(maxsize=None)
def load_dictionary(language: str) -> pd.DataFrame | None:
    path = CONFIG_DIR / "dictionaries" / f"scm_{language}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["term"] = df["term"].astype(str).str.strip().str.lower()
    if "direction" not in df.columns:
        df["direction"] = "high"
    return df[df["dimension"].isin(["warmth", "competence"])]


def _count_en(text: str, terms: list[str]) -> int:
    words = [w.lower() for w in _WORD.findall(text or "")]
    exact = {t for t in terms if not t.endswith("*")}
    prefixes = tuple(t[:-1] for t in terms if t.endswith("*"))
    return sum(1 for w in words if w in exact or (prefixes and w.startswith(prefixes)))


def _count_zh(text: str, terms: list[str]) -> int:
    return sum((text or "").count(t) for t in terms if t)


def scm_counts(text: str, language: str) -> dict:
    d = load_dictionary(language)
    nan = float("nan")
    if d is None:
        return {"warmth_count": nan, "competence_count": nan, "n_words": nan, "warmth_minus_competence": nan}
    count = _count_en if language == "en" else _count_zh
    w = count(text, d.loc[(d["dimension"] == "warmth") & (d["direction"] == "high"), "term"].tolist())
    c = count(text, d.loc[(d["dimension"] == "competence") & (d["direction"] == "high"), "term"].tolist())
    n = len(_WORD.findall(text or "")) if language == "en" else len(re.sub(r"\s", "", text or "")) / 1.5
    per100 = 100.0 / n if n else nan
    return {"warmth_count": float(w), "competence_count": float(c), "n_words": float(n),
            "warmth_minus_competence": (w - c) * per100 if n else nan}
