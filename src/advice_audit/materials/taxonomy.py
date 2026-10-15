"""Taxonomy of ~50 major and ~50 occupation categories (spec §3.6).

The same taxonomy codes Part A free text and serves as the candidate list for Part B's
log-prob readout. Reference attributes are filled by `advice_audit.reference.build`
into data/reference/built/taxonomy_<domain>.csv; until then they are NaN.
"""
from __future__ import annotations

import pandas as pd

from ..common.config import REFERENCE_BUILT, load_taxonomy_config

ATTRIBUTES = ("female_share_us", "median_earnings", "prestige_isei", "people_things", "stem")
OTHER = "other"


def load_taxonomy(domain: str, with_attributes: bool = True) -> pd.DataFrame:
    df = pd.DataFrame(load_taxonomy_config(domain))
    if not with_attributes:
        return df
    built = REFERENCE_BUILT / f"taxonomy_{domain}.csv"
    if built.exists():
        attrs = pd.read_csv(built)
        keep = ["id"] + [a for a in ATTRIBUTES if a in attrs.columns and a != "stem"]
        df = df.merge(attrs[keep], on="id", how="left")
    for a in ATTRIBUTES:
        if a not in df.columns:
            df[a] = float("nan")
    return df


def label(row: pd.Series | dict, language: str) -> str:
    return row["label_en"] if language == "en" else row["label_zh"]


def off_track(category_track: str, profile_orientation: str | None) -> float:
    """1 if a category is outside the profile's academic orientation (spec §4.5).

    Majors: science profiles are off-track on humanities-only categories and vice versa;
    'balanced' profiles have no off-track categories (NaN). Jobs: orientation is derived
    from the graduate's major track (see analysis.outcomes)."""
    if profile_orientation in (None, "balanced"):
        return float("nan")
    if category_track == "either":
        return 0.0
    return float(category_track != profile_orientation)


def validate(domain: str) -> list[str]:
    df = load_taxonomy(domain, with_attributes=False)
    problems = []
    if df["id"].duplicated().any():
        problems.append(f"{domain}: duplicate ids {df.loc[df['id'].duplicated(), 'id'].tolist()}")
    for col in ("label_en", "label_zh"):
        if df[col].duplicated().any():
            problems.append(f"{domain}: duplicate {col} {df.loc[df[col].duplicated(), col].tolist()}")
    bad_track = set(df["track"]) - {"science", "humanities", "either"}
    if bad_track:
        problems.append(f"{domain}: unknown track values {bad_track}")
    if OTHER in set(df["id"]):
        problems.append(f"{domain}: 'other' is reserved")
    return problems
