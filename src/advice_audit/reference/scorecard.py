"""College Scorecard field-of-study earnings (bachelor's, CIP 4-digit): median across programs."""
from __future__ import annotations

import zipfile
from functools import lru_cache

import pandas as pd

from .sources import path

EARN_COLS = ("EARN_MDN_4YR", "EARN_MDN_HI_2YR", "EARN_MDN_1YR", "EARN_MDN_HI_1YR")


@lru_cache(maxsize=1)
def field_of_study() -> tuple[pd.DataFrame, str]:
    zf = zipfile.ZipFile(path("scorecard_fos"))
    member = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
    head = pd.read_csv(zf.open(member), nrows=0).columns
    col = next(c for c in EARN_COLS if c in head)
    df = pd.read_csv(zf.open(member), usecols=["CIPCODE", "CREDLEV", col], dtype={"CIPCODE": str}, low_memory=False)
    df = df[df["CREDLEV"].isin([3, 5, 6])]  # bachelor's; master's / first-professional as fallback
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df["cip4"] = df["CIPCODE"].str.zfill(4)
    return df.dropna(subset=[col]), col


def median_earnings(cips: list[str]) -> float | None:
    """Median over bachelor's programs; falls back to first-professional (6) then master's (5)
    for fields without bachelor's programs (e.g. dentistry)."""
    df, col = field_of_study()
    keys = {c.replace(".", "")[:4].zfill(4) for c in cips}
    sub = df[df["cip4"].isin(keys)]
    for lev in (3, 6, 5):
        s = sub[sub["CREDLEV"] == lev]
        if not s.empty:
            return float(s[col].median())
    return None
