"""IPEDS completions (bachelor's, first major) by CIP and sex: female share of degrees,
and the real-world field distributions of women and men (for mirror-vs-amplify)."""
from __future__ import annotations

import zipfile
from functools import lru_cache

import pandas as pd

from .sources import path

BACHELORS = 5
# Fields that are professional/graduate degrees in the US (MD, DDS, PharmD, JD) have few
# bachelor's completions; below this count we fall back to all award levels >= bachelor's
# (5 bachelor's, 7 master's, 17-19 doctor's) and flag it in provenance.
MIN_BACHELORS = 500
LEVELS_FALLBACK = (5, 7, 17, 18, 19)


@lru_cache(maxsize=2)
def completions(levels: tuple[int, ...] = (BACHELORS,)) -> pd.DataFrame:
    zf = zipfile.ZipFile(path("ipeds_completions"))
    member = sorted(n for n in zf.namelist() if n.lower().endswith(".csv") and "_rv" not in n.lower())[0]
    df = pd.read_csv(zf.open(member), dtype={"CIPCODE": str}, usecols=["CIPCODE", "MAJORNUM", "AWLEVEL", "CTOTALM", "CTOTALW"])
    df = df[df["AWLEVEL"].isin(levels) & (df["MAJORNUM"] == 1)]
    df["cip"] = df["CIPCODE"].str.strip()
    df = df[df["cip"].str.contains(r"^\d{2}\.\d{4}$")]
    return df.groupby("cip")[["CTOTALM", "CTOTALW"]].sum().reset_index()


def _match(cips: list[str], levels: tuple[int, ...] = (BACHELORS,)) -> pd.DataFrame:
    t = completions(levels)
    mask = pd.Series(False, index=t.index)
    for c in cips:
        a, b = c.split(".")
        b = b.rstrip("0") if b.endswith("00") else b
        mask |= t["cip"].str.startswith(f"{int(a):02d}.{b}")
    return t[mask]


def female_share(cips: list[str]) -> tuple[float | None, int, str]:
    """(female share, n completions, award levels used)."""
    sub = _match(cips)
    tot = sub["CTOTALM"].sum() + sub["CTOTALW"].sum()
    used = "bachelors"
    if tot < MIN_BACHELORS:
        sub = _match(cips, LEVELS_FALLBACK)
        tot = sub["CTOTALM"].sum() + sub["CTOTALW"].sum()
        used = "bachelors+graduate"
    return (float(sub["CTOTALW"].sum() / tot) if tot else None), int(tot), used


def field_distribution(taxonomy: pd.DataFrame) -> pd.DataFrame:
    """Completions of women and men per taxonomy category (for real-world gaps)."""
    rows = []
    for r in taxonomy.itertuples():
        sub = _match(list(r.codes))
        rows.append({"id": r.id, "women": int(sub["CTOTALW"].sum()), "men": int(sub["CTOTALM"].sum())})
    return pd.DataFrame(rows)
