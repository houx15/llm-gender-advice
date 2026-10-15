"""ISEI-08 by ISCO-08 unit group, parsed from Ganzeboom's isco08_with_isei.pdf, and
SOC 2018 -> ISEI (mean over linked ISCO-08 units)."""
from __future__ import annotations

import re
from functools import lru_cache

import pandas as pd

from .crosswalks import soc2018_to_isco
from .sources import path

_LINE = re.compile(r"^\s*(\d{4})\s+(.+?)\s+(\d{2})(?:\s+\S+)?\s*$")


@lru_cache(maxsize=1)
def isei08() -> pd.DataFrame:
    from pypdf import PdfReader

    rows = []
    for page in PdfReader(str(path("isei08"))).pages:
        for line in (page.extract_text() or "").splitlines():
            m = _LINE.match(line)
            if m:
                rows.append({"isco08": m.group(1), "title": m.group(2).strip(), "isei08": int(m.group(3))})
    df = pd.DataFrame(rows).drop_duplicates("isco08")
    if len(df) < 400:
        raise ValueError(f"ISEI parse found only {len(df)} ISCO codes; check the PDF layout")
    return df


def isei_for_isco(isco: str) -> float | None:
    t = isei08().set_index("isco08")["isei08"]
    for code in (isco, isco[:3] + "0", isco[:2] + "00", isco[:1] + "000"):
        if code in t.index:
            return float(t[code])
    return None


def isei_for_socs(socs: list[str]) -> float | None:
    x = soc2018_to_isco()
    iscos = x[x["soc2018"].isin(socs)]["isco08"].unique().tolist()
    vals = [v for v in (isei_for_isco(i) for i in iscos) if v is not None]
    return float(sum(vals) / len(vals)) if vals else None
