"""CIP 2020 <-> SOC 2018 (NCES) and SOC 2018 -> ISCO-08 (via BLS ISCO x SOC-2010 + SOC 2010->2018)."""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from .sources import path


def _cip(code: str) -> str:
    s = str(code).strip()
    if "." not in s:
        return s
    a, b = s.split(".")
    return f"{int(a):02d}.{b.ljust(4, '0')[:4]}"


@lru_cache(maxsize=1)
def cip_soc() -> pd.DataFrame:
    df = pd.read_excel(path("cip_soc_crosswalk"), sheet_name="CIP-SOC", dtype=str)
    df["cip"] = df["CIP2020Code"].map(_cip)
    df["soc"] = df["SOC2018Code"].astype(str).str.strip()
    return df[["cip", "soc"]].dropna()


def socs_for_cips(cips: list[str]) -> list[str]:
    t = cip_soc()
    want = {_cip(c) for c in cips}
    socs = t[t["cip"].isin(want)]["soc"].tolist()
    return sorted({s for s in socs if s and s[:2] not in ("99",) and s != "nan"})


def _find_header(df: pd.DataFrame, token: str) -> int:
    for i in range(min(len(df), 30)):
        if any(token.lower() in str(v).lower() for v in df.iloc[i].tolist()):
            return i
    raise ValueError(f"header containing {token!r} not found")


@lru_cache(maxsize=1)
def soc2010_to_2018() -> pd.DataFrame:
    raw = pd.read_excel(path("soc_2010_2018"), header=None, dtype=str)
    h = _find_header(raw, "2010 SOC Code")
    df = raw.iloc[h + 1 :].copy()
    df.columns = [str(c).strip() for c in raw.iloc[h]]
    c10 = next(c for c in df.columns if "2010 SOC Code" in c)
    c18 = next(c for c in df.columns if "2018 SOC Code" in c)
    out = df[[c10, c18]].rename(columns={c10: "soc2010", c18: "soc2018"}).dropna()
    return out.apply(lambda col: col.astype(str).str.strip())


@lru_cache(maxsize=1)
def soc2018_to_isco() -> pd.DataFrame:
    raw = pd.read_excel(path("isco_soc_crosswalk"), sheet_name="2010 SOC to ISCO-08", header=None, dtype=str)
    h = _find_header(raw, "2010 SOC Code")
    df = raw.iloc[h + 1 :].copy()
    df.columns = [str(c).strip() for c in raw.iloc[h]]
    soc_c = next(c for c in df.columns if "2010 SOC Code" in c)
    isco_c = next(c for c in df.columns if "ISCO-08 Code" in c)
    x = df[[soc_c, isco_c]].rename(columns={soc_c: "soc2010", isco_c: "isco08"}).dropna()
    x["soc2010"] = x["soc2010"].astype(str).str.strip()
    x["isco08"] = x["isco08"].astype(str).str.strip().str[:4]
    m = x.merge(soc2010_to_2018(), on="soc2010", how="inner")
    return m[["soc2018", "isco08"]].drop_duplicates()
