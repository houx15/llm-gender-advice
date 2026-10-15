"""BLS CPS Table 11 (female share by occupation) and OEWS national medians (by SOC)."""
from __future__ import annotations

import io
import zipfile

import pandas as pd
import yaml

from ..common.config import CONFIG_DIR
from .sources import path


def cps_table() -> pd.DataFrame:
    t = pd.read_html(path("cps_table11"))[0]
    flat = [c[-1] if isinstance(c, tuple) else str(c) for c in t.columns]
    t.columns = flat
    out = pd.DataFrame({
        "title": t.iloc[:, 0].astype(str).str.strip(),
        "employed_k": pd.to_numeric(t["Total employed"].astype(str).str.replace(",", ""), errors="coerce"),
        "pct_women": pd.to_numeric(t["Women"].astype(str).str.replace(",", ""), errors="coerce"),
    })
    return out.dropna(subset=["pct_women"]).drop_duplicates("title")


def cps_female_share_by_job() -> tuple[dict[str, float], list[str]]:
    """{job_id: employment-weighted female share}; problems list for unmatched titles."""
    mapping = yaml.safe_load((CONFIG_DIR / "taxonomy" / "cps_titles.yaml").read_text(encoding="utf-8"))
    t = cps_table().set_index("title")
    out, problems = {}, []
    for jid, titles in mapping.items():
        rows = [t.loc[x] for x in titles if x in t.index]
        missing = [x for x in titles if x not in t.index]
        if missing:
            problems.append(f"CPS: titles not found for {jid}: {missing}")
        if rows:
            df = pd.DataFrame(rows)
            w = df["employed_k"].fillna(1.0)
            out[jid] = float((df["pct_women"] * w).sum() / w.sum() / 100.0)
    return out, problems


def oews_national() -> pd.DataFrame:
    zf = zipfile.ZipFile(path("oews_national"))
    member = [n for n in zf.namelist() if n.lower().endswith((".xlsx", ".xls"))][0]
    df = pd.read_excel(io.BytesIO(zf.read(member)), dtype={"OCC_CODE": str})
    df.columns = [c.upper() for c in df.columns]
    if "O_GROUP" in df.columns:
        df = df[df["O_GROUP"].isin(["detailed", "broad"])]
    df["A_MEDIAN"] = pd.to_numeric(df["A_MEDIAN"].astype(str).str.replace(",", ""), errors="coerce")
    df["TOT_EMP"] = pd.to_numeric(df["TOT_EMP"].astype(str).str.replace(",", ""), errors="coerce")
    return df[["OCC_CODE", "OCC_TITLE", "A_MEDIAN", "TOT_EMP"]]


def oews_median(socs: list[str], table: pd.DataFrame | None = None) -> float | None:
    """Employment-weighted mean of annual medians over the listed SOC codes."""
    t = table if table is not None else oews_national()
    sub = t[t["OCC_CODE"].isin(socs)].dropna(subset=["A_MEDIAN"])
    if sub.empty:
        return None
    w = sub["TOT_EMP"].fillna(1.0)
    return float((sub["A_MEDIAN"] * w).sum() / w.sum())
