"""Human-coding export (stratified 5% sample) and Cohen's kappa (spec §4.4)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..common.config import OUTPUTS_DIR
from .coder import DISCURSIVE_FIELDS, discursive_codes, source_logs, taxonomy_codes

HUMAN_DIR = OUTPUTS_DIR / "human_coding"
HUMAN_FIELDS = ["top1_category_id", "top2_category_id", "top3_category_id", "warning_unsolicited",
                "ambition_encouragement", "mentions_gender", "mentions_worklife_or_family", "leadership_mention"]


def export_sample(run: str, frac: float = 0.05, seed: int = 20261014) -> Path:
    """Stratify by model x cell x gender condition; blank columns for two human coders.
    LLM codes are written to a separate key file so coders stay blind."""
    rid, src = source_logs(run)
    ok = src[src["status"] == "ok"].copy()
    strata = ["model_key", "cell", "gender_cond"]
    rng = np.random.default_rng(seed)
    shuffled = ok.iloc[rng.permutation(len(ok))]
    quota = shuffled.groupby(strata)["call_id"].transform(lambda s: max(1, int(round(len(s) * frac))))
    sample = shuffled[shuffled.groupby(strata).cumcount() < quota].reset_index(drop=True)
    HUMAN_DIR.mkdir(parents=True, exist_ok=True)
    blind = sample[["call_id", "domain", "mode", "language", "prompt_hash", "response"]].copy()
    blind = blind.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # shuffle: hide strata order
    for coder in ("coder1", "coder2"):
        out = blind.copy()
        for f in HUMAN_FIELDS:
            out[f] = ""
        out.to_csv(HUMAN_DIR / f"{rid}__{coder}.csv", index=False, encoding="utf-8-sig")
    key = sample[["call_id"] + strata].merge(_llm_codes(rid), left_on="call_id", right_on="source_call_id", how="left")
    key.to_csv(HUMAN_DIR / f"{rid}__llm_key.csv", index=False, encoding="utf-8-sig")
    return HUMAN_DIR


def _llm_codes(rid: str) -> pd.DataFrame:
    tax = taxonomy_codes(rid)
    wide = (tax[tax["rank"] <= 3].pivot(index="source_call_id", columns="rank", values="category_id")
            .rename(columns=lambda r: f"top{r}_category_id").reset_index()) if not tax.empty else pd.DataFrame(columns=["source_call_id"])
    disc = discursive_codes(rid)
    if disc.empty:
        disc = pd.DataFrame(columns=["source_call_id", *DISCURSIVE_FIELDS])
    return wide.merge(disc, on="source_call_id", how="outer")


def cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    m = a.notna() & b.notna() & (a.astype(str) != "") & (b.astype(str) != "")
    a, b = a[m].astype(str), b[m].astype(str)
    if len(a) == 0:
        return float("nan")
    cats = sorted(set(a) | set(b))
    po = float((a.values == b.values).mean())
    pa = a.value_counts(normalize=True).reindex(cats, fill_value=0)
    pb = b.value_counts(normalize=True).reindex(cats, fill_value=0)
    pe = float((pa * pb).sum())
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def _norm(v):
    if isinstance(v, str):
        s = v.strip().lower()
        return {"true": "1", "false": "0", "yes": "1", "no": "0", "1.0": "1", "0.0": "0"}.get(s, s)
    if isinstance(v, (bool, np.bool_)):
        return str(int(v))
    if isinstance(v, (int, float)) and not pd.isna(v):
        return str(int(v)) if float(v).is_integer() else str(v)
    return v


def kappa_report(run: str, coder1: Path | None = None, coder2: Path | None = None) -> pd.DataFrame:
    """Kappa human1 vs human2 and human (each) vs LLM, per field and by language."""
    rid, _ = source_logs(run)
    c1 = pd.read_csv(coder1 or HUMAN_DIR / f"{rid}__coder1.csv")
    c2 = pd.read_csv(coder2 or HUMAN_DIR / f"{rid}__coder2.csv")
    key = pd.read_csv(HUMAN_DIR / f"{rid}__llm_key.csv")
    rows = []
    m = c1.merge(c2, on=["call_id", "language"], suffixes=("_h1", "_h2")).merge(key, on="call_id", how="left")
    for lang in ["all", *sorted(m["language"].dropna().unique())]:
        sub = m if lang == "all" else m[m["language"] == lang]
        for f in HUMAN_FIELDS:
            h1, h2 = sub[f + "_h1"].map(_norm), sub[f + "_h2"].map(_norm)
            llm = sub[f].map(_norm) if f in sub.columns else pd.Series(index=sub.index, dtype=object)
            rows.append({"language": lang, "field": f, "n": int(len(sub)),
                         "kappa_h1_h2": cohen_kappa(h1, h2), "kappa_h1_llm": cohen_kappa(h1, llm),
                         "kappa_h2_llm": cohen_kappa(h2, llm)})
    out = pd.DataFrame(rows)
    out.to_csv(HUMAN_DIR / f"{rid}__kappa.csv", index=False)
    return out
