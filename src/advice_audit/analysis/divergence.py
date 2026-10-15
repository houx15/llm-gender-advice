"""H7 male default: Jensen–Shannon divergence of the baseline category distribution from
the female and male distributions, per profile × cell × model, with bootstrap CIs."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .outcomes import RANK_WEIGHTS


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen–Shannon divergence (base 2, in [0, 1])."""
    p = p / p.sum() if p.sum() else p
    q = q / q.sum() if q.sum() else q
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def call_matrix(codes: pd.DataFrame) -> tuple[pd.Index, list[str], np.ndarray]:
    """Position-weighted category vector per response: (call index, categories, matrix)."""
    w = codes.assign(w=codes["rank"].map(RANK_WEIGHTS).astype(float))
    piv = w.pivot_table(index="source_call_id", columns="category_id", values="w", aggfunc="sum", fill_value=0.0)
    return piv.index, list(piv.columns), piv.values


def h7_table(outcomes: pd.DataFrame, codes: pd.DataFrame, n_boot: int = 200, seed: int = 0) -> pd.DataFrame:
    """Rows per (domain, mode, profile, cell, model): JSD(base,F), JSD(base,M), diff = F - M.
    diff > 0 means the baseline is closer to the male distribution (male default)."""
    rng = np.random.default_rng(seed)
    idx, cats, mat = call_matrix(codes)
    pos = pd.Series(np.arange(len(idx)), index=idx)
    rows = []
    keys = ["domain", "mode", "profile_id", "cell", "model"]
    for k, g in outcomes[outcomes["call_id"].isin(idx)].groupby(keys):
        gc = g["gender_cond"]
        ids = {"base": pos[g.loc[gc.eq("baseline"), "call_id"]].values,
               "f": pos[g.loc[gc.str.startswith("f_"), "call_id"]].values,
               "m": pos[g.loc[gc.str.startswith("m_"), "call_id"]].values}
        if any(len(v) == 0 for v in ids.values()):
            continue
        dist = {grp: mat[v].sum(axis=0) for grp, v in ids.items()}
        if any(d.sum() == 0 for d in dist.values()):
            continue
        jf, jm = jsd(dist["base"], dist["f"]), jsd(dist["base"], dist["m"])
        boots = []
        for _ in range(n_boot):
            bs = {grp: mat[rng.choice(v, size=len(v), replace=True)].sum(axis=0) for grp, v in ids.items()}
            if all(d.sum() > 0 for d in bs.values()):
                boots.append(jsd(bs["base"], bs["f"]) - jsd(bs["base"], bs["m"]))
        lo, hi = (np.percentile(boots, [2.5, 97.5]) if len(boots) > 20 else (np.nan, np.nan))
        rows.append({**dict(zip(keys, k)), "jsd_base_f": jf, "jsd_base_m": jm, "diff_f_minus_m": jf - jm,
                     "diff_lo": lo, "diff_hi": hi, "n_base": len(ids["base"]), "n_f": len(ids["f"]), "n_m": len(ids["m"])})
    return pd.DataFrame(rows)


def h7_summary(h7: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Mean diff across profile × cell cells, bootstrap over cells, by domain × mode × model."""
    rng = np.random.default_rng(seed)
    rows = []
    for k, g in h7.groupby(["domain", "mode", "model"]):
        v = g["diff_f_minus_m"].values
        boots = [rng.choice(v, size=len(v), replace=True).mean() for _ in range(n_boot)]
        rows.append({"domain": k[0], "mode": k[1], "model": k[2], "mean_diff": float(v.mean()),
                     "lo": float(np.percentile(boots, 2.5)), "hi": float(np.percentile(boots, 97.5)),
                     "share_male_default": float((v > 0).mean()), "n_cells": len(v)})
    return pd.DataFrame(rows)
