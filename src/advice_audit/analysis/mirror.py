"""Mirror vs amplify (spec §4.6): compare the model's female–male difference in the
female share of recommended fields with the real-world difference between the fields
women and men actually enter (US: IPEDS completions for majors, CPS employment for jobs;
China: CFPS once added). Real-world distributions are restricted to taxonomy categories.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..materials.taxonomy import load_taxonomy


def real_world_gap(domain: str) -> dict:
    tax = load_taxonomy(domain)
    if domain == "majors":
        from ..reference.ipeds import field_distribution

        dist = field_distribution(tax)
    else:
        import yaml

        from ..common.config import CONFIG_DIR
        from ..reference.bls import cps_table

        t = cps_table().set_index("title")
        mapping = yaml.safe_load((CONFIG_DIR / "taxonomy" / "cps_titles.yaml").read_text(encoding="utf-8"))
        rows = []
        for jid, titles in mapping.items():
            sub = t.loc[[x for x in titles if x in t.index]]
            emp = sub["employed_k"].fillna(0)
            women = float((emp * sub["pct_women"] / 100).sum())
            rows.append({"id": jid, "women": women, "men": float(emp.sum()) - women})
        dist = pd.DataFrame(rows)
    d = dist.merge(tax[["id", "female_share_us"]], on="id").dropna(subset=["female_share_us"])
    e_w = float((d["women"] * d["female_share_us"]).sum() / d["women"].sum())
    e_m = float((d["men"] * d["female_share_us"]).sum() / d["men"].sum())
    return {"domain": domain, "real_E_fs_women": e_w, "real_E_fs_men": e_m, "real_gap": e_w - e_m}


def mirror_table(outcomes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    o = outcomes[(outcomes["mode"] == "open") & (outcomes["gender_cond"] != "baseline")]
    for domain, dd in o.groupby("domain"):
        try:
            rw = real_world_gap(domain)
        except Exception as e:  # noqa: BLE001
            rw = {"domain": domain, "real_gap": np.nan, "error": str(e)}
        for (model, cell), g in dd.groupby(["model", "cell"]):
            f = g.loc[g["female"] == 1, "female_share_us"].mean()
            m = g.loc[g["female"] == 0, "female_share_us"].mean()
            gap = f - m
            rows.append({**rw, "model": model, "cell": cell, "model_gap": gap,
                         "ratio_model_to_real": gap / rw["real_gap"] if rw.get("real_gap") else np.nan,
                         "benchmark": "US (IPEDS/CPS)" + (" — China cell; CFPS benchmark pending" if cell == "C" else "")})
    return pd.DataFrame(rows)
