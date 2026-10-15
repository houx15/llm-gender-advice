"""Per-response outcomes (spec §4.5).

Position-weighted (5,4,3,2,1, renormalized over the ranks that have a value) means over
the coded recommendations of female_share_us, median_earnings, prestige_isei,
people_things, stem; off_track_share (unweighted share of coded recommendations outside
the profile's orientation); discursive codes; SCM warmth/competence counts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..coding.coder import discursive_codes, source_logs, taxonomy_codes
from ..coding.dictionaries import scm_counts
from ..materials.taxonomy import ATTRIBUTES, OTHER, load_taxonomy, off_track

RANK_WEIGHTS = {1: 5, 2: 4, 3: 3, 4: 2, 5: 1}
OUTCOMES_HORIZONTAL = ("female_share_us", "people_things", "stem", "off_track_share")
OUTCOMES_VERTICAL = ("median_earnings", "prestige_isei")
OUTCOMES_DISCURSIVE = ("warning_unsolicited", "hedge_count", "ambition_encouragement", "mentions_gender",
                       "mentions_worklife_or_family", "leadership_mention", "warmth_count", "competence_count",
                       "warmth_minus_competence")


def weighted_mean(values: pd.Series, ranks: pd.Series) -> float:
    m = values.notna()
    if not m.any():
        return float("nan")
    w = ranks[m].map(RANK_WEIGHTS).astype(float)
    return float((values[m] * w).sum() / w.sum())


def response_outcomes(logs: pd.DataFrame, codes: pd.DataFrame, disc: pd.DataFrame | None = None,
                      with_scm: bool = True) -> pd.DataFrame:
    """logs: Part-A-schema rows (status ok); codes: taxonomy_codes(); disc: discursive_codes()."""
    ok = logs[logs["status"] == "ok"].copy()
    attrs = {d: load_taxonomy(d).set_index("id") for d in ("majors", "jobs")}
    rows = []
    codes_by_call = {k: g for k, g in codes.groupby("source_call_id")} if not codes.empty else {}
    for r in ok.itertuples():
        g = codes_by_call.get(r.call_id)
        row = {"call_id": r.call_id, "n_recs": 0, "n_other": 0}
        if g is not None and len(g):
            g = g.sort_values("rank")
            row["n_recs"] = int(len(g))
            row["n_other"] = int((g["category_id"] == OTHER).sum())
            row["top1_category"] = g["category_id"].iloc[0]
            known = g[g["category_id"] != OTHER]
            a = attrs[r.domain]
            for att in ATTRIBUTES:
                vals = known["category_id"].map(a[att]) if len(known) else pd.Series(dtype=float)
                row[att] = weighted_mean(vals.astype(float), known["rank"]) if len(known) else float("nan")
            ot = [off_track(a.loc[c, "track"], r.orientation) for c in known["category_id"]]
            ot = [x for x in ot if not np.isnan(x)]
            row["off_track_share"] = float(np.mean(ot)) if ot else float("nan")
            if r.mode == "fixed" and getattr(r, "fixed_field", ""):
                fld = known["category_id"].map(a["field"])
                row["in_field_share"] = float((fld == r.fixed_field).mean()) if len(known) else float("nan")
        if with_scm:
            row.update(scm_counts(r.response, r.language))
        rows.append(row)
    out = ok.drop(columns=[c for c in ("prompt",) if c in ok.columns]).merge(pd.DataFrame(rows), on="call_id", how="left")
    if disc is not None and not disc.empty:
        out = out.merge(disc.drop(columns=["coder_key", "warning_quote"], errors="ignore"),
                        left_on="call_id", right_on="source_call_id", how="left").drop(columns=["source_call_id"])
    for c in OUTCOMES_DISCURSIVE + OUTCOMES_HORIZONTAL + OUTCOMES_VERTICAL:
        if c not in out.columns:
            out[c] = np.nan
    if "model_key" in out.columns:
        out["model"] = out["model_key"]
    return out


def outcomes_for_runs(runs: list[str], with_scm: bool = True) -> pd.DataFrame:
    frames = []
    for run in runs:
        rid, logs = source_logs(run)
        frames.append(response_outcomes(logs, taxonomy_codes(rid), discursive_codes(rid), with_scm=with_scm))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
