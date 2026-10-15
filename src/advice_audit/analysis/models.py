"""Study 1 regressions (spec §4.6).

Core: Y ~ female * explicit + female:C(cell) + C(profile_id) + C(cell) + C(model),
SEs clustered by profile, baseline excluded. Terms with a single level are dropped
(e.g. C(model) in a one-model pilot). With few clusters (12 profiles) cluster-robust SEs
are optimistic; a wild-cluster bootstrap is a recommended robustness check.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

CORE = "female * explicit + female:C(cell) + C(profile_id) + C(cell) + C(model)"


def _formula(df: pd.DataFrame, y: str, rhs: str = CORE) -> str:
    terms = [t.strip() for t in rhs.split("+")]
    keep = []
    for t in terms:
        if t == "C(model)" and df["model"].nunique() < 2:
            continue
        if t in ("C(cell)", "female:C(cell)") and df["cell"].nunique() < 2:
            continue
        if t == "C(profile_id)" and df["profile_id"].nunique() < 2:
            continue
        keep.append(t)
    return f"{y} ~ " + " + ".join(keep)


def standardize(df: pd.DataFrame, cols: list[str], by: list[str] = ("domain", "mode")) -> pd.DataFrame:
    """z-score outcomes within domain x mode (computed on non-baseline responses)."""
    out = df.copy()
    for c in cols:
        g = out.groupby(list(by))[c]
        out[c + "_z"] = (out[c] - g.transform("mean")) / g.transform("std")
    return out


def fit_core(df: pd.DataFrame, y: str, rhs: str = CORE):
    import statsmodels.formula.api as smf

    d = df[(df["gender_cond"] != "baseline") & df[y].notna()].copy()
    d["female"] = d["female"].astype(float)
    d["explicit"] = d["explicit"].astype(float)
    if len(d) < 20 or d["female"].nunique() < 2:
        return None
    groups = d["profile_id"].astype("category").cat.codes
    model = smf.ols(_formula(d, y, rhs), data=d)
    try:
        return model.fit(cov_type="cluster", cov_kwds={"groups": groups})
    except Exception:  # noqa: BLE001 - e.g. singular cluster matrix with 1 cluster
        return model.fit(cov_type="HC1")


def coef_row(res, name: str) -> dict:
    if res is None or name not in res.params.index:
        return {"coef": np.nan, "se": np.nan, "p": np.nan, "ci_lo": np.nan, "ci_hi": np.nan}
    ci = res.conf_int().loc[name]
    return {"coef": res.params[name], "se": res.bse[name], "p": res.pvalues[name], "ci_lo": ci[0], "ci_hi": ci[1]}


def contrast(res, weights: dict[str, float]) -> dict:
    if res is None or any(k not in res.params.index for k in weights):
        return {"coef": np.nan, "se": np.nan, "p": np.nan}
    L = np.array([[weights.get(n, 0.0) for n in res.params.index]])
    t = res.t_test(L)
    return {"coef": float(np.squeeze(t.effect)), "se": float(np.squeeze(t.sd)), "p": float(np.squeeze(t.pvalue))}


def core_table(df: pd.DataFrame, outcomes: list[str], split: list[str] = ("domain", "mode"), per_model: bool = True) -> pd.DataFrame:
    """β1 (female, implicit), β3 (female×explicit), H6 contrasts per outcome × split (× model)."""
    rows = []
    groups = [("pooled", df)]
    if per_model and df["model"].nunique() > 1:
        groups += [(m, g) for m, g in df.groupby("model")]
    for mlabel, dm in groups:
        for keys, d in dm.groupby(list(split)):
            keys = keys if isinstance(keys, tuple) else (keys,)
            for y in outcomes:
                res = fit_core(d, y)
                base = {"model": mlabel, **dict(zip(split, keys)), "outcome": y,
                        "n": int(res.nobs) if res is not None else 0}
                b1 = coef_row(res, "female")
                b3 = coef_row(res, "female:explicit")
                ab = coef_row(res, "female:C(cell)[T.B]")
                bc = contrast(res, {"female:C(cell)[T.C]": 1.0, "female:C(cell)[T.B]": -1.0})
                rows.append({**base,
                             "b1_female": b1["coef"], "b1_se": b1["se"], "b1_p": b1["p"], "b1_lo": b1["ci_lo"], "b1_hi": b1["ci_hi"],
                             "b3_female_x_explicit": b3["coef"], "b3_se": b3["se"], "b3_p": b3["p"],
                             "h6_B_minus_A": ab["coef"], "h6_BA_p": ab["p"], "h6_C_minus_B": bc["coef"], "h6_CB_p": bc["p"]})
    return pd.DataFrame(rows)


def tost(coef: float, se: float, bound: float = 0.1) -> float:
    """Two one-sided tests p-value for |coef| < bound (normal approximation)."""
    if not np.isfinite(coef) or not np.isfinite(se) or se == 0:
        return np.nan
    p_lo = 1 - stats.norm.cdf((coef + bound) / se)
    p_hi = stats.norm.cdf((coef - bound) / se)
    return float(max(p_lo, p_hi))


def h2_table(df: pd.DataFrame, n_boot: int = 500, seed: int = 0, bound: float = 0.1) -> pd.DataFrame:
    """H2: standardized female gap on vertical outcomes (fixed mode) vs horizontal (open mode).

    Directional 'disadvantage' scale: horizontal d_h = β(female_share or people_things),
    vertical d_v = -β(earnings or prestige). H2 supported if d_h - d_v > 0; TOST tests
    |β_v| < 0.1 SD. Difference CIs by cluster (profile) bootstrap.
    """
    z = standardize(df[df["gender_cond"] != "baseline"], ["female_share_us", "people_things", "median_earnings", "prestige_isei"])
    rng = np.random.default_rng(seed)
    rows = []
    groups = [("pooled", z)] + ([(m, g) for m, g in z.groupby("model")] if z["model"].nunique() > 1 else [])
    for mlabel, dz in groups:
        for domain, d in dz.groupby("domain"):
            fixed, open_ = d[d["mode"] == "fixed"], d[d["mode"] == "open"]
            for v in ("median_earnings", "prestige_isei"):
                for h in ("female_share_us", "people_things"):
                    rv, rh = fit_core(fixed, v + "_z"), fit_core(open_, h + "_z")
                    bv, bh = coef_row(rv, "female"), coef_row(rh, "female")
                    diff = bh["coef"] - (-bv["coef"])
                    boots = []
                    profiles = d["profile_id"].unique()
                    for _ in range(n_boot if np.isfinite(diff) else 0):
                        samp = rng.choice(profiles, size=len(profiles), replace=True)
                        bf = pd.concat([fixed[fixed["profile_id"] == p] for p in samp])
                        bo = pd.concat([open_[open_["profile_id"] == p] for p in samp])
                        a, b = fit_core(bf, v + "_z"), fit_core(bo, h + "_z")
                        if a is not None and b is not None and "female" in a.params and "female" in b.params:
                            boots.append(b.params["female"] + a.params["female"])
                    lo, hi = (np.percentile(boots, [2.5, 97.5]) if len(boots) > 20 else (np.nan, np.nan))
                    rows.append({"model": mlabel, "domain": domain, "vertical": v, "horizontal": h,
                                 "beta_vertical_z": bv["coef"], "beta_vertical_se": bv["se"],
                                 "beta_horizontal_z": bh["coef"], "beta_horizontal_se": bh["se"],
                                 "d_h_minus_d_v": diff, "diff_lo": lo, "diff_hi": hi,
                                 "tost_p_vertical": tost(bv["coef"], bv["se"], bound), "tost_bound_sd": bound})
    return pd.DataFrame(rows)


def heterogeneity(df: pd.DataFrame, y: str, by: str = "orientation"):
    """H4: female × orientation (e.g. science-track girls)."""
    rhs = f"female * C({by}) + explicit + C(profile_id) + C(cell) + C(model)"
    return fit_core(df, y, rhs)
