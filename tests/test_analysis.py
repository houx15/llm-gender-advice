"""Analysis: planted female gap recovered by the core model; weights; JSD; H2/TOST."""
import numpy as np
import pandas as pd

from advice_audit.analysis.divergence import jsd
from advice_audit.analysis.models import core_table, fit_core, tost
from advice_audit.analysis.outcomes import weighted_mean


def _synthetic(beta=0.5, n_rep=30, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(12):
        for cell in "ABC":
            for gc in ("f_implicit", "f_explicit", "m_implicit", "m_explicit", "baseline"):
                for s in range(n_rep):
                    f = None if gc == "baseline" else int(gc.startswith("f"))
                    e = None if gc == "baseline" else int(gc.endswith("explicit"))
                    y = 0.3 * p + (beta * f if f else 0) + (0.2 if (f and e) else 0) + (0.3 if (f and cell == "C") else 0) + rng.normal(0, 1)
                    rows.append({"profile_id": f"p{p}", "cell": cell, "gender_cond": gc, "female": f, "explicit": e,
                                 "model": "m1", "domain": "majors", "mode": "open", "y": y})
    df = pd.DataFrame(rows)
    df["female"] = df["female"].astype("Int64")
    df["explicit"] = df["explicit"].astype("Int64")
    return df


def test_core_model_recovers_planted_effects():
    res = fit_core(_synthetic(), "y")
    assert abs(res.params["female"] - 0.5) < 0.15
    assert abs(res.params["female:explicit"] - 0.2) < 0.15
    assert abs(res.params["female:C(cell)[T.C]"] - 0.3) < 0.15
    tab = core_table(_synthetic(), ["y"])
    assert abs(tab["h6_C_minus_B"].iloc[0] - 0.3) < 0.2


def test_weighted_mean():
    v = pd.Series([1.0, 0.0, np.nan])
    r = pd.Series([1, 2, 3])
    assert abs(weighted_mean(v, r) - 5 / 9) < 1e-9


def test_jsd_bounds():
    p = np.array([1.0, 0, 0])
    q = np.array([0, 1.0, 0])
    assert abs(jsd(p, q) - 1.0) < 1e-9
    assert jsd(p, p) == 0


def test_tost():
    assert tost(0.0, 0.02) < 0.05
    assert tost(0.2, 0.02) > 0.5
