"""Plan sizes match spec §4.1; names and paraphrases balanced; gender conditions matched."""
import pandas as pd

from advice_audit.common.config import load_experiment
from advice_audit.materials.design import assign, attach_models, expand_design


def test_main_plan_size_per_model():
    exp = load_experiment("study1_main")
    d = expand_design(exp, strict=False)
    assert len(d) == 7200  # 12 profiles x 5 conditions x 3 cells x 10 samples x 2 domains x 2 modes
    assert d.groupby(["domain", "mode"]).size().eq(1800).all()
    full = attach_models(d, exp.models)
    assert len(full) == 7200 * 5
    assert full["call_id"].is_unique


def test_pilot_plan_size():
    exp = load_experiment("pilot")
    d = attach_models(expand_design(exp, strict=False), exp.models)
    assert len(d) == 1200
    assert set(d["model_key"]) == {"deepseek_v41_flash"}


def test_latin_square_balance():
    pairs = {assign("majors", "open", "p", "A", s) for s in range(9)}
    assert len(pairs) == 9  # every (name, paraphrase) once in 9 samples


def test_balance_across_cells_and_conditions():
    exp = load_experiment("study1_main")
    exp.samples = 9
    d = expand_design(exp, strict=False)
    for col in ("paraphrase_id", "name_idx"):
        counts = d.groupby(["cell", "gender_cond", col]).size().unstack(col)
        assert (counts.nunique(axis=1) == 1).all(), col  # exactly balanced at 9 samples


def test_conditions_matched_on_name_and_paraphrase():
    exp = load_experiment("pilot")
    d = expand_design(exp, strict=False)
    key = ["domain", "mode", "profile_id", "cell", "sample_idx"]
    g = d.groupby(key).agg(n_para=("paraphrase_id", "nunique"), n_idx=("name_idx", "nunique"))
    assert (g["n_para"] == 1).all() and (g["n_idx"] == 1).all()


def test_call_ids_deterministic():
    exp = load_experiment("pilot")
    a = attach_models(expand_design(exp, strict=False), exp.models)
    b = attach_models(expand_design(exp, strict=False), exp.models)
    pd.testing.assert_series_equal(a["call_id"], b["call_id"])
