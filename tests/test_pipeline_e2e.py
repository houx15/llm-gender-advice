"""End-to-end Part B on the tiny model through the real shard entry points (CPU):
plan -> behavior -> readout -> extract -> concepts -> probes/directions -> patching ->
aggregate -> steering -> erasure -> aggregate -> local report."""
import copy

import pandas as pd
import pytest


@pytest.fixture
def small_cfg(monkeypatch, tiny):
    import advice_audit.local.erasure as ER
    import advice_audit.local.patching as PA
    import advice_audit.local.prompts as PR
    import advice_audit.local.steering as ST

    base = PR.interp_cfg()
    cfg = copy.deepcopy(base)
    cfg["extract"].update(samples=3, shard_size=40)
    cfg["concepts"]["shard_size"] = 400
    cfg["probes"].update(n_permutations=1, max_iter=20, cv_folds=3)
    cfg["directions"].update(n_boot=3, cv_folds=3)
    cfg["patching"].update(n_pairs=4, shard_size=2)
    cfg["steering"].update(n_prompts=4, n_random=2, alphas=[-1, 0, 1], n_generate=1, generate_alphas=[0, 1],
                           max_new_tokens=4, shard_size=2)
    cfg["erasure"].update(n_generate=1, max_new_tokens=4, shard_size=50)
    cfg["side_effects"]["ppl_max_docs"] = 1
    for mod in (PR, PA, ST, ER):
        monkeypatch.setattr(mod, "interp_cfg", lambda: cfg)
    monkeypatch.setattr(PR, "require_approval", lambda gate: None)
    from advice_audit.common.config import LocalModelsCfg

    import advice_audit.local.modeling as M

    fake = M.load_local_models()
    monkeypatch.setattr(PR, "load_local_models", lambda: fake)
    import advice_audit.local.generate as G

    monkeypatch.setattr(G, "load_local_models", lambda: fake)
    return cfg


def test_full_local_pipeline(small_cfg, tiny, _isolated_outputs):
    from advice_audit.analysis.local_report import build_local_report
    from advice_audit.local import erasure, extract, generate, patching, probes, readout, steering
    from advice_audit.local.prompts import plan_stage
    from advice_audit.local.shards import n_shards, shard_dirs

    def run_all(stage, fn, **kw):
        plan_stage("tiny", stage, cells=["A"], limit=kw.pop("limit", None))
        for k in range(n_shards("tiny", stage)):
            fn(k)
        assert len(shard_dirs("tiny", stage)) == n_shards("tiny", stage)

    run_all("behavior", lambda k: generate.run("tiny", k, backend="hf"), limit=6)
    run_all("readout", lambda k: readout.run_stage("tiny", k), limit=6)
    run_all("extract", lambda k: extract.run("tiny", k, "extract"))
    run_all("concepts", lambda k: extract.run("tiny", k, "concepts"))
    # probes + directions without the offline guard (tests run online)
    acts, valid, idx, meta = extract.load_activations("tiny", "extract")
    res = probes.run_probes(acts, valid, idx, meta, small_cfg["probes"])
    assert {"cv_acc", "perm_mean", "transfer_en_to_zh"} <= set(res.columns) or "cv_acc" in res
    from advice_audit.local.directions import run_directions

    dirs = run_directions("tiny", acts, valid, idx, meta, small_cfg)
    assert {"g_implicit_last", "s", "p", "resid_norm_last"} <= set(dirs)
    run_all("patching", lambda k: patching.run("tiny", k))
    agg = patching.aggregate("tiny")
    assert {"gap_closed_E_female_share_us", "layer", "position"} <= set(agg.columns)
    run_all("steering", lambda k: steering.run("tiny", k))
    run_all("erasure", lambda k: erasure.run("tiny", k))
    out = erasure.aggregate("tiny")
    assert set(out["gaps"]["condition"]) >= {"none", "leace_all"}
    # resume: re-running a DONE shard is a no-op
    assert steering.run("tiny", 0) is None
    rep = build_local_report("tiny")
    text = rep.read_text()
    assert "Activation patching" in text and "Erasure" in text
    assert "_not run_" not in text.split("## Steering")[1].split("##")[0]  # steering results found
    # behavior logs are in the Part A schema
    from advice_audit.coding.coder import source_logs

    rid, logs = source_logs(str(_isolated_outputs / "local" / "tiny" / "behavior"))
    assert rid == "local__tiny__behavior" and {"call_id", "response", "status", "gender_cond"} <= set(logs.columns)


def test_entry_point_under_offline_guard(small_cfg, tiny, monkeypatch, tmp_path):
    """The real `main()` (guard active, network blocked) runs a shard from the local cache."""
    from advice_audit.local import extract, offline_guard
    from advice_audit.local.prompts import plan_stage
    from advice_audit.local.shards import is_done

    for k, v in {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
                 "VLLM_NO_USAGE_STATS": "1", "HF_HOME": str(tmp_path / "hf")}.items():
        monkeypatch.setenv(k, v)
    plan_stage("tiny", "extract", cells=["A"], limit=4)
    try:
        extract.main(["--model", "tiny", "--shard", "0"])
        assert offline_guard.is_active()
    finally:
        offline_guard.deactivate()
    assert is_done("tiny", "extract", 0)
