"""Runner: resumable by call_id + prompt_hash, Parquet logs, budget refusal, coding round trip."""
import asyncio
import json

import pandas as pd
import pytest

import advice_audit.api.runner as R
from advice_audit.common.config import load_experiment
from advice_audit.materials.design import attach_models, expand_design

ANSWER = "1. Computer Science: fits.\n2. Physics: fits.\n3. Mathematics: fits.\n4. Statistics: ok.\n5. Economics: ok."


class FakeClient:
    calls = 0

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *e):
        pass

    async def verify_model(self, cfg):
        return {"ok": True, "problems": [], "live_price": (0.14, 0.42)}

    async def chat(self, cfg, prompt, max_tokens=None, json_mode=False, system_prompt=None):
        FakeClient.calls += 1
        if json_mode and "discursive" not in prompt and "Categories" in prompt:
            content = json.dumps({"recommendations": [
                {"rank": 1, "raw_text": "Computer Science", "category_id": "computer_science", "confidence": 0.9},
                {"rank": 2, "raw_text": "Physics", "category_id": "physics", "confidence": 0.9},
                {"rank": 3, "raw_text": "Basket weaving", "category_id": "not_a_category", "confidence": 0.2}]})
        elif json_mode:
            content = json.dumps({"warning_unsolicited": False, "warning_quote": "", "hedge_count": 2,
                                  "ambition_encouragement": 1, "mentions_gender": False,
                                  "mentions_worklife_or_family": False, "leadership_mention": True})
        else:
            content = ANSWER
        return {"content": content, "finish_reason": "stop", "http_status": 200, "latency_ms": 5,
                "input_tokens": 300, "output_tokens": 200, "reasoning_tokens": 0, "cost_usd": 0.0001,
                "provider_served": cfg.provider.order[0] if cfg.provider.order else None, "model_served": cfg.slug}


@pytest.fixture
def plan(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setattr(R, "OpenRouterClient", FakeClient)
    exp = load_experiment("pilot")
    exp.cells, exp.samples, exp.domains, exp.modes = ["A"], 1, ["majors"], ["open"]
    exp.profiles = {"majors": ["maj_sci_high_r1", "maj_hum_mid_r1"]}
    return attach_models(expand_design(exp, strict=False), exp.models)


def test_run_and_resume(plan, _isolated_outputs):
    FakeClient.calls = 0
    m = asyncio.run(R.execute_plan(plan, "deepseek_v41_flash", "t__ds", "pilot"))
    assert m["results"]["ok"] == len(plan) == FakeClient.calls
    logs = R.load_run_logs("t__ds")
    assert len(logs) == len(plan) and logs["parse_ok"].all() and (logs["n_items"] == 5).all()
    assert (logs["provider_served"] == "DeepInfra").all()
    m2 = asyncio.run(R.execute_plan(plan, "deepseek_v41_flash", "t__ds", "pilot"))
    assert m2["pending"] == 0 and FakeClient.calls == len(plan)  # nothing re-run
    # editing a prompt (e.g. after filling materials) makes that call pending again
    plan2 = plan.copy()
    plan2.loc[0, "prompt_hash"] = "changed"
    m3 = asyncio.run(R.execute_plan(plan2, "deepseek_v41_flash", "t__ds", "pilot", dry_run=True))
    assert m3["pending"] == len(plan)  # dry runs ignore completion
    done = R.completed(_isolated_outputs / "runs")
    assert (plan2.loc[0, "call_id"], "changed") not in done


def test_budget_refusal(plan, monkeypatch):
    from advice_audit.common.config import BudgetCfg

    tight = BudgetCfg(total_cap_usd=60, reserve_usd=6, phase_caps={"pilot": 0.000001})
    monkeypatch.setattr(R, "load_budget", lambda: tight)
    with pytest.raises(SystemExit, match="BUDGET GUARD REFUSED"):
        asyncio.run(R.execute_plan(plan, "deepseek_v41_flash", "t__refuse", "pilot"))


def test_coding_round_trip_and_outcomes(plan, monkeypatch):
    import advice_audit.coding.coder as K
    from advice_audit.analysis.outcomes import response_outcomes

    asyncio.run(R.execute_plan(plan, "deepseek_v41_flash", "t__ds", "pilot"))
    K.run_coding("t__ds")
    codes = K.taxonomy_codes("t__ds")
    assert set(codes["category_id"]) == {"computer_science", "physics", "other"}  # invalid id -> other
    disc = K.discursive_codes("t__ds")
    assert len(disc) == len(plan)
    out = response_outcomes(R.load_run_logs("t__ds"), codes, disc)
    assert out["female_share_us"].notna().all()
    assert (out["leadership_mention"] == 1).all()
    assert (out["n_other"] == 1).all()


def test_report_and_human_export(plan):
    import advice_audit.coding.coder as K
    from advice_audit.analysis.report import build_report
    from advice_audit.coding.human import export_sample, kappa_report

    asyncio.run(R.execute_plan(plan, "deepseek_v41_flash", "t__ds", "pilot"))
    K.run_coding("t__ds")
    rep = build_report(["t__ds"], pilot=True, n_boot=5)
    text = rep.read_text()
    assert "PILOT" in text and "Run health" in text and "Core regression" in text
    d = export_sample("t__ds", frac=0.5)
    c1 = pd.read_csv(d / "t__ds__coder1.csv")
    assert len(c1) >= 2 and "top1_category_id" in c1.columns
    key = pd.read_csv(d / "t__ds__llm_key.csv")
    c1["top1_category_id"] = c1["call_id"].map(dict(zip(key["call_id"], key["top1_category_id"])))
    c1.to_csv(d / "t__ds__coder1.csv", index=False)
    c1.to_csv(d / "t__ds__coder2.csv", index=False)
    k = kappa_report("t__ds")
    row = k[(k["field"] == "top1_category_id") & (k["language"] == "all")].iloc[0]
    assert row["kappa_h1_llm"] == 1.0 or pd.isna(row["kappa_h1_llm"])
