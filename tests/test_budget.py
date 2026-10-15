"""Budget guard refuses over-cap runs (mock ledger)."""
import pandas as pd

from advice_audit.api.budget import BudgetGuard, estimate_cost, price_for
from advice_audit.common.config import BudgetCfg, load_api_models


class FakeLedger:
    def __init__(self, rows):
        self.df = pd.DataFrame(rows, columns=["run_id", "call_id", "phase", "model_key", "cost_usd", "request_ts"])

    def load(self):
        return self.df

    def spent_total(self):
        return float(self.df["cost_usd"].sum())

    def spent_by(self, col):
        return self.df.groupby(col)["cost_usd"].sum().to_dict() if len(self.df) else {}


CFG = BudgetCfg(total_cap_usd=60, reserve_usd=6, phase_caps={"pilot": 3, "main": 45, "coding": 12},
                kill_switch={"overrun_fraction": 0.2})


def test_refuses_over_phase_cap():
    g = BudgetGuard(CFG, FakeLedger([["r", "c", "pilot", "m", 2.5, ""]]))
    assert g.check(0.4, "pilot")[0]
    ok, msg = g.check(0.6, "pilot")
    assert not ok and "phase:pilot" in msg


def test_refuses_over_total_minus_reserve():
    g = BudgetGuard(CFG, FakeLedger([["r", "c", "main", "m", 44.0, ""], ["r", "d", "coding", "m", 9.0, ""]]))
    ok, msg = g.check(1.5, "coding")  # coding room 3.0 but total room 60-6-53 = 1.0
    assert not ok and "total" in msg
    assert g.check(1.5, "coding", allow_reserve=True)[0]


def test_kill_switch():
    g = BudgetGuard(CFG, FakeLedger([]))
    assert g.kill_switch(1.0, 1.25, 10.0)[0]       # > estimate * 1.2
    assert not g.kill_switch(1.0, 1.1, 10.0)[0]
    assert g.kill_switch(5.0, 2.0, 2.0)[0]         # room exhausted


def test_estimates_match_spec_order_of_magnitude():
    m = load_api_models().get("gpt54_mini")
    pin, pout = price_for(m, peak=False)
    assert abs(estimate_cost(7200, 500, 600, pin, pout) - 22.14) < 0.01
    m = load_api_models().get("deepseek_v41_flash")
    pin, pout = price_for(m, peak=True, live=(0.15, 0.60))
    assert pin >= 0.15 and pout >= 0.60  # max(config, live)


def test_pinned_endpoint_ignores_fast_variants():
    from advice_audit.api.client import OpenRouterClient

    eps = [{"provider_name": "OpenAI", "tag": "openai/flex", "price": (0.375, 2.25)},
           {"provider_name": "OpenAI", "tag": "openai/fast", "price": (1.5, 9.0)},
           {"provider_name": "OpenAI", "tag": "openai", "price": (0.75, 4.5)},
           {"provider_name": "DeepInfra", "tag": "deepinfra/fp8", "price": (0.14, 0.42)}]
    assert OpenRouterClient.pinned_endpoint(eps, "OpenAI")["price"] == (0.75, 4.5)
    assert OpenRouterClient.pinned_endpoint(eps, "DeepInfra")["price"] == (0.14, 0.42)
    assert OpenRouterClient.pinned_endpoint(eps, "deepinfra/fp8")["tag"] == "deepinfra/fp8"
    assert OpenRouterClient.pinned_endpoint(eps, "Google") is None
