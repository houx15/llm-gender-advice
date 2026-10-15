"""Expand an experiment config into a call plan and estimate its cost (spec §4.1, §8)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..common.config import PLANS_DIR, ExperimentCfg, load_api_models, load_budget
from ..materials.design import attach_models, expand_design
from .budget import estimate_cost, price_for


def build_plan(exp: ExperimentCfg, strict: bool = True) -> pd.DataFrame:
    design = expand_design(exp, strict=strict)
    return attach_models(design, exp.models)


def write_plan(df: pd.DataFrame, name: str) -> Path:
    PLANS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLANS_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False)
    return path


def load_plan(name_or_path: str) -> pd.DataFrame:
    p = Path(name_or_path)
    if not p.exists():
        p = PLANS_DIR / f"{name_or_path}.parquet"
    if not p.exists():
        raise SystemExit(f"plan not found: {name_or_path} (run `advice-audit plan --experiment ...`)")
    return pd.read_parquet(p)


def cost_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-model call counts and config-price estimates (peak if configured)."""
    models = load_api_models()
    b = load_budget()
    rows = []
    for mk, g in df.groupby("model_key"):
        m = models.get(mk)
        peak = b.assume_peak_pricing and m.peak_multiplier > 1
        pin, pout = price_for(m, peak)
        rows.append({
            "model_key": mk,
            "calls": len(g),
            "price_in": pin,
            "price_out": pout,
            "estimate_usd": round(estimate_cost(len(g), models.est_input_tokens, models.est_output_tokens, pin, pout,
                                                b.estimate_safety_factor), 3),
            "estimate_no_safety_usd": round(estimate_cost(len(g), models.est_input_tokens, models.est_output_tokens,
                                                          pin, pout), 3),
        })
    t = pd.DataFrame(rows)
    if not t.empty:
        t.loc[len(t)] = {"model_key": "TOTAL", "calls": t["calls"].sum(), "price_in": None, "price_out": None,
                         "estimate_usd": round(t["estimate_usd"].sum(), 3),
                         "estimate_no_safety_usd": round(t["estimate_no_safety_usd"].sum(), 3)}
    return t


def balance_report(df: pd.DataFrame) -> dict:
    """Balance of names and paraphrases across cells and conditions (spec §9)."""
    one = df[df["model_key"] == df["model_key"].iloc[0]]
    named = one[one["gender_cond"] != "baseline"]
    return {
        "calls_per_domain_mode": one.groupby(["domain", "mode"]).size().to_dict(),
        "paraphrase_by_cell": one.groupby(["cell", "paraphrase_id"]).size().unstack().to_dict(),
        "paraphrase_by_condition": one.groupby(["gender_cond", "paraphrase_id"]).size().unstack().to_dict(),
        "name_idx_by_cell_condition": named.groupby(["cell", "gender_cond", "name_idx"]).size().unstack().to_dict(),
    }
