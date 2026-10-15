"""Markdown tables + PNG figures per run (spec §4.6), including the pilot review report (A3)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..coding.coder import coding_summary, source_logs, taxonomy_codes
from ..common.config import OUTPUTS_DIR
from ..common.io import utcnow
from .divergence import h7_summary, h7_table
from .mirror import mirror_table
from .models import core_table, h2_table, standardize
from .outcomes import OUTCOMES_DISCURSIVE, OUTCOMES_HORIZONTAL, OUTCOMES_VERTICAL, outcomes_for_runs

REPORTS_DIR = OUTPUTS_DIR / "reports"


def _md(df: pd.DataFrame, floatfmt: str = ".3f", max_rows: int = 200) -> str:
    if df is None or df.empty:
        return "_(no rows)_\n"
    d = df.head(max_rows).copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else format(v, floatfmt))
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in d.astype(str).values.tolist()]
    return "\n".join(lines) + "\n"


def run_health(logs: pd.DataFrame) -> pd.DataFrame:
    g = logs.groupby(["model_key", "domain", "mode", "cell"])
    return pd.DataFrame({
        "calls": g.size(),
        "ok": g["status"].apply(lambda s: (s == "ok").sum()),
        "refusal_rate": g["status"].apply(lambda s: (s == "refusal").mean()),
        "error_rate": g["status"].apply(lambda s: (s == "error").mean()),
        "parse_ok_rate": g["parse_ok"].apply(lambda s: s.fillna(False).astype(bool).mean()) if "parse_ok" in logs else np.nan,
        "truncated_rate": g["finish_reason"].apply(lambda s: (s == "length").mean()) if "finish_reason" in logs else np.nan,
        "cost_usd": g["cost_usd"].sum(),
    }).reset_index()


def refusal_by_condition(logs: pd.DataFrame) -> pd.DataFrame:
    return (logs.assign(refusal=logs["status"].eq("refusal"))
            .pivot_table(index=["model_key", "cell"], columns="gender_cond", values="refusal", aggfunc="mean").reset_index())


def gap_means(out: pd.DataFrame, outcomes: list[str]) -> pd.DataFrame:
    rows = []
    for k, g in out.groupby(["model", "domain", "mode", "cell"]):
        r = dict(zip(["model", "domain", "mode", "cell"], k))
        for y in outcomes:
            f = g.loc[g["gender_cond"].str.startswith("f_"), y].mean()
            m = g.loc[g["gender_cond"].str.startswith("m_"), y].mean()
            b = g.loc[g["gender_cond"].eq("baseline"), y].mean()
            r[f"{y}:F-M"] = f - m
            r[f"{y}:base"] = b
        rows.append(r)
    return pd.DataFrame(rows)


def coef_plot(tab: pd.DataFrame, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = tab[(tab["model"] == "pooled")].dropna(subset=["b1_female"])
    if t.empty:
        return
    t = t.assign(label=t["outcome"] + " | " + t["domain"] + "/" + t["mode"])
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(t) + 1.2))
    y = np.arange(len(t))
    ax.errorbar(t["b1_female"], y, xerr=[t["b1_female"] - t["b1_lo"], t["b1_hi"] - t["b1_female"]], fmt="o",
                color="#3b6ea5", ecolor="#9db7d5", capsize=2)
    ax.axvline(0, color="#888", lw=0.8)
    ax.set_yticks(y, t["label"], fontsize=8)
    ax.set_xlabel("β1 female (implicit cue), standardized outcome (SD)")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def gap_by_cell_plot(out: pd.DataFrame, y: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = out[out["gender_cond"] != "baseline"].dropna(subset=[y])
    if d.empty:
        return
    gm = d.groupby(["domain", "mode", "cell", "female"])[y].mean().unstack("female")
    if 0 not in gm.columns or 1 not in gm.columns:
        return
    gap = (gm[1] - gm[0]).unstack("cell")
    ax = gap.plot.bar(figsize=(7, 3.5), color=["#3b6ea5", "#e1a33b", "#5a9e6f"][: gap.shape[1]])
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_ylabel(f"female − male: {y}")
    ax.figure.tight_layout()
    ax.figure.savefig(path, dpi=150)
    import matplotlib.pyplot as plt  # noqa: F811

    plt.close(ax.figure)


def build_report(runs: list[str], name: str | None = None, pilot: bool = False, n_boot: int = 300) -> Path:
    name = name or ("__".join(runs) if len(runs) < 3 else f"{runs[0]}__plus{len(runs) - 1}")
    rdir = REPORTS_DIR / name
    rdir.mkdir(parents=True, exist_ok=True)
    logs = pd.concat([source_logs(r)[1] for r in runs], ignore_index=True)
    out = outcomes_for_runs(runs)
    out.to_parquet(rdir / "outcomes.parquet", index=False)
    z_cols = list(OUTCOMES_HORIZONTAL + OUTCOMES_VERTICAL + OUTCOMES_DISCURSIVE)
    oz = standardize(out[out["gender_cond"] != "baseline"], z_cols) if not out.empty else out
    md = [f"# Report: {name}\n", f"_generated {utcnow()}; runs: {', '.join(runs)}_\n"]
    if pilot:
        md.append("> **PILOT — STOP for researcher review before the main run (spec §10, A3).** "
                  "Approve the main run by adding `APPROVED: A4` to QUESTIONS.md.\n")

    md.append("## Run health\n")
    md.append(_md(run_health(logs)))
    md.append(f"Total cost: ${logs['cost_usd'].sum():.4f}. Providers served: "
              f"{logs['provider_served'].value_counts(dropna=False).to_dict() if 'provider_served' in logs else 'n/a'}\n")
    md.append("## Refusal rate by condition\n")
    md.append(_md(refusal_by_condition(logs)))

    md.append("## Coding success\n")
    rows = []
    for r in runs:
        rid, lg = source_logs(r)
        cs = coding_summary(rid, int((lg["status"] == "ok").sum()))
        rows.append({"run": rid, **{f"{p}_rate": v["rate"] for p, v in cs.items()},
                     **{f"{p}_coded": v["coded_ok"] for p, v in cs.items()}})
    md.append(_md(pd.DataFrame(rows)))
    if "n_other" in out:
        md.append(f"Share of coded recommendations mapped to `other`: "
                  f"{out['n_other'].sum() / max(out['n_recs'].sum(), 1):.1%}\n")

    md.append("## Raw female − male gaps (and baseline means)\n")
    md.append(_md(gap_means(out, ["female_share_us", "people_things", "median_earnings", "prestige_isei",
                                   "warmth_minus_competence", "warning_unsolicited"])))

    md.append("## Core regression (standardized outcomes; β1 = female under name-only cue; β3 = female × explicit)\n")
    md.append("`Y ~ female * explicit + female:C(cell) + C(profile_id) + C(cell) + C(model)`, SEs clustered by profile.\n")
    tab = core_table(oz, [c + "_z" for c in z_cols]) if not oz.empty else pd.DataFrame()
    tab.to_csv(rdir / "core_regressions.csv", index=False)
    md.append(_md(tab[["model", "domain", "mode", "outcome", "n", "b1_female", "b1_se", "b1_p", "b3_female_x_explicit",
                       "b3_p", "h6_B_minus_A", "h6_BA_p", "h6_C_minus_B", "h6_CB_p"]] if not tab.empty else tab))
    coef_plot(tab, rdir / "fig_core_b1.png", "Female gap (β1) by outcome")
    for y in ("female_share_us", "prestige_isei", "median_earnings"):
        gap_by_cell_plot(out, y, rdir / f"fig_gap_by_cell_{y}.png")

    md.append("## H2: difference without hierarchy\n")
    h2 = h2_table(out, n_boot=n_boot) if not out.empty else pd.DataFrame()
    h2.to_csv(rdir / "h2.csv", index=False)
    md.append("d_h = β(horizontal, open mode); d_v = −β(vertical, fixed mode); H2 supported if d_h − d_v > 0. "
              "TOST p < .05 ⇒ vertical gap within ±0.1 SD.\n\n")
    md.append(_md(h2))

    md.append("## H7: male default (Jensen–Shannon divergence)\n")
    codes = pd.concat([taxonomy_codes(source_logs(r)[0]) for r in runs], ignore_index=True)
    if not codes.empty and not out.empty:
        h7 = h7_table(out, codes, n_boot=min(n_boot, 200))
        h7.to_csv(rdir / "h7_cells.csv", index=False)
        md.append(_md(h7_summary(h7)) if not h7.empty else "_(no complete cells)_\n")
    else:
        md.append("_(no coded recommendations yet)_\n")

    md.append("## Mirror vs amplify (open mode, US benchmark)\n")
    try:
        md.append(_md(mirror_table(out)))
    except Exception as e:  # noqa: BLE001
        md.append(f"_mirror table unavailable: {e}_\n")

    md.append("\n## Figures\n")
    for p in sorted(rdir.glob("*.png")):
        md.append(f"![{p.stem}]({p.name})\n")
    (rdir / "report.md").write_text("\n".join(md), encoding="utf-8")
    return rdir / "report.md"
