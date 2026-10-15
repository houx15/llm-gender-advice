"""Part B reports (spec §10 B2–B4): readout validation, probes by layer, direction geometry,
patching heatmap, steering dose–response with random-direction placebo bands, erasure
gaps, leakage, and side effects. Reads outputs/local/<model>/...; runs on CPU."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..common.config import LOCAL_DIR, OUTPUTS_DIR
from ..common.io import read_parquet_dir, utcnow
from .report import _md


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _concat(model_key: str, stage: str, sub: str) -> pd.DataFrame:
    d = LOCAL_DIR / model_key / stage
    frames = [read_parquet_dir(p / sub) for p in sorted(d.glob("shard_*")) if (p / "DONE").exists()]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def readout_validation(model_key: str, behavior_run: str | None = None) -> pd.DataFrame:
    """Agreement of the readout's top category with the first coded free-text recommendation."""
    from ..coding.coder import taxonomy_codes

    ro = _concat(model_key, "readout", "results")
    if ro.empty:
        return pd.DataFrame()
    beh = _concat(model_key, "behavior", "logs")
    codes = taxonomy_codes(behavior_run or f"local__{model_key}__behavior")
    if beh.empty or codes.empty:
        return pd.DataFrame()
    top1 = codes[codes["rank"] == 1][["source_call_id", "category_id"]]
    m = beh.merge(top1, left_on="call_id", right_on="source_call_id").merge(
        ro[["prompt_id", "top_category", "top_category_lennorm"]], on="prompt_id")
    m = m[m["category_id"] != "other"]
    return (m.assign(agree=m["category_id"] == m["top_category"], agree_lennorm=m["category_id"] == m["top_category_lennorm"])
            .groupby(["domain", "mode", "language"])[["agree", "agree_lennorm"]].mean().reset_index()
            .assign(n=m.groupby(["domain", "mode", "language"]).size().values))


def plot_probes(df: pd.DataFrame, path: Path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for (pos, subset), g in df[df["subset"] == "all"].groupby(["position", "subset"]):
        ax.plot(g["layer"], g["cv_acc"], label=f"{pos}")
        ax.fill_between(g["layer"], g["perm_mean"], g["perm_max"], alpha=0.15)
    for col, ls in (("transfer_en_to_zh", "--"), ("transfer_zh_to_en", ":")):
        if col in df:
            g = df[(df["subset"] == "all") & (df["position"] == "last")]
            ax.plot(g["layer"], g[col], ls=ls, color="#555", label=col)
    ax.axhline(0.5, color="#aaa", lw=0.8)
    ax.set_xlabel("layer")
    ax.set_ylabel("gender probe accuracy")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_geometry(geo: pd.DataFrame, path: Path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for c, color in (("cos_g_s", "#c0504d"), ("cos_g_p", "#3b6ea5"), ("cos_s_p", "#888888"), ("cos_gEN_gZH", "#5a9e6f")):
        if c in geo:
            ax.plot(geo["layer"], geo[c], label=c, color=color)
            if f"{c}_lo" in geo:
                ax.fill_between(geo["layer"], geo[f"{c}_lo"], geo[f"{c}_hi"], color=color, alpha=0.15)
    ax.axhline(0, color="#aaa", lw=0.8)
    ax.set_xlabel("layer")
    ax.set_ylabel("cosine")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_patching(agg: pd.DataFrame, path: Path, attr: str = "gap_closed_E_female_share_us") -> None:
    plt = _plt()
    piv = agg.pivot_table(index="position", columns="layer", values=attr)
    fig, ax = plt.subplots(figsize=(8, 2.2))
    im = ax.imshow(piv.values, aspect="auto", cmap="viridis", vmin=0, vmax=max(1.0, np.nanmax(piv.values)))
    ax.set_yticks(range(len(piv.index)), piv.index)
    ax.set_xlabel("layer")
    ax.set_title(attr, fontsize=9)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_dose_response(st: pd.DataFrame, path: Path, attr: str) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    layer = st["layer"].min()
    s = st[st["layer"] == layer]
    rnd = s[s["direction"].str.startswith("random")].groupby(["direction", "alpha"])[attr].mean().unstack("direction")
    if not rnd.empty:
        ax.fill_between(rnd.index, rnd.min(axis=1), rnd.max(axis=1), color="#bbb", alpha=0.5, label="random (placebo range)")
    for d, color in (("g", "#c0504d"), ("s", "#3b6ea5"), ("p", "#5a9e6f")):
        g = s[s["direction"] == d].groupby("alpha")[attr].mean()
        if not g.empty:
            ax.plot(g.index, g.values, marker="o", label=d, color=color)
    ax.set_xlabel(f"α (layer {layer})")
    ax.set_ylabel(attr)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_local_report(model_key: str) -> Path:
    base = LOCAL_DIR / model_key
    rdir = OUTPUTS_DIR / "reports" / f"local__{model_key}"
    rdir.mkdir(parents=True, exist_ok=True)
    md = [f"# Part B report: {model_key}\n", f"_generated {utcnow()}_\n"]

    md.append("## Readout validation (top readout category vs first coded recommendation)\n")
    try:
        md.append(_md(readout_validation(model_key)))
    except Exception as e:  # noqa: BLE001
        md.append(f"_unavailable: {e}_\n")

    p = base / "probes" / "probes.parquet"
    md.append("## Gender probes by layer\n")
    if p.exists():
        pr = pd.read_parquet(p)
        plot_probes(pr, rdir / "fig_probes.png")
        best = pr.sort_values("cv_acc", ascending=False).groupby(["position", "subset"]).head(1)
        md.append(_md(best))
    else:
        md.append("_not run_\n")

    md.append("## Direction geometry (H8: g aligns with p more than with s)\n")
    g = base / "directions" / "geometry.csv"
    if g.exists():
        geo = pd.read_csv(g)
        plot_geometry(geo, rdir / "fig_geometry.png")
        md.append(_md(geo.describe().T.reset_index().rename(columns={"index": "metric"})))
        mpath = base / "directions" / "metrics.json"
        if mpath.exists():
            md.append(f"Direction checks: `{json.loads(mpath.read_text())}`\n")
        cp = base / "directions" / "concept_projections_on_g.csv"
        if cp.exists():
            md.append("Concept projections on g vs real-world attributes (best layers):\n\n")
            md.append(_md(pd.read_csv(cp).sort_values("corr_female_share_us", ascending=False).head(5)))
    else:
        md.append("_not run_\n")

    md.append("## Activation patching (share of readout gap closed)\n")
    a = base / "patching" / "gap_closed.csv"
    if a.exists():
        agg = pd.read_csv(a)
        plot_patching(agg, rdir / "fig_patching.png")
        md.append(_md(agg.sort_values("gap_closed_E_female_share_us", ascending=False).head(10)))
    else:
        md.append("_not aggregated (python -m advice_audit.local.patching --model K --aggregate)_\n")

    md.append("## Steering dose–response (H9) with random-direction placebo\n")
    st = _concat(model_key, "steering", "readout")
    if not st.empty:
        for attr in ("E_female_share_us", "E_prestige_isei", "E_people_things"):
            plot_dose_response(st, rdir / f"fig_steering_{attr}.png", attr)
        summ = st.groupby(["direction", "layer", "alpha"])[["E_female_share_us", "E_prestige_isei", "E_people_things",
                                                           "entropy", "off_track_mass"]].mean().reset_index()
        md.append(_md(summ[~summ["direction"].str.startswith("random")]))
        se = _concat(model_key, "steering", "side_effects")
        if not se.empty:
            md.append("Side effects:\n\n" + _md(se[~se["direction"].str.startswith("random")]))
    else:
        md.append("_not run_\n")

    md.append("## Erasure (H10)\n")
    e = base / "erasure" / "gaps_by_condition.csv"
    if e.exists():
        md.append(_md(pd.read_csv(e)))
        lk = base / "erasure" / "leakage_probes.csv"
        if lk.exists():
            md.append("Gender probe accuracy downstream of erasure (leakage):\n\n" + _md(pd.read_csv(lk)))
        se = _concat(model_key, "erasure", "side_effects")
        if not se.empty:
            md.append("Side effects:\n\n" + _md(se))
    else:
        md.append("_not aggregated (python -m advice_audit.local.erasure --model K --aggregate)_\n")

    md.append("\n## Figures\n")
    for f in sorted(rdir.glob("*.png")):
        md.append(f"![{f.stem}]({f.name})\n")
    (rdir / "report.md").write_text("\n".join(md), encoding="utf-8")
    return rdir / "report.md"
