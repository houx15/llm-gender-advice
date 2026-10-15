"""Activation patching (spec §6.3, Step 3): copy the male prompt's residual at layer × position
into the matched female prompt and measure how much of the readout gap closes.

Per pair rows: clean_f, clean_m, and patched (layer, position) readouts. Aggregation:
gap_closed = Σ(E_f − E_patched) / Σ(E_f − E_m) over pairs (ratio of sums; stable when a
single pair's gap is ~0).

Entry point: python -m advice_audit.local.patching --model <key> --shard <k>
"""
from __future__ import annotations

from .offline_guard import activate  # noqa: I001

import argparse
import json

import numpy as np
import pandas as pd
import torch

from ..common.io import append_parquet, write_json
from ..common.logs import get_logger
from .hooks import capture, n_layers, patch
from .prompts import interp_cfg
from .shards import run_shard, shard_dirs, stage_dir

log = get_logger("advice_audit.local.patching")
ATTRS = ("E_female_share_us", "E_prestige_isei", "E_median_earnings", "E_people_things")


def _side(row: pd.Series, s: str) -> dict:
    keep = ["domain", "mode", "profile_id", "cell", "name_idx", "paraphrase_id", "language", "orientation"]
    d = {k: row[k] for k in keep if k in row}
    for k in ("prompt", "gender_cond", "name_span", "cue_span", "prompt_id", "language", "orientation"):
        if f"{k}_{s}" in row:
            d[k] = row[f"{k}_{s}"]
    return d


@torch.no_grad()
def patch_pair(rt, row: pd.Series, layers: list[int], positions: list[str]) -> list[dict]:
    f, m = _side(row, "f"), _side(row, "m")
    pf, pm = rt.prepare(f), rt.prepare(m)
    base = {"pair_id": row["pair_id"], "cue": row["cue"], "domain": row["domain"], "mode": row["mode"],
            "cell": row["cell"], "profile_id": row["profile_id"]}
    rows = []
    sf, _ = rt.readout(f, prepared=pf)
    sm, _ = rt.readout(m, prepared=pm)
    rows.append({**base, "layer": -1, "position": "clean_f", **sf})
    rows.append({**base, "layer": -1, "position": "clean_m", **sm})
    dev = next(rt.model.parameters()).device
    ids_m = torch.tensor([pm[0]], device=dev)
    pos_m = [pm[1].get(p) for p in positions]
    valid = [i for i, p in enumerate(positions) if pos_m[i] is not None and pf[1].get(p) is not None]
    if not valid:
        return rows
    with capture(rt.model, layers, store_all=True) as store:
        rt.model(input_ids=ids_m)
    for l in layers:
        for i in valid:
            p = positions[i]
            val = store[l][0, pm[1][p]][None]
            tgt = torch.tensor([pf[1][p]], device=dev)
            ctx = lambda l=l, tgt=tgt, val=val: patch(rt.model, l, tgt, val)  # noqa: E731
            s, _ = rt.readout(f, ctx=ctx, prefix_only=True, prepared=pf)
            rows.append({**base, "layer": l, "position": p, **s})
    return rows


def run(model_key: str, shard: int) -> None:
    from .runtime import Runtime

    cfg = interp_cfg()["patching"]
    state = {}

    def work(df: pd.DataFrame, out):
        rt = state.setdefault("rt", Runtime(model_key))
        L = n_layers(rt.model)
        layers = list(range(0, L, int(cfg.get("layer_stride", 1))))
        for _, row in df.iterrows():
            append_parquet(out / "results", patch_pair(rt, row, layers, cfg["positions"]))
        return {"pairs": len(df), "layers": len(layers)}

    run_shard(model_key, "patching", shard, work)


def aggregate(model_key: str, top_k: int = 3) -> pd.DataFrame:
    from ..common.io import read_parquet_dir

    df = pd.concat([read_parquet_dir(d / "results") for d in shard_dirs(model_key, "patching")], ignore_index=True)
    clean = df[df["layer"] == -1].pivot_table(index="pair_id", columns="position", values=list(ATTRS))
    pat = df[df["layer"] >= 0]
    rows = []
    for (l, p), g in pat.groupby(["layer", "position"]):
        r = {"layer": l, "position": p, "n_pairs": g["pair_id"].nunique()}
        for a in ATTRS:
            ef = clean[(a, "clean_f")].reindex(g["pair_id"]).to_numpy()
            em = clean[(a, "clean_m")].reindex(g["pair_id"]).to_numpy()
            ep = g[a].to_numpy()
            denom = np.nansum(ef - em)
            r[f"gap_closed_{a}"] = float(np.nansum(ef - ep) / denom) if denom else np.nan
            r[f"gap_{a}"] = float(np.nanmean(ef - em))
        rows.append(r)
    agg = pd.DataFrame(rows).sort_values(["position", "layer"])
    out = stage_dir(model_key, "patching")
    agg.to_csv(out / "gap_closed.csv", index=False)
    key = "gap_closed_E_female_share_us"
    best = agg.groupby("layer")[key].max().sort_values(ascending=False)
    selected = [int(x) for x in best.head(top_k).index.tolist()]
    write_json(out / "selected_layers.json", {"selected_layers": selected, "criterion": f"max over positions of {key}"})
    return agg


def selected_layers(model_key: str, n_total: int) -> list[int]:
    p = stage_dir(model_key, "patching") / "selected_layers.json"
    if p.exists():
        return json.loads(p.read_text())["selected_layers"]
    return [n_total // 2]


def main(argv=None) -> None:
    activate()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--shard", type=int)
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args(argv)
    if a.aggregate:
        aggregate(a.model)
    else:
        run(a.model, a.shard)


if __name__ == "__main__":
    main()
