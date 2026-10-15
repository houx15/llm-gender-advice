"""Steering (spec §6.6): add α·direction at selected layers, over an α grid, for g, s, p and
≥10 random placebo directions of equal norm; readout on every condition, free-text
generations (greedy) for a subset, and side effects.

Vector = α · unit(direction) · α_unit(layer), α_unit = alpha_unit_fraction × mean residual
norm at the layer (last token, from extraction). α = 0 is the unmodified model.

Entry point: python -m advice_audit.local.steering --model <key> --shard <k>
"""
from __future__ import annotations

from .offline_guard import activate  # noqa: I001

import argparse

import pandas as pd
import torch

from ..common.ids import sha256
from ..common.io import append_parquet, utcnow
from ..common.logs import get_logger
from .hooks import add_vector, n_layers
from .prompts import interp_cfg
from .shards import run_shard, stage_dir

log = get_logger("advice_audit.local.steering")


def load_directions(model_key: str) -> dict[str, torch.Tensor]:
    p = stage_dir(model_key, "directions") / "directions.pt"
    if not p.exists():
        raise SystemExit(f"{p} missing — run the probes/directions job first")
    return torch.load(p)


def unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm().clamp_min(1e-8)


def random_direction(d: int, seed: int, layer: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(1_000_003 * seed + layer)
    return unit(torch.randn(d, generator=g))


def build_vectors(dirs: dict, layers: list[int], all_layers: list[int], cfg: dict) -> dict[str, dict[int, torch.Tensor]]:
    """{direction_name: {layer: unit vector}} incl. random placebos."""
    name_map = {"g": "g_implicit_last", "s": "s", "p": "p"}
    out: dict[str, dict[int, torch.Tensor]] = {}
    li = {l: all_layers.index(l) for l in layers}
    d = next(iter(dirs.values())).shape[-1]
    for name in cfg.get("directions", ["g", "s", "p"]):
        key = name_map.get(name, name)
        if key in dirs:
            out[name] = {l: unit(dirs[key][li[l]].float()) for l in layers}
        else:
            log.warning(f"direction {key} not found; skipping")
    for s in range(int(cfg.get("n_random", 10))):
        out[f"random{s}"] = {l: random_direction(d, s, l) for l in layers}
    return out


def alpha_units(dirs: dict, layers: list[int], all_layers: list[int], cfg: dict) -> dict[int, float]:
    frac = float(cfg.get("alpha_unit_fraction", 0.05))
    norms = dirs.get("resid_norm_last")
    return {l: frac * (float(norms[all_layers.index(l)]) if norms is not None else 1.0) for l in layers}


def run(model_key: str, shard: int) -> None:
    from .generate import generate_hf
    from .patching import selected_layers
    from .runtime import Runtime
    from .sidefx import load_ppl_texts, perplexity, refusal_rate

    icfg = interp_cfg()
    cfg = icfg["steering"]
    state = {}

    def work(df: pd.DataFrame, out):
        rt = state.setdefault("rt", Runtime(model_key))
        L = n_layers(rt.model)
        all_layers = list(range(L))
        dirs = load_directions(model_key)
        layers = selected_layers(model_key, L) if cfg.get("layers", "auto") == "auto" else [int(x) for x in cfg["layers"]]
        vecs = build_vectors(dirs, layers, all_layers, cfg)
        units = alpha_units(dirs, layers, all_layers, cfg)
        alphas = [float(a) for a in cfg["alphas"]]
        res = []
        for _, row in df.iterrows():
            prep = rt.prepare(row)
            s0, _ = rt.readout(row, prepared=prep)
            base = {"prompt_id": row["prompt_id"], "domain": row["domain"], "mode": row["mode"], "cell": row["cell"],
                    "profile_id": row["profile_id"], "orientation": row.get("orientation")}
            for name, per_layer in vecs.items():
                for l in layers:
                    for a in alphas:
                        if a == 0:
                            s = s0
                        else:
                            v = per_layer[l] * a * units[l]
                            s, _ = rt.readout(row, ctx=lambda v=v, l=l: add_vector(rt.model, {l: v}), prepared=prep)
                        res.append({**base, "direction": name, "layer": l, "alpha": a, **s})
        append_parquet(out / "readout", res)

        # generations: g/s/p only, at the best layer, for the first n_generate prompts of the plan
        gen_rows = []
        n_gen = int(cfg.get("n_generate", 12))
        gdf = df[df.index < n_gen] if shard == 0 else df.iloc[:0]
        gen = {"max_new_tokens": int(cfg.get("max_new_tokens", 400))}
        l0 = layers[0]
        texts_by_cond: dict[tuple, list[str]] = {}
        for name in [n for n in vecs if not n.startswith("random")]:
            for a in [float(x) for x in cfg.get("generate_alphas", [-8, -4, 0, 4, 8])]:
                if gdf.empty:
                    break
                v = vecs[name][l0] * a * units[l0]
                ctx = (lambda v=v: add_vector(rt.model, {l0: v})) if a else None
                texts, fins, ntoks = generate_hf(gdf, model_key, gen, model=rt.model, tok=rt.tok, greedy=True, ctx_factory=ctx)
                texts_by_cond[(name, a)] = texts
                for r, t, fin, nt in zip(gdf.itertuples(), texts, fins, ntoks):
                    rec = {c: getattr(r, c) for c in gdf.columns if c != "prompt"}
                    rec.update({"run_id": f"local__{model_key}__steering", "model_key": model_key,
                                "call_id": sha256([r.prompt_id, "steer", name, l0, a]), "status": "ok" if t else "error",
                                "response": t, "finish_reason": fin, "output_tokens": nt, "cost_usd": 0.0,
                                "request_ts": utcnow(), "intervention": "steering", "direction": name, "layer": l0, "alpha": a})
                    gen_rows.append(rec)
        append_parquet(out / "generations" / "logs", gen_rows)

        # side effects (shard 0 only): perplexity per condition at layer l0, refusal rate of generations
        if shard == 0:
            se = icfg["side_effects"]
            ppl_texts = load_ppl_texts(se["ppl_texts"], int(se.get("ppl_max_docs", 50)))
            rows = []
            for name in vecs:
                for a in alphas:
                    v = vecs[name][l0] * a * units[l0]
                    ctx = (lambda v=v: add_vector(rt.model, {l0: v})) if a else None
                    r = {"direction": name, "layer": l0, "alpha": a,
                         "refusal_rate": refusal_rate(texts_by_cond.get((name, a), []))}
                    for lang, texts in ppl_texts.items():
                        r[f"ppl_{lang}"] = perplexity(rt.model, rt.tok, texts, ctx, int(se.get("ppl_max_tokens", 512)))
                    rows.append(r)
            append_parquet(out / "side_effects", rows)
        return {"prompts": len(df), "layers": layers, "conditions": len(vecs) * len(layers) * len(alphas)}

    run_shard(model_key, "steering", shard, work)


def main(argv=None) -> None:
    activate()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--shard", type=int, required=True)
    a = ap.parse_args(argv)
    run(a.model, a.shard)


if __name__ == "__main__":
    main()
