"""Concept erasure (spec §6.6): fit LEACE (concept-erasure) on gender at selected layers using
extraction activations from the FIT profiles; apply at every position during readout and
generation on held-out (EVAL) profiles.

Conditions: none, leace_L<l> (each selected layer alone), leace_all (all selected layers).
H10 leakage: under leace_all, residuals at downstream layers are saved at the last token so
gender recoverability can be probed after erasure (aggregate()).

Entry point: python -m advice_audit.local.erasure --model <key> --shard <k> | --aggregate
"""
from __future__ import annotations

from .offline_guard import activate  # noqa: I001

import argparse

import numpy as np
import pandas as pd
import torch

from ..common.ids import sha256
from ..common.io import append_parquet, utcnow, write_json
from ..common.logs import get_logger
from .hooks import apply_fn, capture, n_layers
from .prompts import erasure_fit_profiles, interp_cfg
from .shards import run_shard, shard_dirs, stage_dir

log = get_logger("advice_audit.local.erasure")


class ClosedFormLeace:
    """Minimal LEACE (Belrose et al. 2023) used only if `concept-erasure` is unavailable:
    x' = x - W^+ P W (x - mu), W = Σ_xx^{-1/2}, P = projection onto colspace(W Σ_xz)."""

    def __init__(self, X: torch.Tensor, z: torch.Tensor, eps: float = 1e-6):
        X = X.double()
        z = z.double().reshape(len(z), -1)
        self.mu = X.mean(0)
        Xc, zc = X - self.mu, z - z.mean(0)
        sxx = Xc.T @ Xc / len(X)
        sxz = Xc.T @ zc / len(X)
        evals, evecs = torch.linalg.eigh(sxx)
        evals = evals.clamp_min(eps)
        W = evecs @ torch.diag(evals.rsqrt()) @ evecs.T
        Winv = evecs @ torch.diag(evals.sqrt()) @ evecs.T
        u, _, _ = torch.linalg.svd(W @ sxz, full_matrices=False)
        P = u @ u.T
        self.proj = (Winv @ P @ W).float()
        self.mu = self.mu.float()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        dev = x.device
        return x - (x - self.mu.to(dev)) @ self.proj.to(dev).T


def fit_erasers(model_key: str, layers: list[int]) -> dict[int, object]:
    from .extract import load_activations

    acts, valid, idx, meta = load_activations(model_key, "extract", layers=layers)
    fit_prof = set(erasure_fit_profiles(interp_cfg()["erasure"]["fit_fraction"]))
    keep = (idx["gender_cond"] != "baseline").to_numpy() & idx["profile_id"].isin(fit_prof).to_numpy()
    erasers = {}
    for li, l in enumerate(layers):
        xs, zs = [], []
        for pi in range(len(meta["positions"])):  # pool all captured positions
            m = keep & valid[:, pi].numpy()
            xs.append(acts[m, pi, li].float())
            zs.append(torch.as_tensor(idx.loc[m, "female"].astype(int).to_numpy()))
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        X, z = torch.cat(xs).to(dev), torch.cat(zs).to(dev)  # eraser tensors live where they are fit
        try:
            from concept_erasure import LeaceEraser

            erasers[l] = LeaceEraser.fit(X, z)  # package default: Ledoit-Wolf shrinkage
        except ImportError:
            erasers[l] = ClosedFormLeace(X, z)
    return erasers


def run(model_key: str, shard: int) -> None:
    from .generate import generate_hf
    from .patching import selected_layers
    from .runtime import Runtime
    from .sidefx import load_ppl_texts, perplexity, refusal_rate

    icfg = interp_cfg()
    cfg = icfg["erasure"]
    state = {}

    def work(df: pd.DataFrame, out):
        rt = state.setdefault("rt", Runtime(model_key))
        L = n_layers(rt.model)
        layers = selected_layers(model_key, L) if cfg.get("layers", "auto") == "auto" else [int(x) for x in cfg["layers"]]
        if "erasers" not in state:
            ers = fit_erasers(model_key, layers)
            dev = next(rt.model.parameters()).device
            state["erasers"] = {l: _to_device(e, dev) for l, e in ers.items()}
        ers = state["erasers"]
        conds = {"none": None, **{f"leace_L{l}": {l: ers[l]} for l in layers}, "leace_all": ers}
        downstream = sorted({min(L - 1, max(layers) + k) for k in (0, 1, 4, 8)})
        res, leak_acts, leak_idx = [], [], []
        for _, row in df.iterrows():
            prep = rt.prepare(row)
            base = {"prompt_id": row["prompt_id"], "domain": row["domain"], "mode": row["mode"], "cell": row["cell"],
                    "profile_id": row["profile_id"], "gender_cond": row["gender_cond"], "female": row["female"],
                    "orientation": row.get("orientation")}
            for cname, fns in conds.items():
                ctx = (lambda fns=fns: apply_fn(rt.model, fns)) if fns else None
                s, _ = rt.readout(row, ctx=ctx, prepared=prep)
                res.append({**base, "condition": cname, **s})
            if row["gender_cond"] != "baseline":
                dev = next(rt.model.parameters()).device
                with apply_fn(rt.model, ers), capture(rt.model, downstream, store_all=True) as store, torch.no_grad():
                    rt.model(input_ids=torch.tensor([prep[0]], device=dev))
                leak_acts.append(torch.stack([store[l][0, prep[1]["last"]].float().cpu() for l in downstream]))
                leak_idx.append(base)
        append_parquet(out / "readout", res)
        if leak_acts:
            torch.save({"acts": torch.stack(leak_acts).half(), "layers": downstream}, out / "leakage.pt")
            pd.DataFrame(leak_idx).to_parquet(out / "leakage_index.parquet", index=False)

        gen_rows, texts_by = [], {}
        n_gen = int(cfg.get("n_generate", 24))
        gdf = df.head(n_gen) if shard == 0 else df.iloc[:0]
        gen = {"max_new_tokens": int(cfg.get("max_new_tokens", 400))}
        for cname in ("none", "leace_all"):
            if gdf.empty:
                break
            fns = conds[cname]
            texts, fins, ntoks = generate_hf(gdf, model_key, gen, model=rt.model, tok=rt.tok, greedy=True,
                                             ctx_factory=(lambda fns=fns: apply_fn(rt.model, fns)) if fns else None)
            texts_by[cname] = texts
            for r, t, fin, nt in zip(gdf.itertuples(), texts, fins, ntoks):
                rec = {c: getattr(r, c) for c in gdf.columns if c != "prompt"}
                rec.update({"run_id": f"local__{model_key}__erasure", "model_key": model_key,
                            "call_id": sha256([r.prompt_id, "erase", cname]), "status": "ok" if t else "error",
                            "response": t, "finish_reason": fin, "output_tokens": nt, "cost_usd": 0.0,
                            "request_ts": utcnow(), "intervention": "erasure", "condition": cname})
                gen_rows.append(rec)
        append_parquet(out / "generations" / "logs", gen_rows)
        if shard == 0:
            se = icfg["side_effects"]
            ppl = load_ppl_texts(se["ppl_texts"], int(se.get("ppl_max_docs", 50)))
            rows = []
            for cname, fns in conds.items():
                ctx = (lambda fns=fns: apply_fn(rt.model, fns)) if fns else None
                r = {"condition": cname, "refusal_rate": refusal_rate(texts_by.get(cname, []))}
                for lang, texts in ppl.items():
                    r[f"ppl_{lang}"] = perplexity(rt.model, rt.tok, texts, ctx, int(se.get("ppl_max_tokens", 512)))
                rows.append(r)
            append_parquet(out / "side_effects", rows)
        return {"prompts": len(df), "layers": layers}

    run_shard(model_key, "erasure", shard, work)


def _to_device(e, dev):
    if isinstance(e, ClosedFormLeace):
        e.mu, e.proj = e.mu.to(dev), e.proj.to(dev)
    return e  # concept-erasure erasers are fit on the model device already


def aggregate(model_key: str) -> dict:
    """Readout gaps before/after erasure and H10 leakage probes."""
    from ..common.io import read_parquet_dir
    from .probes import cv_accuracy

    dirs = shard_dirs(model_key, "erasure")
    rd = pd.concat([read_parquet_dir(d / "readout") for d in dirs], ignore_index=True)
    named = rd[rd["gender_cond"] != "baseline"]
    gaps = (named.groupby(["condition", "domain", "mode", "female"])[["E_female_share_us", "E_prestige_isei",
                                                                         "E_median_earnings", "E_people_things"]]
            .mean().unstack("female"))
    gap = pd.DataFrame({a: gaps[(a, 1)] - gaps[(a, 0)] for a in ("E_female_share_us", "E_prestige_isei",
                                                                 "E_median_earnings", "E_people_things")}).reset_index()
    out = stage_dir(model_key, "erasure")
    gap.to_csv(out / "gaps_by_condition.csv", index=False)
    leak = []
    tensors = [torch.load(d / "leakage.pt") for d in dirs if (d / "leakage.pt").exists()]
    if tensors:
        A = torch.cat([t["acts"] for t in tensors]).float()
        idx = pd.concat([pd.read_parquet(d / "leakage_index.parquet") for d in dirs if (d / "leakage.pt").exists()],
                        ignore_index=True)
        y = torch.as_tensor(idx["female"].astype(int).to_numpy())
        for li, l in enumerate(tensors[0]["layers"]):
            acc = cv_accuracy(A[:, li], y, idx["profile_id"].to_numpy(), 5, 1e-2, 200)
            leak.append({"layer": l, "probe_acc_after_erasure": acc, "n": len(y)})
        pd.DataFrame(leak).to_csv(out / "leakage_probes.csv", index=False)
    write_json(out / "aggregate.json", {"conditions": gap["condition"].unique().tolist(), "leakage": leak})
    return {"gaps": gap, "leakage": leak}


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
