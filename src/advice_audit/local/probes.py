"""Layer-wise linear probes for inferred gender (spec §6.3, Step 1), then directions.

- logistic probes (torch, L2, full-batch LBFGS; GPU if available, CPU ok)
- grouped CV by profile_id (no profile leaks across folds)
- label-permutation baseline
- cross-lingual transfer: train EN cells (A, B) -> test ZH cell C, and ZH -> EN
Outputs to outputs/local/<model>/probes/: probes.parquet, summary.json, and the directions
(directions.py) as .pt tensors + metadata.

Entry point: python -m advice_audit.local.probes --model <key>
"""
from __future__ import annotations

from .offline_guard import activate  # noqa: I001

import argparse
import json

import numpy as np
import pandas as pd
import torch

from ..common.io import utcnow, write_json
from ..common.logs import get_logger
from .shards import stage_dir

log = get_logger("advice_audit.local.probes")


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def fit_logistic(X: torch.Tensor, y: torch.Tensor, l2: float = 1e-2, max_iter: int = 200):
    """Standardized L2 logistic regression; returns (w, b, mean, std) in original units."""
    dev = X.device
    mu, sd = X.mean(0), X.std(0).clamp_min(1e-6)
    Z = (X - mu) / sd
    w = torch.zeros(Z.shape[1], device=dev, requires_grad=True)
    b = torch.zeros(1, device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=max_iter, line_search_fn="strong_wolfe")
    yf = y.float()

    def closure():
        opt.zero_grad()
        logits = Z @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, yf) + l2 * (w ** 2).sum() / 2
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach(), b.detach(), mu, sd


def predict(model, X: torch.Tensor) -> torch.Tensor:
    w, b, mu, sd = model
    return (((X - mu) / sd) @ w + b) > 0


def grouped_folds(groups: np.ndarray, k: int, seed: int = 0) -> list[np.ndarray]:
    uniq = np.array(sorted(set(groups)))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    k = min(k, len(uniq))
    return [np.isin(groups, part) for part in np.array_split(uniq, k)]


def cv_accuracy(X: torch.Tensor, y: torch.Tensor, groups: np.ndarray, k: int, l2: float, max_iter: int) -> float:
    accs, ns = [], []
    for test in grouped_folds(groups, k):
        te = torch.as_tensor(test, device=X.device)
        if te.all() or (~te).sum() < 4:
            continue
        m = fit_logistic(X[~te], y[~te], l2, max_iter)
        accs.append((predict(m, X[te]) == y[te].bool()).float().mean().item())
        ns.append(int(te.sum()))
    return float(np.average(accs, weights=ns)) if accs else float("nan")


def run_probes(acts: torch.Tensor, valid: torch.Tensor, idx: pd.DataFrame, meta: dict, cfg: dict) -> pd.DataFrame:
    dev = _device()
    k, l2, it, nperm = cfg.get("cv_folds", 5), cfg.get("l2", 1e-2), cfg.get("max_iter", 200), cfg.get("n_permutations", 5)
    keep = (idx["gender_cond"] != "baseline").to_numpy()
    rng = np.random.default_rng(0)
    rows = []
    for pi, pos in enumerate(meta["positions"]):
        for subset in ("all", "implicit", "explicit"):
            m = keep & valid[:, pi].numpy()
            if subset != "all":
                m &= idx["gender_cond"].str.endswith(subset).to_numpy()
            if m.sum() < 20:
                continue
            sub = idx[m]
            y = torch.as_tensor(sub["female"].astype(int).to_numpy(), device=dev)
            groups = sub["profile_id"].to_numpy()
            en = (sub["language"] == "en").to_numpy()
            for li, layer in enumerate(meta["layers"]):
                X = acts[m, pi, li].to(dev).float()
                acc = cv_accuracy(X, y, groups, k, l2, it)
                perm = []
                for _ in range(nperm):
                    yp = y[torch.as_tensor(rng.permutation(len(y)), device=dev)]
                    perm.append(cv_accuracy(X, yp, groups, k, l2, it))
                row = {"position": pos, "subset": subset, "layer": layer, "n": int(m.sum()), "cv_acc": acc,
                       "perm_mean": float(np.mean(perm)) if perm else np.nan,
                       "perm_max": float(np.max(perm)) if perm else np.nan}
                for name, tr, te in (("en_to_zh", en, ~en), ("zh_to_en", ~en, en)):
                    if tr.sum() >= 10 and te.sum() >= 10:
                        mdl = fit_logistic(X[torch.as_tensor(tr, device=dev)], y[torch.as_tensor(tr, device=dev)], l2, it)
                        te_t = torch.as_tensor(te, device=dev)
                        row[f"transfer_{name}"] = (predict(mdl, X[te_t]) == y[te_t].bool()).float().mean().item()
                rows.append(row)
            log.info(f"probes: position={pos} subset={subset} done")
    return pd.DataFrame(rows)


def probe_weights(acts, valid, idx, meta, pos: str = "last", l2: float = 1e-2, max_iter: int = 200) -> torch.Tensor:
    """Probe weight vector per layer (original units, unit-normalized) as an alternative g."""
    pi = meta["positions"].index(pos)
    m = (idx["gender_cond"] != "baseline").to_numpy() & valid[:, pi].numpy()
    y = torch.as_tensor(idx.loc[m, "female"].astype(int).to_numpy())
    out = []
    for li in range(len(meta["layers"])):
        w, _b, _mu, sd = fit_logistic(acts[m, pi, li].float(), y, l2, max_iter)
        v = w / sd
        out.append(v / v.norm())
    return torch.stack(out)


def main(argv=None) -> None:
    activate()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--skip-probes", action="store_true")
    a = ap.parse_args(argv)
    from .directions import run_directions
    from .extract import load_activations
    from .prompts import interp_cfg

    cfg = interp_cfg()
    out = stage_dir(a.model, "probes")
    out.mkdir(parents=True, exist_ok=True)
    acts, valid, idx, meta = load_activations(a.model, "extract")
    if not a.skip_probes:
        res = run_probes(acts, valid, idx, meta, cfg["probes"])
        res.to_parquet(out / "probes.parquet", index=False)
        best = res.sort_values("cv_acc", ascending=False).groupby(["position", "subset"]).head(1)
        write_json(out / "summary.json", {"finished_at": utcnow(), "best_layers": best.to_dict("records")})
    run_directions(a.model, acts, valid, idx, meta, cfg)
    (out / "DONE").write_text(json.dumps({"finished_at": utcnow()}))


if __name__ == "__main__":
    main()
