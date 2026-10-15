"""Category log-prob readout (spec §6.4).

After the chat-templated prompt plus a fixed answer prefix (EN "My top recommendation is:",
ZH "我最推荐的是："), compute the summed log-likelihood of each taxonomy label (EN or ZH
label matching the cell), plus a length-normalized variant. Softmax over categories gives
a distribution and expected female_share_us, prestige_isei, median_earnings, people_things.

Implementation: one prefix forward pass per prompt (KV cache), then all labels in one
right-padded batch continuing from the cache; a full-sequence path is kept for testing.
Interventions are applied by wrapping calls in hook context managers (hooks.py).
"""
from __future__ import annotations

from contextlib import nullcontext
from typing import Callable

import numpy as np
import pandas as pd
import torch

from ..materials.taxonomy import load_taxonomy

READOUT_ATTRS = ("female_share_us", "prestige_isei", "median_earnings", "people_things", "stem")


def label_texts(domain: str, language: str) -> tuple[list[str], list[str]]:
    tax = load_taxonomy(domain)
    col = "label_en" if language == "en" else "label_zh"
    sep = " " if language == "en" else ""
    return tax["id"].tolist(), [sep + x for x in tax[col].tolist()]


def label_token_ids(tok, labels: list[str]) -> list[list[int]]:
    return [tok(l, add_special_tokens=False)["input_ids"] for l in labels]


@torch.no_grad()
def label_logprobs(model, prompt_ids: list[int], label_ids: list[list[int]], ctx: Callable | None = None,
                   use_cache: bool = True, prefix_only: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Summed log p(label | prompt) for every label, and label token counts.

    `ctx` is a zero-arg context-manager factory wrapping the forward passes (interventions).
    prefix_only=True applies it to the prompt pass only (position-specific patching, whose
    positions refer to prompt tokens); otherwise to both passes (steering, erasure)."""
    dev = next(model.parameters()).device
    ctx = ctx or nullcontext
    lens = np.array([len(l) for l in label_ids])
    if use_cache:
        try:
            return _with_cache(model, prompt_ids, label_ids, ctx, dev, nullcontext if prefix_only else ctx), lens
        except (AttributeError, TypeError, NotImplementedError):
            pass
    return _full_sequence(model, prompt_ids, label_ids, ctx, dev), lens


def _with_cache(model, prompt_ids, label_ids, ctx, dev, label_ctx) -> np.ndarray:
    n = len(label_ids)
    p = torch.tensor([prompt_ids], device=dev)
    with ctx():
        out = model(input_ids=p, use_cache=True)
    first_lp = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
    cache = out.past_key_values
    if hasattr(cache, "batch_repeat_interleave"):
        cache.batch_repeat_interleave(n)
    else:
        raise NotImplementedError("cache repeat not supported")
    L = max(len(l) for l in label_ids)
    ids = torch.full((n, L), 0, dtype=torch.long, device=dev)
    mask = torch.zeros((n, L), dtype=torch.long, device=dev)
    for i, l in enumerate(label_ids):
        ids[i, : len(l)] = torch.tensor(l, device=dev)
        mask[i, : len(l)] = 1
    attn = torch.cat([torch.ones((n, len(prompt_ids)), dtype=torch.long, device=dev), mask], dim=1)
    total = np.array([first_lp[l[0]].item() for l in label_ids])
    if L > 1:
        with label_ctx():
            out2 = model(input_ids=ids, attention_mask=attn, past_key_values=cache, use_cache=True)
        lp = torch.log_softmax(out2.logits.float(), dim=-1)
        for i, l in enumerate(label_ids):
            for j in range(1, len(l)):
                total[i] += lp[i, j - 1, l[j]].item()
    return total


def _full_sequence(model, prompt_ids, label_ids, ctx, dev) -> np.ndarray:
    seqs = [prompt_ids + l for l in label_ids]
    L = max(len(s) for s in seqs)
    n = len(seqs)
    ids = torch.zeros((n, L), dtype=torch.long, device=dev)
    mask = torch.zeros((n, L), dtype=torch.long, device=dev)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s, device=dev)
        mask[i, : len(s)] = 1
    with ctx():
        out = model(input_ids=ids, attention_mask=mask)
    lp = torch.log_softmax(out.logits.float(), dim=-1)
    P = len(prompt_ids)
    total = np.zeros(n)
    for i, l in enumerate(label_ids):
        for j, t in enumerate(l):
            total[i] += lp[i, P + j - 1, t].item()
    return total


def distribution(logp: np.ndarray, lens: np.ndarray | None = None, normalize_length: bool = False) -> np.ndarray:
    s = logp / lens if (normalize_length and lens is not None) else logp
    s = s - s.max()
    p = np.exp(s)
    return p / p.sum()


def expected_attributes(p: np.ndarray, attrs: pd.DataFrame) -> dict[str, float]:
    out = {}
    for a in READOUT_ATTRS:
        v = attrs[a].to_numpy(dtype=float)
        m = ~np.isnan(v)
        out[f"E_{a}"] = float((p[m] * v[m]).sum() / p[m].sum()) if m.any() else float("nan")
    out["entropy"] = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    return out


def summarize(logp: np.ndarray, lens: np.ndarray, domain: str, cat_ids: list[str], orientation: str | None = None) -> dict:
    """Expected attributes under sum and length-normalized scoring, top category, off-track mass."""
    from ..materials.taxonomy import off_track

    tax = load_taxonomy(domain).set_index("id").loc[cat_ids]
    p = distribution(logp)
    pn = distribution(logp, lens, normalize_length=True)
    res = {**expected_attributes(p, tax), **{f"{k}_lennorm": v for k, v in expected_attributes(pn, tax).items()}}
    res["top_category"] = cat_ids[int(np.argmax(p))]
    res["top_category_lennorm"] = cat_ids[int(np.argmax(pn))]
    ot = np.array([off_track(t, orientation) for t in tax["track"]])
    res["off_track_mass"] = float(np.nansum(p * ot)) if orientation not in (None, "balanced") else float("nan")
    return res


# ---------------------------------------------------------------- stage: readout on Step 0 prompts
def run_stage(model_key: str, shard: int) -> None:
    """Readout for the Step 0 prompts, used to validate the readout against the first coded
    free-text recommendation (spec §6.4)."""
    from ..common.io import append_parquet
    from .runtime import Runtime
    from .shards import run_shard

    state = {}

    def work(df: pd.DataFrame, out):
        rt = state.setdefault("rt", Runtime(model_key))
        rows = []
        for _, row in df.iterrows():
            s, _ = rt.readout(row)
            rows.append({"prompt_id": row["prompt_id"], "domain": row["domain"], "mode": row["mode"],
                         "profile_id": row["profile_id"], "cell": row["cell"], "gender_cond": row["gender_cond"],
                         "name_idx": row["name_idx"], "paraphrase_id": row["paraphrase_id"], **s})
        append_parquet(out / "results", rows)
        return {"n": len(df)}

    run_shard(model_key, "readout", shard, work)


def main(argv=None) -> None:
    import argparse

    from .offline_guard import activate

    activate()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--shard", type=int, required=True)
    a = ap.parse_args(argv)
    run_stage(a.model, a.shard)


if __name__ == "__main__":
    main()
