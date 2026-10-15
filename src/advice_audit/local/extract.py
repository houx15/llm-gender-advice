"""Activation extraction (spec §6.3): forward passes with hooks on every decoder layer,
capturing the residual stream at named token positions (name/cue tokens, last prompt token).

Per shard: activations.safetensors with tensor `acts` [n_prompts, n_positions, n_layers, d]
(fp16) and `valid` [n_prompts, n_positions] (0 where a position does not exist, e.g. no name
in the baseline), plus index.parquet (prompt fields + token positions).

Entry point: python -m advice_audit.local.extract --model <key> --shard <k> [--stage extract|concepts]
"""
from __future__ import annotations

from .offline_guard import activate  # noqa: I001

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..common.config import load_local_models
from ..common.logs import get_logger
from .hooks import capture, n_layers
from .modeling import encode_with_positions, format_chat, load_model, load_tokenizer, resolve_cfg
from .readout import label_texts  # noqa: F401  (keeps readout labels consistent)
from .shards import run_shard, shard_dirs

log = get_logger("advice_audit.local.extract")
POSITIONS = {"extract": None, "concepts": ["label_last", "last"]}


def answer_prefix(language: str) -> str:
    return load_local_models().readout["answer_prefix"][language]


def prepare(tok, cfg, row, with_prefix: bool = True) -> tuple[list[int], dict]:
    """Templated ids (+ answer prefix) and token positions of name, cue, label, last."""
    lang = row["language"]
    chat = format_chat(tok, row["prompt"], cfg, answer_prefix(lang) if with_prefix else None)
    spans = {}
    for k in ("name", "cue", "label"):
        col = f"{k}_span"
        sp = json.loads(row[col]) if col in row and isinstance(row[col], str) else None
        spans[f"{k}_last"] = tuple(sp) if sp else None
    ids, pos = encode_with_positions(tok, chat, spans)
    return ids, pos


@torch.no_grad()
def extract_rows(model, tok, cfg, df: pd.DataFrame, positions: list[str], layers: list[int], batch_size: int = 8,
                 with_prefix: bool = True):
    """Returns acts [n, P, L, d] float16, valid [n, P] bool, positions table.
    Design prompts include the readout answer prefix, so `last` is the readout position."""
    dev = next(model.parameters()).device
    prepped = [prepare(tok, cfg, r, with_prefix) for _, r in df.iterrows()]
    n, P, L = len(df), len(positions), len(layers)
    d = model.config.get_text_config().hidden_size if hasattr(model.config, "get_text_config") else model.config.hidden_size
    acts = torch.zeros((n, P, L, d), dtype=torch.float16)
    valid = torch.zeros((n, P), dtype=torch.bool)
    pos_tab = []
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    for b0 in range(0, n, batch_size):
        chunk = prepped[b0 : b0 + batch_size]
        T = max(len(ids) for ids, _ in chunk)
        ids = torch.full((len(chunk), T), pad, dtype=torch.long)
        mask = torch.zeros((len(chunk), T), dtype=torch.long)
        for i, (x, _) in enumerate(chunk):
            ids[i, : len(x)] = torch.tensor(x)  # right padding keeps absolute positions
            mask[i, : len(x)] = 1
        with capture(model, layers, store_all=True) as store:
            model(input_ids=ids.to(dev), attention_mask=mask.to(dev))
        for i, (_, pos) in enumerate(chunk):
            pos_tab.append({p: pos.get(p) for p in positions})
            for j, p in enumerate(positions):
                t = pos.get(p)
                if t is None:
                    continue
                valid[b0 + i, j] = True
                for li, l in enumerate(layers):
                    acts[b0 + i, j, li] = store[l][i, t].to(torch.float16).cpu()
    return acts, valid, pd.DataFrame(pos_tab)


def resolve_layers(model) -> list[int]:
    spec = load_local_models().extraction.get("layers", "all")
    L = n_layers(model)
    return list(range(L)) if spec == "all" else [int(x) for x in spec]


def run(model_key: str, shard: int, stage: str = "extract") -> None:
    from safetensors.torch import save_file

    cfg = resolve_cfg(model_key)
    ext = load_local_models().extraction
    positions = POSITIONS[stage] or list(ext.get("positions", ["name_last", "cue_last", "last"]))
    state = {}

    def work(df: pd.DataFrame, out: Path):
        if "model" not in state:
            state["tok"], state["model"] = load_tokenizer(cfg), load_model(cfg)
        model, tok = state["model"], state["tok"]
        layers = resolve_layers(model)
        acts, valid, pos = extract_rows(model, tok, cfg, df, positions, layers, int(ext.get("batch_size", 8)),
                                        with_prefix=(stage == "extract"))
        save_file({"acts": acts.contiguous(), "valid": valid.to(torch.uint8).contiguous()}, str(out / "activations.safetensors"),
                  metadata={"positions": json.dumps(positions), "layers": json.dumps(layers), "model": model_key})
        idx = df.drop(columns=[c for c in ("prompt",) if c in df.columns]).reset_index(drop=True)
        idx = pd.concat([idx, pos.add_prefix("tok_")], axis=1)
        idx.to_parquet(out / "index.parquet", index=False)
        return {"n": len(df), "layers": len(layers), "positions": positions}

    run_shard(model_key, stage, shard, work)


def load_activations(model_key: str, stage: str = "extract", layers: list[int] | None = None):
    """Concatenate all DONE shards: (acts [n,P,L,d] float16 tensor, valid [n,P] bool, index df, meta)."""
    from safetensors import safe_open

    accs, vals, idxs, meta = [], [], [], None
    for d in shard_dirs(model_key, stage):
        with safe_open(str(d / "activations.safetensors"), framework="pt") as f:
            meta = f.metadata()
            a = f.get_tensor("acts")
            all_layers = json.loads(meta["layers"])
            if layers is not None:
                a = a[:, :, [all_layers.index(l) for l in layers]]
            accs.append(a)
            vals.append(f.get_tensor("valid").bool())
        idxs.append(pd.read_parquet(d / "index.parquet"))
    if not accs:
        raise SystemExit(f"no DONE shards for {model_key}/{stage}")
    meta = dict(meta)
    meta["positions"] = json.loads(meta["positions"])
    meta["layers"] = layers if layers is not None else json.loads(meta["layers"])
    return torch.cat(accs), torch.cat(vals), pd.concat(idxs, ignore_index=True), meta


def main(argv=None) -> None:
    activate()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--stage", default="extract", choices=["extract", "concepts"])
    a = ap.parse_args(argv)
    run(a.model, a.shard, a.stage)


if __name__ == "__main__":
    main()
