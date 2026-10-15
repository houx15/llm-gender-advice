"""Residual-stream hooks with plain PyTorch (spec §6.4) — no TransformerLens dependency.

Convention: "layer l" = the OUTPUT of decoder block l (the residual stream after block l),
i.e. `output_hidden_states[l + 1]` for l < n_layers - 1 (HF applies the final norm to the
last hidden state, so the last block's raw output differs from hidden_states[-1]).

All hook editors take per-row positions: a LongTensor [batch] (one position per row),
a list of LongTensors, or None (= every position).
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterable

import torch
from torch import nn

# Candidate attribute paths to the decoder layer ModuleList, by architecture.
LAYER_PATHS = (
    "model.layers",                    # Llama, Mistral, Qwen2/3, Gemma, Phi-3, OLMo
    "model.language_model.layers",     # multimodal wrappers (Gemma 3, Qwen-VL style)
    "language_model.model.layers",
    "model.model.layers",
    "transformer.h",                   # GPT-2, GPT-J, Falcon
    "gpt_neox.layers",                 # Pythia / GPT-NeoX
    "model.decoder.layers",            # OPT
    "transformer.blocks",              # MPT
)


def _get(obj, path: str):
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def decoder_layers(model: nn.Module) -> nn.ModuleList:
    for p in LAYER_PATHS:
        try:
            mod = _get(model, p)
        except AttributeError:
            continue
        if isinstance(mod, nn.ModuleList) and len(mod) > 0:
            return mod
    raise ValueError(f"cannot locate decoder layers for {type(model).__name__}; add its path to hooks.LAYER_PATHS")


def layer_path(model: nn.Module) -> str:
    for p in LAYER_PATHS:
        try:
            mod = _get(model, p)
        except AttributeError:
            continue
        if isinstance(mod, nn.ModuleList) and len(mod) > 0:
            return p
    raise ValueError("no decoder layer path")


def _hidden(out):
    return out[0] if isinstance(out, tuple) else out


def _replace(out, h):
    if isinstance(out, tuple):
        return (h,) + tuple(out[1:])
    return h


def _rows_positions(positions, batch: int, device) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Flatten per-row positions into (row_idx, pos_idx) index tensors; None = all positions."""
    if positions is None:
        return None
    if isinstance(positions, torch.Tensor) and positions.dim() == 1:
        rows = torch.arange(batch, device=device)
        keep = positions >= 0
        return rows[keep], positions.to(device)[keep]
    rows, pos = [], []
    for r, p in enumerate(positions):
        p = torch.as_tensor(p, device=device).reshape(-1)
        p = p[p >= 0]
        rows.append(torch.full_like(p, r))
        pos.append(p)
    return torch.cat(rows), torch.cat(pos)


@contextmanager
def capture(model: nn.Module, layers: Iterable[int], positions=None, store_all: bool = False):
    """Capture residual outputs at `layers`.

    Yields a dict that is filled during the forward pass:
      store_all=False: {layer: Tensor[n_selected, d]} rows ordered as (row, position) pairs
      store_all=True : {layer: Tensor[batch, seq, d]}
    Only the FIRST forward pass is captured per layer (the prefix pass in generation)."""
    mods = decoder_layers(model)
    store: dict[int, torch.Tensor] = {}
    handles = []

    def make(l):
        def hook(_m, _inp, out):
            if l in store:
                return
            h = _hidden(out)
            if store_all or positions is None:
                store[l] = h.detach()
            else:
                idx = _rows_positions(positions, h.shape[0], h.device)
                store[l] = h[idx[0], idx[1]].detach()
        return hook

    try:
        for l in layers:
            handles.append(mods[l].register_forward_hook(make(l)))
        yield store
    finally:
        for h in handles:
            h.remove()


@contextmanager
def edit(model: nn.Module, edits: dict[int, Callable[[torch.Tensor], torch.Tensor]]):
    """Generic residual edit: {layer: fn(hidden [batch, seq, d]) -> hidden}."""
    mods = decoder_layers(model)
    handles = []

    def make(fn):
        def hook(_m, _inp, out):
            return _replace(out, fn(_hidden(out)))
        return hook

    try:
        for l, fn in edits.items():
            handles.append(mods[l].register_forward_hook(make(fn)))
        yield
    finally:
        for h in handles:
            h.remove()


def add_vector(model: nn.Module, vectors: dict[int, torch.Tensor], positions=None):
    """Steering: h[..., pos, :] += v at each layer (all positions if positions is None)."""

    def make(v):
        def fn(h):
            vv = v.to(device=h.device, dtype=h.dtype)
            if positions is None:
                return h + vv
            h = h.clone()
            idx = _rows_positions(positions, h.shape[0], h.device)
            keep = idx[1] < h.shape[1]
            h[idx[0][keep], idx[1][keep]] += vv
            return h
        return fn

    return edit(model, {l: make(v) for l, v in vectors.items()})


def patch(model: nn.Module, layer: int, positions: torch.Tensor, values: torch.Tensor):
    """Activation patching: set h[row, positions[row]] = values[row] at `layer` (prefix pass only:
    positions beyond the current sequence length are ignored, so cached decoding steps are untouched)."""

    def fn(h):
        idx = _rows_positions(positions, h.shape[0], h.device)
        keep = idx[1] < h.shape[1]
        if h.shape[1] == 1 and not keep.all():
            return h
        h = h.clone()
        h[idx[0][keep], idx[1][keep]] = values.to(device=h.device, dtype=h.dtype)[keep]
        return h

    return edit(model, {layer: fn})


def apply_fn(model: nn.Module, fns: dict[int, Callable[[torch.Tensor], torch.Tensor]]):
    """Apply a per-vector transform (e.g. a LEACE eraser) to every position at each layer."""

    def make(f):
        def fn(h):
            shp = h.shape
            out = f(h.reshape(-1, shp[-1]).float()).to(h.dtype)
            return out.reshape(shp)
        return fn

    return edit(model, {l: make(f) for l, f in fns.items()})


def n_layers(model: nn.Module) -> int:
    return len(decoder_layers(model))
