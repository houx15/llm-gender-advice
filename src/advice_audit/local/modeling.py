"""Offline model/tokenizer loading and chat formatting for Part B."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from ..common.config import LocalModelCfg, load_local_models


def resolve_cfg(model_key: str) -> LocalModelCfg:
    cfg = load_local_models().get(model_key)
    if not cfg.ready and not Path(cfg.repo_id).exists():
        raise SystemExit(f"local model {model_key}: repo_id/revision still TODO in config/local_models.yaml")
    return cfg


def _source(cfg: LocalModelCfg) -> tuple[str, dict]:
    """A local directory (tests / pre-downloaded) or a pinned HF repo from the offline cache."""
    if Path(cfg.repo_id).exists():
        return cfg.repo_id, {}
    return cfg.repo_id, {"revision": cfg.revision, "cache_dir": os.environ.get("HF_HUB_CACHE")}


def load_tokenizer(cfg: LocalModelCfg):
    from transformers import AutoTokenizer

    src, kw = _source(cfg)
    tok = AutoTokenizer.from_pretrained(src, **{k: v for k, v in kw.items() if v is not None})
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(cfg: LocalModelCfg, device: str | None = None):
    import torch
    from transformers import AutoModelForCausalLM

    src, kw = _source(cfg)
    dtype = getattr(torch, cfg.dtype) if isinstance(cfg.dtype, str) else cfg.dtype
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(src, dtype=dtype, **{k: v for k, v in kw.items() if v is not None})
    model.to(device).eval()
    return model


@dataclass
class ChatPrompt:
    text: str             # full templated string (incl. answer prefix if any)
    user_offset: int      # char offset of the user message inside `text`
    prefix_offset: int | None = None  # char offset of the answer prefix


def format_chat(tok, user_text: str, cfg: LocalModelCfg | None = None, answer_prefix: str | None = None) -> ChatPrompt:
    kwargs = dict(cfg.chat_template_kwargs) if cfg else {}
    text = tok.apply_chat_template([{"role": "user", "content": user_text}], tokenize=False,
                                   add_generation_prompt=True, **kwargs)
    off = text.find(user_text)
    if off < 0:
        raise ValueError("user text not found verbatim in the chat template output")
    pre_off = None
    if answer_prefix:
        pre_off = len(text)
        text = text + answer_prefix
    return ChatPrompt(text=text, user_offset=off, prefix_offset=pre_off)


def token_index_for_span(offsets: list[tuple[int, int]], span: tuple[int, int] | None, which: str = "last") -> int | None:
    """Index of the last (or first) token overlapping a char span; robust to CJK merges."""
    if span is None:
        return None
    s, e = span
    hits = [i for i, (a, b) in enumerate(offsets) if a < e and b > s and b > a]
    if not hits:
        return None
    return hits[-1] if which == "last" else hits[0]


def encode_with_positions(tok, chat: ChatPrompt, spans: dict[str, tuple[int, int] | None]) -> tuple[list[int], dict[str, int | None]]:
    """Token ids of the templated text and token positions for named char spans
    (spans are relative to the USER text; shifted by chat.user_offset)."""
    enc = tok(chat.text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], [tuple(o) for o in enc["offset_mapping"]]
    pos = {}
    for k, sp in spans.items():
        pos[k] = token_index_for_span(offsets, (sp[0] + chat.user_offset, sp[1] + chat.user_offset) if sp else None)
    pos["last"] = len(ids) - 1
    return ids, pos
