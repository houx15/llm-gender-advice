"""Shared model runtime for readout-based stages (readout, patching, steering, erasure)."""
from __future__ import annotations

import pandas as pd

from .extract import prepare
from .modeling import load_model, load_tokenizer, resolve_cfg
from .readout import label_logprobs, label_texts, label_token_ids, summarize


class Runtime:
    def __init__(self, model_key: str, model=None, tok=None):
        self.cfg = resolve_cfg(model_key)
        self.model_key = model_key
        self.tok = tok or load_tokenizer(self.cfg)
        self.model = model or load_model(self.cfg)
        self._labels: dict[tuple[str, str], tuple[list[str], list[list[int]]]] = {}

    def labels(self, domain: str, language: str):
        k = (domain, language)
        if k not in self._labels:
            ids, texts = label_texts(domain, language)
            self._labels[k] = (ids, label_token_ids(self.tok, texts))
        return self._labels[k]

    def prepare(self, row: pd.Series | dict):
        return prepare(self.tok, self.cfg, row, with_prefix=True)

    def readout(self, row, ctx=None, prefix_only: bool = False, prepared=None) -> tuple[dict, object]:
        ids, _pos = prepared or self.prepare(row)
        cat_ids, lab = self.labels(row["domain"], row["language"])
        lp, lens = label_logprobs(self.model, ids, lab, ctx=ctx, prefix_only=prefix_only)
        return summarize(lp, lens, row["domain"], cat_ids, row.get("orientation")), lp
