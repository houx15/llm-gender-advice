"""Side-effect metrics for every intervention (spec §6.6): perplexity on held-out EN+ZH
text, refusal rate, readout entropy, off-track share (from the readout distribution)."""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..api.parse import is_refusal, extract_ranked
from ..common.config import ROOT
from ..common.logs import get_logger

log = get_logger("advice_audit.local.sidefx")


def load_ppl_texts(paths: dict[str, str], max_docs: int = 50) -> dict[str, list[str]]:
    out = {}
    for lang, p in paths.items():
        path = Path(p) if Path(p).is_absolute() else ROOT / p
        if not path.exists():
            log.warning(f"perplexity text for {lang} missing: {path} (see QUESTIONS.md)")
            continue
        if path.suffix == ".parquet":
            texts = pd.read_parquet(path)["text"].tolist()
        else:
            texts = path.read_text(encoding="utf-8").split("\n\n")
        texts = [t.strip() for t in texts if len(t.strip()) > 200]
        out[lang] = texts[:max_docs]
    return out


@torch.no_grad()
def perplexity(model, tok, texts: list[str], ctx_factory=None, max_tokens: int = 512) -> float:
    if not texts:
        return float("nan")
    dev = next(model.parameters()).device
    nll, n = 0.0, 0
    for t in texts:
        ids = tok(t, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=max_tokens)["input_ids"].to(dev)
        if ids.shape[1] < 2:
            continue
        with (ctx_factory() if ctx_factory else nullcontext()):
            logits = model(input_ids=ids).logits.float()
        lp = torch.log_softmax(logits[0, :-1], dim=-1)
        nll -= lp.gather(1, ids[0, 1:, None]).sum().item()
        n += ids.shape[1] - 1
    return float(np.exp(nll / n)) if n else float("nan")


def refusal_rate(texts: list[str]) -> float:
    if not texts:
        return float("nan")
    return float(np.mean([is_refusal(t, len(extract_ranked(t))) for t in texts]))
