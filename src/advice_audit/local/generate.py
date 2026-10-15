"""Step 0 behavior (spec §6.3): generate the reduced Study 1 protocol with a local model.

Output: Parquet in the Part A log schema, so the same coding and analysis code runs on it
(`advice-audit code --run outputs/local/<model>/behavior` on the online side).
Uses vLLM when available; falls back to HF transformers `generate` (slow; CPU tests).

Entry point: python -m advice_audit.local.generate --model <key> --shard <k>
"""
from __future__ import annotations

from .offline_guard import activate  # noqa: I001  (imported first by design)

import argparse

import pandas as pd

from ..common.config import load_local_models
from ..common.ids import call_id, stable_int
from ..common.io import append_parquet, utcnow
from ..common.logs import get_logger
from .modeling import format_chat, load_model, load_tokenizer, resolve_cfg
from .shards import run_shard

log = get_logger("advice_audit.local.generate")


def _log_rows(df: pd.DataFrame, texts: list[str], finishes: list[str], ntoks: list[int], model_key: str, run_id: str) -> list[dict]:
    rows = []
    for r, text, fin, nt in zip(df.itertuples(), texts, finishes, ntoks):
        rec = {c: getattr(r, c) for c in df.columns if c not in ("prompt",)}
        rec.update({
            "run_id": run_id, "model_key": model_key, "slug": model_key, "provider_served": "local",
            "call_id": call_id(domain=r.domain, mode=r.mode, profile_id=r.profile_id, cell=r.cell,
                               gender_cond=r.gender_cond, name=r.name, paraphrase_id=int(r.paraphrase_id),
                               sample_idx=int(r.sample_idx), model_key=model_key, prompt_version=r.prompt_version),
            "status": "ok" if text else "error", "response": text, "finish_reason": fin,
            "output_tokens": nt, "cost_usd": 0.0, "request_ts": utcnow(), "error": None,
        })
        from ..api.parse import parse_response

        rec.update(parse_response(text))
        if rec["refusal_flag"]:
            rec["status"] = "refusal"
        rows.append(rec)
    return rows


def generate_vllm(df: pd.DataFrame, model_key: str, gen: dict) -> tuple[list[str], list[str], list[int]]:
    from vllm import LLM, SamplingParams

    cfg = resolve_cfg(model_key)
    llm = LLM(model=cfg.repo_id, revision=cfg.revision, dtype=cfg.dtype, quantization=cfg.quantization,
              seed=0, enforce_eager=False, max_model_len=4096)
    tok = llm.get_tokenizer()
    texts = [format_chat(tok, p, cfg).text for p in df["prompt"]]
    params = [SamplingParams(temperature=gen.get("temperature", 0.7), top_p=gen.get("top_p", 0.95),
                             max_tokens=gen.get("max_new_tokens", 800), seed=stable_int(pid, mod=2**31))
              for pid in df["prompt_id"]]
    outs = llm.generate(texts, params)
    return ([o.outputs[0].text for o in outs], [o.outputs[0].finish_reason for o in outs],
            [len(o.outputs[0].token_ids) for o in outs])


def generate_hf(df: pd.DataFrame, model_key: str, gen: dict, model=None, tok=None, greedy: bool = False,
                batch_size: int = 4, ctx_factory=None) -> tuple[list[str], list[str], list[int]]:
    import torch
    from contextlib import nullcontext

    cfg = resolve_cfg(model_key)
    tok = tok or load_tokenizer(cfg)
    model = model or load_model(cfg)
    tok.padding_side = "left"
    texts, fins, ntoks = [], [], []
    prompts = list(df["prompt"])
    seeds = [stable_int(pid, mod=2**31) for pid in df["prompt_id"]] if "prompt_id" in df else [0] * len(prompts)
    for i in range(0, len(prompts), batch_size):
        chunk = [format_chat(tok, p, cfg).text for p in prompts[i : i + batch_size]]
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        torch.manual_seed(seeds[i])
        kw = dict(max_new_tokens=gen.get("max_new_tokens", 800), pad_token_id=tok.pad_token_id)
        kw.update({"do_sample": False} if greedy else
                  {"do_sample": True, "temperature": gen.get("temperature", 0.7), "top_p": gen.get("top_p", 0.95)})
        with (ctx_factory() if ctx_factory else nullcontext()), torch.no_grad():
            out = model.generate(**enc, **kw)
        new = out[:, enc["input_ids"].shape[1]:]
        for row in new:
            ids = row.tolist()
            eos_hit = tok.eos_token_id in ids
            if eos_hit:
                ids = ids[: ids.index(tok.eos_token_id)]
            ids = [t for t in ids if t != tok.pad_token_id]
            texts.append(tok.decode(ids, skip_special_tokens=True).strip())
            fins.append("stop" if eos_hit else "length")
            ntoks.append(len(ids))
    return texts, fins, ntoks


def run(model_key: str, shard: int, backend: str = "auto") -> None:
    gen = load_local_models().generation
    run_id = f"local__{model_key}__behavior"

    def work(df: pd.DataFrame, out):
        use_vllm = backend == "vllm"
        if backend == "auto":
            try:
                import vllm  # noqa: F401

                use_vllm = True
            except ImportError:
                use_vllm = False
        texts, fins, ntoks = (generate_vllm if use_vllm else generate_hf)(df, model_key, gen)
        append_parquet(out / "logs", _log_rows(df, texts, fins, ntoks, model_key, run_id))
        return {"n": len(df), "backend": "vllm" if use_vllm else "hf"}

    run_shard(model_key, "behavior", shard, work)


def main(argv=None) -> None:
    activate()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--backend", default="auto", choices=["auto", "vllm", "hf"])
    a = ap.parse_args(argv)
    run(a.model, a.shard, a.backend)


if __name__ == "__main__":
    main()
