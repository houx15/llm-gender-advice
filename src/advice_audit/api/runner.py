"""Async, resumable, budgeted execution of a call plan (spec §4.3, §5).

- Idempotent: a call is skipped if any run already logged a final outcome (ok or
  refusal) for the same call_id AND prompt_hash (so editing materials re-runs calls).
- Append-only Parquet logs: outputs/runs/<run_id>/logs/part-*.parquet
- outputs/runs/<run_id>/manifest.json: git hash, config snapshot, live prices, timestamps.
- Pre-run budget check + kill switch (estimate +20% or cap minus reserve).
"""
from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Callable

import pandas as pd

from ..common.config import CONFIG_DIR, RUNS_DIR, ApiModelCfg, load_api_models, load_budget, load_yaml
from ..common.io import append_parquet, git_hash, read_parquet_dir, utcnow, write_json
from ..common.logs import get_logger
from .budget import BudgetGuard, estimate_cost, price_for
from .client import OpenRouterClient, RefusalError, TransportFailure
from .parse import parse_response

log = get_logger("advice_audit.runner")

DESIGN_COLS = [
    "experiment", "phase", "domain", "mode", "profile_id", "orientation", "level", "cell", "language",
    "country", "gender_cond", "female", "explicit", "name", "name_idx", "paraphrase_id", "sample_idx",
    "prompt_version", "fixed_field", "model_key", "call_id", "prompt_hash",
    # coding / name-check plans
    "task", "source_run_id", "source_call_id", "coder_key", "check_cell",
]


def completed(runs_dir: Path | None = None) -> set[tuple[str, str]]:
    """(call_id, prompt_hash) pairs with a final outcome in any run."""
    runs_dir = runs_dir or RUNS_DIR
    done: set[tuple[str, str]] = set()
    if not runs_dir.exists():
        return done
    for p in runs_dir.glob("**/logs/*.parquet"):
        df = pd.read_parquet(p, columns=None)
        if df.empty or "status" not in df.columns:
            continue
        fin = df[df["status"].isin(["ok", "refusal"])]
        done.update(zip(fin["call_id"], fin["prompt_hash"]))
    return done


def load_run_logs(run_id: str, runs_dir: Path | None = None, sub: str = "") -> pd.DataFrame:
    """Latest logged row per call_id for a run (earlier error rows superseded by retries)."""
    runs_dir = runs_dir or RUNS_DIR
    d = runs_dir / run_id / sub / "logs" if sub else runs_dir / run_id / "logs"
    df = read_parquet_dir(d)
    if df.empty:
        return df
    rank = df["status"].map({"ok": 0, "refusal": 1}).fillna(2)
    df = df.assign(_r=rank).sort_values(["call_id", "_r", "request_ts"]).drop_duplicates("call_id", keep="first")
    return df.drop(columns="_r").reset_index(drop=True)


def _config_snapshot(extra: dict | None = None) -> dict:
    snap = {
        "api_models": load_yaml(CONFIG_DIR / "api_models.yaml"),
        "budget": load_yaml(CONFIG_DIR / "budget.yaml"),
    }
    if extra:
        snap.update(extra)
    return snap


async def execute_plan(
    plan: pd.DataFrame,
    model_key: str,
    run_id: str,
    phase: str,
    *,
    run_dir: Path | None = None,
    dry_run: bool = False,
    max_calls: int | None = None,
    allow_reserve: bool = False,
    concurrency: int = 8,
    json_mode: bool = False,
    max_tokens: int | None = None,
    est_input_tokens: int | None = None,
    est_output_tokens: int | None = None,
    postprocess: Callable[[str], dict] = parse_response,
    snapshot_extra: dict | None = None,
    skip_verify: bool = False,
) -> dict:
    """Run every row of `plan` (columns: call_id, prompt, prompt_hash, design fields) on one model."""
    models = load_api_models()
    mcfg: ApiModelCfg = models.get(model_key)
    budget_cfg = load_budget()
    guard = BudgetGuard(budget_cfg)
    run_dir = run_dir or (RUNS_DIR / run_id)
    logs_dir = run_dir / "logs"
    est_in = est_input_tokens or models.est_input_tokens
    est_out = est_output_tokens or models.est_output_tokens
    peak = budget_cfg.assume_peak_pricing and mcfg.peak_multiplier > 1

    done = completed() if not dry_run else set()
    pending = plan[[(c, h) not in done for c, h in zip(plan["call_id"], plan["prompt_hash"])]].reset_index(drop=True)
    if max_calls:
        pending = pending.head(max_calls)

    key = os.environ.get("OPENROUTER_API_KEY")
    verification: dict = {}
    live = None
    if dry_run and not key:
        log.info("dry run without OPENROUTER_API_KEY: using config prices")
    else:
        if not key:
            raise SystemExit("OPENROUTER_API_KEY not set (copy .env.example to .env)")
    pin, pout = price_for(mcfg, peak, None)

    async def _verify(client: OpenRouterClient):
        nonlocal verification, live, pin, pout
        try:
            verification = await client.verify_model(mcfg)
            live = verification.get("live_price")
            pin, pout = price_for(mcfg, peak, live)
        except Exception as e:  # noqa: BLE001
            verification = {"ok": None, "problems": [f"verification failed: {e}"]}

    estimate = estimate_cost(len(pending), est_in, est_out, pin, pout, budget_cfg.estimate_safety_factor)
    manifest = {
        "run_id": run_id,
        "phase": phase,
        "model": mcfg.model_dump(),
        "git_hash": git_hash(),
        "started_at": utcnow(),
        "plan_calls": int(len(plan)),
        "already_done": int(sum((c, h) in done for c, h in zip(plan["call_id"], plan["prompt_hash"]))),
        "pending": int(len(pending)),
        "dry_run": dry_run,
        "config_snapshot": _config_snapshot(snapshot_extra),
    }

    if dry_run:
        if key and not skip_verify:
            async with OpenRouterClient(key) as client:
                await _verify(client)
            estimate = estimate_cost(len(pending), est_in, est_out, pin, pout, budget_cfg.estimate_safety_factor)
        ok, msg = guard.check(estimate, phase, allow_reserve)
        manifest.update({"pricing": {"peak": peak, "in": pin, "out": pout, "live": live}, "verification": verification,
                         "estimate_usd": round(estimate, 4), "budget_check": msg, "budget_ok": ok})
        samples = run_dir / "dry_run_samples"
        samples.mkdir(parents=True, exist_ok=True)
        for i, r in enumerate(pending.head(6).itertuples()):
            (samples / f"{i:02d}_{r.call_id[:10]}.txt").write_text(r.prompt, encoding="utf-8")
        write_json(run_dir / "manifest.dryrun.json", manifest)
        log.info(f"[dry-run] {run_id}: {len(pending)} pending calls, estimate ${estimate:.3f} ({msg}); ZERO API calls")
        return manifest

    async with OpenRouterClient(key) as client:
        if not skip_verify:
            await _verify(client)
            if verification.get("ok") is False:
                raise SystemExit(f"model verification failed for {model_key}: {verification.get('problems')}")
        estimate = estimate_cost(len(pending), est_in, est_out, pin, pout, budget_cfg.estimate_safety_factor)
        ok, msg = guard.check(estimate, phase, allow_reserve)
        log.info(f"[budget] {msg}")
        manifest.update({"pricing": {"peak": peak, "in": pin, "out": pout, "live": live}, "verification": verification,
                         "estimate_usd": round(estimate, 4), "budget_check": msg})
        if not ok:
            write_json(run_dir / "manifest.refused.json", manifest)
            raise SystemExit(f"BUDGET GUARD REFUSED: {msg}")
        write_json(run_dir / "manifest.json", manifest)
        room_at_start, _ = guard.remaining(phase, allow_reserve)
        if pending.empty:
            log.info(f"{run_id}: nothing to do")
            return manifest

        sem = asyncio.Semaphore(concurrency)
        stop = asyncio.Event()
        lock = asyncio.Lock()
        buf: list[dict] = []
        stats = {"ok": 0, "refusal": 0, "error": 0, "cost": 0.0}
        design_cols = [c for c in DESIGN_COLS if c in pending.columns]

        async def one(row: pd.Series) -> None:
            if stop.is_set():
                return
            async with sem:
                if stop.is_set():
                    return
                base = {c: row[c] for c in design_cols}
                base.update({"run_id": run_id, "slug": mcfg.slug, "provider_requested": ",".join(mcfg.provider.order),
                             "request_ts": utcnow()})
                try:
                    resp = await client.chat(mcfg, row["prompt"], max_tokens=max_tokens, json_mode=json_mode)
                except RefusalError as e:
                    rec = {**base, "status": "refusal", "response": "", "error": f"provider refusal: {e}"[:500],
                           "refusal_flag": True, "cost_usd": 0.0}
                except (TransportFailure, Exception) as e:  # noqa: BLE001
                    rec = {**base, "status": "error", "response": "", "error": f"{type(e).__name__}: {e}"[:500],
                           "refusal_flag": False, "cost_usd": 0.0}
                else:
                    cost = resp["cost_usd"]
                    if cost is None:
                        cost = ((resp["input_tokens"] or 0) * pin + (resp["output_tokens"] or 0) * pout) / 1e6
                    parsed = postprocess(resp["content"])
                    status = "refusal" if parsed.get("refusal_flag") else "ok"
                    if not resp["content"]:
                        status = "error"
                    rec = {**base, "status": status, "response": resp["content"], "error": None,
                           "cost_usd": float(cost),
                           **{k: resp[k] for k in ("finish_reason", "http_status", "latency_ms", "input_tokens",
                                                   "output_tokens", "reasoning_tokens", "provider_served", "model_served")},
                           **parsed}
                    if mcfg.provider.order and resp["provider_served"] and resp["provider_served"] not in mcfg.provider.order:
                        rec["error"] = f"provider drift: served by {resp['provider_served']}"
                async with lock:
                    buf.append(rec)
                    stats[rec["status"]] += 1
                    stats["cost"] += float(rec.get("cost_usd") or 0)
                    if len(buf) >= 50:
                        append_parquet(logs_dir, buf)
                        buf.clear()
                    hit, why = guard.kill_switch(estimate, stats["cost"], room_at_start)
                    if hit:
                        if buf:
                            append_parquet(logs_dir, buf)
                            buf.clear()
                        if not stop.is_set():
                            log.error(f"[kill-switch] {why}")
                        stop.set()

        t0 = time.time()
        chunk = 200
        for c0 in range(0, len(pending), chunk):
            if stop.is_set():
                break
            await asyncio.gather(*(one(r) for _, r in pending.iloc[c0 : c0 + chunk].iterrows()))
            async with lock:
                append_parquet(logs_dir, buf)
                buf.clear()
            log.info(f"{run_id}: {min(c0 + chunk, len(pending))}/{len(pending)} | ok={stats['ok']} "
                     f"refusal={stats['refusal']} err={stats['error']} cost=${stats['cost']:.4f} | {time.time() - t0:.0f}s")

    manifest.update({"finished_at": utcnow(), "results": stats, "killed": stop.is_set(),
                     "total_spent_usd": round(guard.ledger.spent_total(), 4)})
    write_json(run_dir / "manifest.json", manifest)
    return manifest
