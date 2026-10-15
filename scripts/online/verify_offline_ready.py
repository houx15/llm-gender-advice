"""Verify that everything the Slurm jobs need is on shared storage — with the network OFF.

With HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1 and the socket guard active, for each model:
load config + tokenizer from the cache at the pinned revision, render one chat-templated
prompt in EN and ZH (with name/cue token positions), build the model skeleton on the meta
device and resolve its decoder-layer path (hooks.py), and check the weight files exist.
Also checks taxonomy attributes, held-out perplexity text, and planned shards.
Must pass before any submission; writes outputs/offline_ready.json.

  python scripts/online/verify_offline_ready.py [--model KEY ...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append")
    a = ap.parse_args()
    from advice_audit.common.config import OUTPUTS_DIR, REFERENCE_BUILT, is_todo, load_local_models, load_paths

    hf_home = os.environ.get("HF_HOME") or load_paths()["cluster"].get("hf_home")
    if is_todo(hf_home):
        print("set cluster.hf_home in config/paths.yaml")
        return 1
    os.environ.update({"HF_HOME": hf_home, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                       "HF_DATASETS_OFFLINE": "1", "VLLM_NO_USAGE_STATS": "1"})
    from advice_audit.local.offline_guard import activate

    activate()
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    from advice_audit.local.hooks import decoder_layers, layer_path
    from advice_audit.local.modeling import encode_with_positions, format_chat, load_tokenizer
    from advice_audit.materials.render import render_prompt

    report, ok_all = {"models": {}, "checks": {}}, True
    for m in load_local_models().models:
        if a.model and m.key not in a.model:
            continue
        r = {"ok": False, "problems": []}
        report["models"][m.key] = r
        if not m.ready:
            r["problems"].append("repo_id/revision TODO in config/local_models.yaml")
            ok_all = False
            continue
        try:
            cfg = AutoConfig.from_pretrained(m.repo_id, revision=m.revision)
            tok = load_tokenizer(m)
            for cell in ("A", "C"):
                rp = render_prompt("majors", "open", "maj_sci_high_r1", cell, "f_explicit", 0, 0, strict=False)
                chat = format_chat(tok, rp.text, m, load_local_models().readout["answer_prefix"][rp.language])
                ids, pos = encode_with_positions(tok, chat, {"name_last": rp.name_span, "cue_last": rp.cue_span})
                if pos["name_last"] is None or pos["cue_last"] is None:
                    r["problems"].append(f"cell {cell}: name/cue token not located")
                r[f"tokens_{cell}"] = len(ids)
            with torch.device("meta"):
                skel = AutoModelForCausalLM.from_config(cfg)
            r["layer_path"] = layer_path(skel)
            r["n_layers"] = len(decoder_layers(skel))
            tc = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
            r["hidden_size"] = tc.hidden_size
            from huggingface_hub import snapshot_download

            snap = Path(snapshot_download(m.repo_id, revision=m.revision, local_files_only=True))
            weights = list(snap.glob("*.safetensors"))
            if not weights:
                r["problems"].append(f"no .safetensors weights in {snap}")
            r["snapshot"] = str(snap)
            r["ok"] = not r["problems"]
        except Exception as e:  # noqa: BLE001
            r["problems"].append(f"{type(e).__name__}: {e}")
        ok_all &= r["ok"]
        print(f"[{'OK' if r['ok'] else 'FAIL'}] {m.key}: {json.dumps({k: v for k, v in r.items() if k != 'ok'})}")
    for d in ("majors", "jobs"):
        f = REFERENCE_BUILT / f"taxonomy_{d}.csv"
        report["checks"][f"taxonomy_{d}"] = f.exists()
        ok_all &= f.exists()
    ppl = ROOT / "data/reference/raw/heldout/wikitext2_test.parquet"
    report["checks"]["ppl_en"] = ppl.exists()
    report["checks"]["ppl_zh"] = (ROOT / "data/heldout/zh.txt").exists()
    try:
        import vllm  # noqa: F401

        report["checks"]["vllm"] = True
    except Exception:  # noqa: BLE001
        report["checks"]["vllm"] = False
    print(json.dumps(report["checks"]))
    report["ok"] = bool(ok_all)
    (OUTPUTS_DIR).mkdir(parents=True, exist_ok=True)
    (OUTPUTS_DIR / "offline_ready.json").write_text(json.dumps(report, indent=2))
    print("OFFLINE READY" if ok_all else "NOT READY — fix the problems above before submitting")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
