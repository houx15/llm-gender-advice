"""Download pinned HF models + tokenizers (+ chat templates) to the shared cache (LOGIN node).

Reads config/local_models.yaml; snapshot_download(repo_id, revision=<pinned commit>) into
$HF_HOME (from config/paths.yaml); records the resolved commit, file list and size in
outputs/model_cache_manifest.json. Supports HF_TOKEN for gated models. Refuses to run on a
compute node (hostname pattern in config/paths.yaml).

  python scripts/online/cache_models.py [--model KEY ...] [--allow-unpinned]
"""
from __future__ import annotations

import argparse
import os
import re
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from advice_audit.common.config import OUTPUTS_DIR, is_todo, load_local_models, load_paths  # noqa: E402
from advice_audit.common.io import utcnow, write_json  # noqa: E402

IGNORE = ["original/*", "*.pth", "*.gguf", "*.onnx", "*.msgpack", "*.h5"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", help="model key(s); default all")
    ap.add_argument("--allow-unpinned", action="store_true", help="resolve 'main' when revision is TODO and print the hash to pin")
    a = ap.parse_args()
    paths = load_paths()["cluster"]
    pat = paths.get("compute_hostname_pattern")
    if pat and re.search(pat, socket.gethostname()):
        sys.exit(f"refusing to run on a compute node ({socket.gethostname()}); use the login node")
    hf_home = os.environ.get("HF_HOME") or paths.get("hf_home")
    if is_todo(hf_home):
        sys.exit("set cluster.hf_home in config/paths.yaml (or HF_HOME)")
    os.environ["HF_HOME"] = hf_home
    os.environ.pop("HF_HUB_OFFLINE", None)
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    manifest_path = OUTPUTS_DIR / "model_cache_manifest.json"
    manifest = {}
    if manifest_path.exists():
        import json

        manifest = json.loads(manifest_path.read_text())
    for m in load_local_models().models:
        if a.model and m.key not in a.model:
            continue
        if is_todo(m.repo_id):
            print(f"[skip] {m.key}: repo_id TODO")
            continue
        rev = m.revision
        if is_todo(rev):
            if not a.allow_unpinned:
                print(f"[skip] {m.key}: revision TODO (use --allow-unpinned to resolve 'main' and print the commit)")
                continue
            rev = api.model_info(m.repo_id, revision="main").sha
            print(f"[pin] {m.key}: resolved main -> {rev}  (write this into config/local_models.yaml)")
        path = Path(snapshot_download(m.repo_id, revision=rev, token=os.environ.get("HF_TOKEN"), ignore_patterns=IGNORE))
        files = sorted(str(p.relative_to(path)) for p in path.rglob("*") if p.is_file())
        size = sum((path / f).resolve().stat().st_size for f in files)
        manifest[m.key] = {"repo_id": m.repo_id, "requested_revision": m.revision, "resolved_commit": path.name,
                           "snapshot_path": str(path), "files": files, "total_bytes": size, "cached_at": utcnow()}
        print(f"[ok] {m.key}: {m.repo_id}@{path.name} {size / 1e9:.1f} GB, {len(files)} files")
    write_json(manifest_path, manifest)
    print(f"manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
