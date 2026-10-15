#!/usr/bin/env bash
# Build the shared Python environment on the LOGIN node (has internet). Compute nodes only
# activate it (scripts/slurm/common.sh). Usage:
#   bash scripts/online/setup_env.sh [--wheelhouse]
# Env overrides: CUDA_TAG (default cu124), PYTHON_VERSION (default 3.11), ENV_DIR (default from config/paths.yaml)
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

read_cfg() { python3 -c "import yaml,sys; c=yaml.safe_load(open('config/paths.yaml')); print(c['cluster'].get('$1') or '')"; }
ENV_DIR="${ENV_DIR:-$(read_cfg env_dir)}"
CUDA_TAG="${CUDA_TAG:-cu124}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
if [[ -z "$ENV_DIR" || "$ENV_DIR" == TODO* ]]; then
  echo "set cluster.env_dir in config/paths.yaml (or ENV_DIR=...)" >&2; exit 1
fi
PATTERN="$(read_cfg compute_hostname_pattern)"
if [[ -n "$PATTERN" ]] && hostname | grep -Eq "$PATTERN"; then
  echo "refusing to run on a compute node ($(hostname)); run on the login node" >&2; exit 1
fi

if command -v uv >/dev/null 2>&1; then
  uv venv --python "$PYTHON_VERSION" "$ENV_DIR"
  # shellcheck disable=SC1091
  source "$ENV_DIR/bin/activate"
  uv pip install --index-url "https://download.pytorch.org/whl/${CUDA_TAG}" torch
  uv pip install -e ".[local,analysis,api]"
  uv pip install pytest pytest-asyncio tokenizers
else
  python3 -m venv "$ENV_DIR"
  # shellcheck disable=SC1091
  source "$ENV_DIR/bin/activate"
  pip install --upgrade pip
  pip install --index-url "https://download.pytorch.org/whl/${CUDA_TAG}" torch
  pip install -e ".[local,analysis,api]"
  pip install pytest pytest-asyncio tokenizers
fi

python - <<'PY'
import torch, transformers
print("torch", torch.__version__, "cuda build", torch.version.cuda, "| transformers", transformers.__version__)
try:
    import vllm; print("vllm", vllm.__version__)
except Exception as e:
    print("WARNING: vllm not importable:", e)
PY

if [[ "${1:-}" == "--wheelhouse" ]]; then
  pip freeze --exclude-editable > "$REPO/wheelhouse.requirements.txt"
  pip wheel -r "$REPO/wheelhouse.requirements.txt" -w "$REPO/wheelhouse"
  echo "wheelhouse -> $REPO/wheelhouse"
fi
echo "environment ready: $ENV_DIR"
