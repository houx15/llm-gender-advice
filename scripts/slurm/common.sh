#!/usr/bin/env bash
# Sourced by every sbatch script: module loads, shared env activation, OFFLINE env vars.
# Compute nodes have NO network; everything comes from shared storage.
set -euo pipefail
REPO="${ADVICE_AUDIT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
cd "$REPO"
export ADVICE_AUDIT_ROOT="$REPO"

cfg() { python3 - "$1" <<'PY'
import sys, yaml
c = yaml.safe_load(open("config/paths.yaml"))["cluster"]
v = c.get(sys.argv[1])
print(" ".join(v) if isinstance(v, list) else (v or ""))
PY
}
for m in $(cfg modules); do module load "$m"; done

ENV_DIR="$(cfg env_dir)"
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"

export HF_HOME="$(cfg hf_home)"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

OUT="$(cfg outputs_dir)"
if [[ -n "$OUT" && "$OUT" != /* ]]; then OUT="$REPO/$OUT"; fi
export ADVICE_AUDIT_OUTPUTS="${OUT:-$REPO/outputs}"
mkdir -p "$ADVICE_AUDIT_OUTPUTS/slurm_logs"
echo "[common] host=$(hostname) repo=$REPO env=$ENV_DIR HF_HOME=$HF_HOME job=${SLURM_JOB_ID:-} task=${SLURM_ARRAY_TASK_ID:-}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
