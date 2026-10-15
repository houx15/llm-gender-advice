#!/usr/bin/env bash
# Chain the Part B Slurm jobs with --dependency=afterok (spec §6.3).
#
#   bash scripts/slurm/submit_pipeline.sh <model_key> [pipeline]
#
# pipelines:
#   behavior   Step 0: generation + readout validation (array jobs)
#   repr       B3: extract + concepts -> probes/directions                  (needs APPROVED: B3)
#   interv     B4: patching -> aggregate -> steering + erasure -> aggregate (needs APPROVED: B4)
#   all        repr then interv
#
# Prerequisites (login node): `advice-audit local plan --model K --stage <stage>` for each stage,
# and `python scripts/online/verify_offline_ready.py --model K` must have passed.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
MODEL="${1:?usage: submit_pipeline.sh <model_key> [behavior|repr|interv|all]}"
PIPE="${2:-all}"

py() { python3 - "$@"; }
slurm_opt() { py "$1" <<'PY'
import sys, yaml
s = yaml.safe_load(open("config/paths.yaml"))["slurm"]
k = sys.argv[1]
v = s["time"].get(k.split(".", 1)[1]) if k.startswith("time.") else s.get(k)
print(v or "")
PY
}
OUT="${ADVICE_AUDIT_OUTPUTS:-$REPO/outputs}"
READY="$OUT/offline_ready.json"
python3 -c "import json,sys; r=json.load(open('$READY')); sys.exit(0 if r.get('ok') or r['models'].get('$MODEL',{}).get('ok') else 1)" \
  || { echo "verify_offline_ready has not passed for $MODEL ($READY)"; exit 1; }

PART="$(slurm_opt partition)"; ACCT="$(slurm_opt account)"
BASE=(--export=ALL,MODEL="$MODEL",ADVICE_AUDIT_ROOT="$REPO" --output="$OUT/slurm_logs/%x-%A_%a.out")
[[ -n "$PART" && "$PART" != TODO* ]] && BASE+=(--partition="$PART")
[[ -n "$ACCT" && "$ACCT" != TODO* ]] && BASE+=(--account="$ACCT")
mkdir -p "$OUT/slurm_logs"

nshards() { python3 -c "import json; print(json.load(open('$OUT/local/$MODEL/$1/plan/plan.json'))['n_shards'])"; }
approved() { grep -Eq "^[[:space:]]*>?[[:space:]]*APPROVED: $1" QUESTIONS.md || { echo "Phase $1 needs 'APPROVED: $1' in QUESTIONS.md"; exit 1; }; }

submit() {  # submit <script> <time-key> <dependency-or-empty> [extra sbatch args...]
  local script=$1 tkey=$2 dep=$3; shift 3
  local args=("${BASE[@]}" --time="$(slurm_opt "time.$tkey")" --parsable "$@")
  [[ -n "$dep" ]] && args+=(--dependency="afterok:$dep")
  sbatch "${args[@]}" "scripts/slurm/$script"
}
array() { echo "--array=0-$(( $(nshards "$1") - 1 ))"; }

case "$PIPE" in
  behavior)
    J1=$(submit local_generate.sbatch behavior "" "$(array behavior)")
    J2=$(submit local_generate.sbatch behavior "" "$(array readout)" --export=ALL,MODEL="$MODEL",GEN_MODULE=readout,ADVICE_AUDIT_ROOT="$REPO")
    echo "behavior: generate=$J1 readout=$J2" ;;
  repr|interv|all)
    if [[ "$PIPE" != interv ]]; then
      approved B3
      JE=$(submit extract_activations.sbatch extract "" "$(array extract)")
      JC=$(submit extract_activations.sbatch extract "" "$(array concepts)" --export=ALL,MODEL="$MODEL",STAGE=concepts,ADVICE_AUDIT_ROOT="$REPO")
      JP=$(submit fit_probes.sbatch probes "$JE:$JC")
      echo "repr: extract=$JE concepts=$JC probes=$JP"
      DEP="$JP"
    else
      DEP=""
    fi
    if [[ "$PIPE" != repr ]]; then
      approved B4
      JPA=$(submit patching.sbatch patching "$DEP" "$(array patching)")
      JPG=$(submit aggregate.sbatch probes "$JPA" --export=ALL,MODEL="$MODEL",STAGE=patching,ADVICE_AUDIT_ROOT="$REPO")
      JS=$(submit steering.sbatch steering "$JPG" "$(array steering)")
      JR=$(submit erasure.sbatch erasure "$JPG" "$(array erasure)")
      JRG=$(submit aggregate.sbatch probes "$JR" --export=ALL,MODEL="$MODEL",STAGE=erasure,ADVICE_AUDIT_ROOT="$REPO")
      echo "interv: patching=$JPA agg=$JPG steering=$JS erasure=$JR erasure_agg=$JRG"
    fi ;;
  *) echo "unknown pipeline $PIPE"; exit 1 ;;
esac
