#!/usr/bin/env bash
#SBATCH --job-name=aa-extract
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --output=outputs/slurm_logs/%x-%A_%a.out
# Array job over prompt shards; partition/account/time/array are passed by submit_pipeline.sh.
# Usage: MODEL=<key> sbatch --array=0-<N-1> scripts/slurm/extract_activations.sbatch
source "${ADVICE_AUDIT_ROOT:-$PWD}/scripts/slurm/common.sh"
: "${MODEL:?set MODEL=<local model key>}"
python -m advice_audit.local.extract --model "$MODEL" --shard "${SLURM_ARRAY_TASK_ID:-0}" --stage "${STAGE:-extract}"
