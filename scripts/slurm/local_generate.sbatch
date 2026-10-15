#!/usr/bin/env bash
#SBATCH --job-name=aa-generate
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --output=outputs/slurm_logs/%x-%A_%a.out
# Array job over prompt shards; partition/account/time/array are passed by submit_pipeline.sh.
# Usage: MODEL=<key> sbatch --array=0-<N-1> scripts/slurm/local_generate.sbatch
# GEN_MODULE=readout runs the readout-validation stage on the same prompts instead.
source "${ADVICE_AUDIT_ROOT:-$PWD}/scripts/slurm/common.sh"
: "${MODEL:?set MODEL=<local model key>}"
python -m advice_audit.local.${GEN_MODULE:-generate} --model "$MODEL" --shard "${SLURM_ARRAY_TASK_ID:-0}"
