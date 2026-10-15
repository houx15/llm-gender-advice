#!/usr/bin/env bash
#SBATCH --job-name=aa-aggregate
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=outputs/slurm_logs/%x-%j.out
# CPU aggregation of shard outputs. Usage: MODEL=<key> STAGE=patching|erasure sbatch scripts/slurm/aggregate.sbatch
source "${ADVICE_AUDIT_ROOT:-$PWD}/scripts/slurm/common.sh"
: "${MODEL:?}"; : "${STAGE:?patching|erasure}"
python -m advice_audit.local."$STAGE" --model "$MODEL" --aggregate
