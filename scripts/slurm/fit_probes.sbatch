#!/usr/bin/env bash
#SBATCH --job-name=aa-probes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --output=outputs/slurm_logs/%x-%j.out
# Probes + directions (g, s, p). GPU speeds up the logistic fits; CPU works too (drop --gres).
# Usage: MODEL=<key> sbatch scripts/slurm/fit_probes.sbatch
source "${ADVICE_AUDIT_ROOT:-$PWD}/scripts/slurm/common.sh"
: "${MODEL:?set MODEL=<local model key>}"
python -m advice_audit.local.probes --model "$MODEL"
