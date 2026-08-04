#!/bin/bash
# Tursa submission for the scalar-afe mock recovery campaign.
# ADAPT the #SBATCH header (account, partition, qos, module loads, venv
# activation) from your existing v2-campaign JADES submit scripts —
# placeholders below are marked <...>.
#
# Truth grid: 4 afe x 3 Z x 3 SFH classes x 3 S/N = 108 fits -> array job.
# Null test rows (afe_true = 0.0) are included by construction.
#
#   sbatch --array=0-107 submit_afe_tursa.sh
#
#SBATCH --job-name=afe_mock
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=<GPU_PARTITION>        # e.g. gpu
#SBATCH --account=<ACCOUNT>                # your tursa allocation
#SBATCH --qos=<QOS>
#SBATCH --output=logs/afe_mock_%A_%a.out

set -euo pipefail
mkdir -p logs

# --- environment: copy the working block from your v2-campaign scripts ---
# module load <cuda/...> <gcc/...>
# source /path/to/ceridwen-venv/bin/activate     # blackjax pin f73e12956
export SPS_HOME=${SPS_HOME:-/path/to/fsps-v4.0}  # only for provenance; FSPS
                                                 # itself is NOT needed at fit time

GRID=${GRID:-$HOME/grids/ssp_data_afe_c3k.h5}
OUTBASE=${OUTBASE:-runs/out/afe_mocks}

# --- decode SLURM_ARRAY_TASK_ID -> (afe, Z, sfh, snr) --------------------
AFES=(0.0 0.2 0.4 0.6)
ZS=(-2.2 -1.9 -1.6)
SFHS=(fastquench extended residual)
SNRS=(10 25 50)

i=${SLURM_ARRAY_TASK_ID}
afe=${AFES[$(( i % 4 ))]};        i=$(( i / 4 ))
z=${ZS[$(( i % 3 ))]};            i=$(( i / 3 ))
sfh=${SFHS[$(( i % 3 ))]};        i=$(( i / 3 ))
snr=${SNRS[$(( i % 3 ))]}

tag=$(printf "afe%+.1f_z%+.1f_%s_snr%02d" "$afe" "$z" "$sfh" "$snr")
echo "task ${SLURM_ARRAY_TASK_ID}: ${tag}"

srun python scripts_afe/run_afe_mock_recovery.py \
    --grid   "${GRID}" \
    --out    "${OUTBASE}/${tag}" \
    --afe-true "${afe}" \
    --z-true   "${z}" \
    --sfh      "${sfh}" \
    --snr      "${snr}" \
    --seed     $(( 1000 + SLURM_ARRAY_TASK_ID ))
