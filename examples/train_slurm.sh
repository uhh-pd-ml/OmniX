#!/bin/bash
# Training of OmniX on one node with 8 GPUs. Adapt the SBATCH lines and the environment to your
# cluster.
#
#   trunk:    sbatch examples/train_slurm.sh
#   resumed:  sbatch examples/train_slurm.sh resume_ckpt_path=<run>/checkpoints/last.ckpt
#   anneal:   sbatch examples/train_slurm.sh \
#                 resume_ckpt_path=<trunk run>/checkpoints/step_step=1376256.ckpt \
#                 model.schedule.decay_start=1376256 model.schedule.total_steps=1654784 \
#                 trainer.max_steps=1654784
#SBATCH --job-name=omnix
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-gpu=8
#SBATCH --time=24:00:00
set -euo pipefail

humite-train +experiment=omnix trainer.devices=8 trainer.strategy=ddp "$@"
