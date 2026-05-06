#!/bin/bash
# Stage 2: Distill FLMEdit → FMLMEdit (two-time flow map, one-step capable)
# Set TEACHER_CKPT to the FLMEdit checkpoint produced by stage 1.
TEACHER_CKPT=${1:-/path/to/flm_edit_checkpoint.ckpt}

python main.py \
  algo=fmlm_edit \
  algo.teacher_path="${TEACHER_CKPT}" \
  data=c4_edit \
  model=small \
  trainer.max_steps=200000 \
  loader.global_batch_size=256 \
  optim.lr=1e-4 \
  training.ema=0.9999 \
  wandb.name=fmlm_edit_small_distill
