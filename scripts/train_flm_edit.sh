#!/bin/bash
# Stage 1: Train FLMEdit (single-time denoiser)
python main.py \
  algo=flm_edit \
  data=c4_edit \
  model=small \
  trainer.max_steps=500000 \
  loader.global_batch_size=256 \
  optim.lr=3e-4 \
  training.ema=0.9999 \
  wandb.name=flm_edit_small
