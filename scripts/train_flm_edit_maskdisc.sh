#!/bin/bash
# Train FLMEdit on maskdisc (partial→gold) data.
# Combines both DialogSum and CNN/DM by passing the larger dataset;
# swap 'data=cnndm_edit' to use CNN/DM only.

DATASET=${1:-dialogsum_edit}   # dialogsum_edit | cnndm_edit

python main.py \
  algo=flm_edit \
  data="${DATASET}" \
  model=qwen_small \
  trainer.max_steps=100000 \
  loader.global_batch_size=128 \
  optim.lr=3e-4 \
  training.ema=0.9999 \
  wandb.name="flm_edit_${DATASET}"
