#!/bin/bash
# Stage 2: distill FLMEdit → FMLMEdit on maskdisc data.
TEACHER_CKPT=${1:?'Usage: $0 <teacher_ckpt> [dataset]'}
DATASET=${2:-dialogsum_edit}

python main.py \
  algo=fmlm_edit \
  algo.teacher_path="${TEACHER_CKPT}" \
  data="${DATASET}" \
  model=qwen_small \
  trainer.max_steps=50000 \
  loader.global_batch_size=128 \
  optim.lr=1e-4 \
  training.ema=0.9999 \
  wandb.name="fmlm_edit_${DATASET}_distill"
