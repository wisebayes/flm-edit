#!/bin/bash
# Stage 1b: Finetune FLMEdit warm-started from owt_flm.ckpt (GPT-2 tokenizer, length=1024)
# Usage:
#   bash scripts/finetune_flm_edit_owt.sh [dialogsum_edit_owt | cnndm_edit_owt]
#
# The backbone DIT layers are loaded from the pretrained checkpoint;
# cross-attention and SourceEncoder layers train from random init.
# Note: owt_flm uses GPT-2 tokenizer (vocab 50258), max_length capped to 128 in data configs.

DATASET=${1:-dialogsum_edit_owt}   # dialogsum_edit_owt | cnndm_edit_owt
CKPT=/mnt/swordfish-pool2/ck3255/flm-edit/checkpoints/owt_flm.ckpt

python main.py \
  algo=flm_edit_finetune \
  algo.pretrained_flm_path="${CKPT}" \
  algo.pretrained_use_ema=true \
  algo.freeze_backbone=false \
  data="${DATASET}" \
  model=small \
  model.length=128 \
  trainer.max_steps=50000 \
  loader.global_batch_size=64 \
  optim.lr=1e-4 \
  training.ema=0.9999 \
  wandb.name="flm_edit_finetune_owt_${DATASET}"
