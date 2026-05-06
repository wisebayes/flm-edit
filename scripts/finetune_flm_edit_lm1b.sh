#!/bin/bash
# Stage 1b: Finetune FLMEdit warm-started from lm1b_flm.ckpt (BERT tokenizer, length=128)
# Usage:
#   bash scripts/finetune_flm_edit_lm1b.sh [dialogsum_edit_lm1b | cnndm_edit_lm1b]
#
# The backbone DIT layers are loaded from the pretrained checkpoint;
# cross-attention and SourceEncoder layers train from random init.

DATASET=${1:-dialogsum_edit_lm1b}   # dialogsum_edit_lm1b | cnndm_edit_lm1b
CKPT=/mnt/swordfish-pool2/ck3255/flm-edit/checkpoints/lm1b_flm.ckpt

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
  wandb.name="flm_edit_finetune_lm1b_${DATASET}"
