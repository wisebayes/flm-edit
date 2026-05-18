#!/bin/bash
# Stage 2: Distill FLMEdit → FMLMEdit (two-time PSD, one-step capable)
# Warm-started from a trained FLMEditFinetune checkpoint.
# Usage:
#   # Without context:
#   bash scripts/distill_fmlm_edit_lm1b.sh \
#       outputs/dialogsum_edit_lm1b/2026.05.11/163622/checkpoints/best_nll.ckpt
#
#   # With context:
#   bash scripts/distill_fmlm_edit_lm1b.sh \
#       outputs/dialogsum_edit_lm1b/2026.05.12/115608/checkpoints/best_nll.ckpt \
#       true

TEACHER_CKPT=${1:-outputs/dialogsum_edit_lm1b/2026.05.11/163622/checkpoints/best_nll.ckpt}
USE_CTX=${2:-false}

export WANDB_MODE=${WANDB_MODE:-offline}

CTX_FLAG=""
RUN_NAME="fmlm_edit_distill_lm1b"
if [ "${USE_CTX}" = "true" ]; then
  CTX_FLAG="data.use_context=true"
  RUN_NAME="fmlm_edit_distill_lm1b_ctx"
fi

python main.py \
  algo=fmlm_edit \
  algo.teacher_path="${TEACHER_CKPT}" \
  algo.initialize_student_from_teacher=true \
  data=dialogsum_edit_lm1b \
  ${CTX_FLAG} \
  model=small \
  model.length=128 \
  trainer.max_steps=50000 \
  trainer.val_check_interval=1.0 \
  loader.global_batch_size=64 \
  optim.lr=3e-5 \
  training.ema=0.9999 \
  wandb.name="${RUN_NAME}"
