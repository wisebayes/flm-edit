#!/bin/bash
# Evaluate FMLMEdit at 1, 4, and 16 steps
CKPT=${1:-/path/to/fmlm_edit_checkpoint.ckpt}

python main.py \
  mode=sample_eval \
  algo=fmlm_edit \
  data=c4_edit \
  model=small \
  eval.checkpoint_path="${CKPT}" \
  sampling.steps=[1,4,16] \
  eval.generated_samples_path=./eval_samples.json
