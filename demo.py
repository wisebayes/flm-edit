"""Quick CLI demo for one-step text editing with a trained FMLMEdit checkpoint.

Usage:
    python demo.py --ckpt /path/to/fmlm_edit.ckpt \
                   --source "The cat sat on the mat." \
                   --steps 1

    # Partial edit (start from partially-noisy source, preserves more of original)
    python demo.py --ckpt /path/to/fmlm_edit.ckpt \
                   --source "The cat sat on the mat." \
                   --steps 4 \
                   --start_t 0.3
"""
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

import algo_edit


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True, help='Path to FMLMEdit checkpoint')
    p.add_argument('--source', required=True, help='Source text to edit')
    p.add_argument('--steps', type=int, default=1,
                   help='Number of flow-map steps (1 = pure one-step)')
    p.add_argument('--start_t', type=float, default=0.0,
                   help='Start interpolation time (0=pure source, >0=partial edit)')
    p.add_argument('--max_length', type=int, default=128)
    p.add_argument('--tokenizer', default='gpt2',
                   help='HuggingFace tokenizer name or path')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = algo_edit.FMLMEdit.load_from_checkpoint(
        args.ckpt, weights_only=False)
    model.eval().to(args.device)
    if model.ema is not None:
        model.ema.copy_to(model._get_parameters())

    src_ids = tokenizer.encode(
        args.source,
        add_special_tokens=True,
        max_length=args.max_length,
        truncation=True,
        padding='max_length',
        return_tensors='pt',
    ).to(args.device)   # (1, L)

    with torch.no_grad():
        if args.start_t > 0.0:
            # Partial edit: start from X_{0, start_t}(x_src) rather than x_src
            V = model.vocab_size
            z = F.one_hot(src_ids, V).float()
            tau_s = torch.tensor([0.0], device=args.device)
            tau_t = torch.tensor([args.start_t], device=args.device)
            log_D = model.forward(z, src_ids, tau_s, tau_t)
            D = log_D.exp()
            t_s = model._tau_to_t(tau_s).view(-1, 1, 1)
            t_t = model._tau_to_t(tau_t).view(-1, 1, 1)
            z = ((1 - t_t) / (1 - t_s + 1e-5)) * z \
              + ((t_t - t_s) / (1 - t_s + 1e-5)) * D
            out_ids = model.generate_samples(
                src_ids, num_steps=args.steps)
        else:
            out_ids = model.generate_samples(src_ids, num_steps=args.steps)

    result = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    print(f'Source : {args.source}')
    print(f'Edited : {result}')


if __name__ == '__main__':
    main()
