"""Inference script for FLMEditFinetune checkpoints.

Loads a trained checkpoint and runs generation on a JSONL test file,
decoding partial (dirty) summaries toward the gold summary.

Usage
-----
# Without context (no-ctx checkpoint):
python infer.py \
    --ckpt outputs/dialogsum_edit_lm1b/2026.05.11/163622/checkpoints/best_nll.ckpt \
    --data data/dialogsum_val_maskdisc_partial_500.jsonl \
    --tokenizer bert-base-uncased \
    --device cuda:0 \
    --num_steps 16 \
    --n 20

# With context (ctx checkpoint):
python infer.py \
    --ckpt outputs/dialogsum_edit_lm1b_ctx/YYYY.MM.DD/HHMMSS/checkpoints/best_nll.ckpt \
    --data data/dialogsum_val_maskdisc_partial_500.jsonl \
    --tokenizer bert-base-uncased \
    --use_context \
    --device cuda:0 \
    --num_steps 16 \
    --n 20
"""
import argparse
import json
import sys
import textwrap

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from omegaconf import OmegaConf

# ── helpers ──────────────────────────────────────────────────────────────────

def _pad_or_truncate(ids, max_length, pad_value):
    ids = ids[:max_length]
    if len(ids) < max_length:
        ids = ids + [pad_value] * (max_length - len(ids))
    return ids


def load_checkpoint(ckpt_path, device):
    """Load Lightning checkpoint and return (model, config, tokenizer)."""
    import omegaconf
    try:
        from omegaconf._internal.utils import Metadata
        torch.serialization.add_safe_globals([omegaconf.DictConfig, Metadata])
    except Exception:
        pass

    raw = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    config = OmegaConf.create(raw['hyper_parameters']['config'])

    # Tokenizer
    tok_name = config.data.get('tokenizer_name_or_path', 'bert-base-uncased')
    cache_dir = config.data.get('cache_dir', None)
    tokenizer = AutoTokenizer.from_pretrained(tok_name, cache_dir=cache_dir)
    if tokenizer.mask_token_id is None:
        tokenizer.add_special_tokens({'mask_token': '[MASK]'})
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    vocab_size = max(tokenizer.vocab_size, len(tokenizer))

    # Build model
    import algo_edit
    model_cls = algo_edit.FLMEditFinetune
    model = model_cls(config, tokenizer)
    model.load_state_dict(raw['state_dict'], strict=False)
    model.eval().to(device)
    return model, config, tokenizer


def encode_text(text, tokenizer, max_length, pad_id):
    ids = tokenizer.encode(text, add_special_tokens=True,
                           max_length=max_length, truncation=True)
    return _pad_or_truncate(ids, max_length, pad_id)


def run_inference(model, tokenizer, samples, max_length, num_steps,
                  use_context, ctx_max_length, device, batch_size=8):
    """Run generation on a list of sample dicts, return list of result dicts."""
    pad_id  = tokenizer.pad_token_id
    mask_id = tokenizer.mask_token_id
    results = []

    for i in range(0, len(samples), batch_size):
        batch_samples = samples[i:i + batch_size]

        src_list = []
        tgt_list = []
        ctx_list = []
        for s in batch_samples:
            tgt_ids = encode_text(s['summary_gold'], tokenizer, max_length, pad_id)
            # Build partial/dirty source: use pre-masked partial if available,
            # otherwise apply ~25% random masking to gold
            if 'summary_partial' in s:
                partial_text = s['summary_partial'].replace('<|mdm_mask|>',
                                                            tokenizer.mask_token)
                src_ids = encode_text(partial_text, tokenizer, max_length, mask_id)
            else:
                src_ids = list(tgt_ids)
                for j, tok in enumerate(src_ids):
                    if tok not in (tokenizer.bos_token_id, tokenizer.eos_token_id, pad_id):
                        if torch.rand(1).item() < 0.25:
                            src_ids[j] = mask_id
            src_list.append(src_ids)
            tgt_list.append(tgt_ids)

            if use_context and 'dialogue' in s:
                ctx_ids = encode_text(s['dialogue'], tokenizer, ctx_max_length, pad_id)
                ctx_list.append(ctx_ids)

        src_t  = torch.tensor(src_list, dtype=torch.long, device=device)
        tgt_t  = torch.tensor(tgt_list, dtype=torch.long, device=device)
        ctx_t  = (torch.tensor(ctx_list, dtype=torch.long, device=device)
                  if ctx_list else None)

        with torch.no_grad():
            pred_ids = model.generate_samples(src_t, num_steps=num_steps,
                                              context_ids=ctx_t)  # (B, L)

        for j, s in enumerate(batch_samples):
            # Find actual content length (first pad in target)
            tgt_seq = tgt_list[j]
            try:
                seq_len = tgt_seq.index(pad_id)
            except ValueError:
                seq_len = len(tgt_seq)

            src_decoded  = tokenizer.decode(src_list[j][:seq_len],  skip_special_tokens=True)
            tgt_decoded  = tokenizer.decode(tgt_seq[:seq_len],       skip_special_tokens=True)
            pred_decoded = tokenizer.decode(pred_ids[j].tolist()[:seq_len], skip_special_tokens=True)

            # Simple token-level accuracy on non-pad positions
            L = sum(1 for x in tgt_list[j] if x != pad_id)
            correct = sum(1 for a, b in zip(pred_ids[j].tolist()[:L], tgt_list[j][:L])
                          if a == b)
            acc = correct / max(L, 1)

            results.append({
                'idx': i + j,
                'dialogue':    s.get('dialogue', ''),
                'source':      src_decoded,
                'gold':        tgt_decoded,
                'prediction':  pred_decoded,
                'token_acc':   round(acc, 4),
            })

    return results


def print_results(results, n_show):
    sep = '─' * 72
    for r in results[:n_show]:
        print(sep)
        print(f"[{r['idx']}] token_acc={r['token_acc']:.3f}")
        if r['dialogue']:
            print('DIALOGUE : ' + textwrap.shorten(r['dialogue'], 120))
        print('SOURCE   : ' + r['source'])
        print('GOLD     : ' + r['gold'])
        print('PREDICTED: ' + r['prediction'])
    print(sep)

    accs = [r['token_acc'] for r in results]
    print(f"\nMean token accuracy over {len(results)} samples: "
          f"{sum(accs)/len(accs):.4f}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='FLMEdit inference')
    parser.add_argument('--ckpt',        required=True,  help='Path to .ckpt file')
    parser.add_argument('--data',        required=True,  help='JSONL data file')
    parser.add_argument('--tokenizer',   default=None,   help='Override tokenizer (default: from ckpt config)')
    parser.add_argument('--device',      default='cuda:0')
    parser.add_argument('--num_steps',   type=int, default=16, help='ODE steps (1=one-shot)')
    parser.add_argument('--n',           type=int, default=20,  help='Samples to show')
    parser.add_argument('--max_n',       type=int, default=500, help='Total samples to eval')
    parser.add_argument('--batch_size',  type=int, default=8)
    parser.add_argument('--use_context', action='store_true',  help='Pass dialogue as context_ids')
    parser.add_argument('--ctx_max_length', type=int, default=256)
    parser.add_argument('--output_jsonl', default=None, help='Save results to JSONL')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Loading checkpoint: {args.ckpt}')
    model, config, tokenizer = load_checkpoint(args.ckpt, device)

    if args.tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    max_length = config.data.get('max_length', 128)
    print(f'max_length={max_length}, num_steps={args.num_steps}, '
          f'use_context={args.use_context}')

    print(f'Loading data: {args.data}')
    with open(args.data) as f:
        samples = [json.loads(l) for l in f if l.strip()]
    samples = samples[:args.max_n]
    print(f'Evaluating {len(samples)} samples...')

    results = run_inference(
        model, tokenizer, samples,
        max_length=max_length,
        num_steps=args.num_steps,
        use_context=args.use_context,
        ctx_max_length=args.ctx_max_length,
        device=device,
        batch_size=args.batch_size,
    )

    print_results(results, args.n)

    if args.output_jsonl:
        with open(args.output_jsonl, 'w') as f:
            for r in results:
                f.write(json.dumps(r) + '\n')
        print(f'Results saved to {args.output_jsonl}')


if __name__ == '__main__':
    main()
