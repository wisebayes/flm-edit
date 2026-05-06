"""Dataset for mask-discriminator (maskdisc) edit data.

Reads JSONL files where each line contains pre-tokenized summary fields:
    {
        "summary_partial_ids": [int, ...],   # x_t: partially denoised summary
        "summary_gold_ids":    [int, ...],   # x_1: clean gold summary
        "dialogue"/"article":  str,          # optional source document context
        "summary_partial":     str,          # decoded partial (for logging)
        "summary_gold":        str,          # decoded gold (for logging)
        ...
    }

The partial summary contains mask tokens (default ID 126336 for Qwen) at
positions not yet denoised, plus model-filled tokens at other positions.
Both partial and gold have the same token count per example.

FLM-Edit training maps:
    source_ids = summary_partial_ids  (x_0 of the flow)
    target_ids = summary_gold_ids     (x_1 of the flow)
"""
import json
import torch
from torch.utils.data import Dataset


MASK_TOKEN_ID = 126336   # <|mdm_mask|> in Qwen2/2.5 tokenizers


class MaskDiscEditDataset(Dataset):
    """Loads pre-tokenized (partial, gold) summary pairs from a JSONL file.

    Args:
        path: path to JSONL file
        max_length: pad/truncate token sequences to this length
        context_key: field name for the source document (e.g. 'dialogue',
            'article'). Set to None to skip document context entirely.
        tokenizer: only needed when context_key is set (to re-tokenize the
            document text). If context_key is None, pass None.
        context_max_length: max tokens for the document context field.
        mask_token_id: the special mask token ID in the vocabulary.
    """

    def __init__(self, path: str, max_length: int = 128,
                 context_key: str = None,
                 tokenizer=None,
                 context_max_length: int = 512,
                 mask_token_id: int = MASK_TOKEN_ID):
        self.max_length = max_length
        self.context_key = context_key
        self.tokenizer = tokenizer
        self.context_max_length = context_max_length
        self.mask_token_id = mask_token_id

        with open(path) as f:
            self.samples = [json.loads(line) for line in f if line.strip()]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        src = self._pad_or_truncate(s['summary_partial_ids'], self.max_length,
                                    pad_value=self.mask_token_id)
        tgt = self._pad_or_truncate(s['summary_gold_ids'], self.max_length,
                                    pad_value=0)

        # attention_mask: 1 at all non-pad target positions
        # Pad positions in tgt are token 0 (EOS/PAD); real content starts at 0
        # Use original length to set mask correctly
        orig_len = min(len(s['summary_gold_ids']), self.max_length)
        attn_mask = torch.zeros(self.max_length, dtype=torch.long)
        attn_mask[:orig_len] = 1

        item = {
            'source_ids': src,
            'target_ids': tgt,
            'attention_mask': attn_mask,
        }

        # Optionally include document context tokens
        if self.context_key is not None and self.tokenizer is not None:
            doc_text = s.get(self.context_key, '')
            ctx = self._tokenize_text(doc_text, self.context_max_length)
            item['context_ids'] = ctx

        return item

    # ------------------------------------------------------------------

    def _pad_or_truncate(self, ids, max_len: int, pad_value: int):
        ids = ids[:max_len]
        if len(ids) < max_len:
            ids = ids + [pad_value] * (max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def _tokenize_text(self, text: str, max_len: int):
        enc = self.tokenizer(
            text,
            max_length=max_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )
        return enc['input_ids'][0]


def get_maskdisc_split_stats(path: str):
    """Quick stats scan for a maskdisc JSONL file — useful for debugging."""
    import statistics
    gold_lens, partial_lens, remaining = [], [], []
    for line in open(path):
        d = json.loads(line)
        gold_lens.append(len(d['summary_gold_ids']))
        partial_lens.append(len(d['summary_partial_ids']))
        remaining.append(d.get('num_mask_remaining', 0))
    print(f"  samples: {len(gold_lens)}")
    print(f"  gold len  — mean {statistics.mean(gold_lens):.1f}, "
          f"max {max(gold_lens)}, min {min(gold_lens)}")
    print(f"  partial len — mean {statistics.mean(partial_lens):.1f}")
    print(f"  masks remaining — mean {statistics.mean(remaining):.1f}, "
          f"max {max(remaining)}")


if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else \
        '/mnt/swordfish-pool2/ck3255/flm-edit/data/dialogsum_train_maskdisc_partial_12460.jsonl'
    print(f'Stats for {path}:')
    get_maskdisc_split_stats(path)
    ds = MaskDiscEditDataset(path, max_length=128)
    item = ds[0]
    print(f'\nSample item keys: {list(item.keys())}')
    for k, v in item.items():
        print(f'  {k}: shape={v.shape}, dtype={v.dtype}')
    print(f'  source_ids (first 10): {item["source_ids"][:10].tolist()}')
    print(f'  target_ids (first 10): {item["target_ids"][:10].tolist()}')
    mask_count = (item['source_ids'] == MASK_TOKEN_ID).sum().item()
    print(f'  mask tokens in source_ids: {mask_count}')
