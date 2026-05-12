"""Edit dataset re-tokenized with GPT-2 for finetuning from pretrained FLM checkpoints.

Reads the maskdisc JSONL files (which have raw text fields `summary_gold` and
`summary_partial`), re-tokenizes with GPT-2, and reconstructs source/target pairs.

Source reconstruction strategy
-------------------------------
The Qwen-tokenized `summary_partial` text is decoded and re-tokenized with GPT-2,
but we cannot directly map Qwen mask positions to GPT-2 positions (BPE boundaries
differ).  Instead we use the `mask_ratio` metadata to re-apply random masking at
the same aggregate rate.  This faithfully reproduces the *distribution* of partial
summaries the original masked-diffusion model would have generated.

Two modes (set via `masking_strategy` arg):
  - 'random': re-apply mask_ratio-level random masking to GPT-2-tokenized gold
  - 'text':   decode `summary_partial` as-is (with <|mdm_mask|> → [MASK] token),
              re-tokenize with GPT-2 — preserves which words were originally masked
"""
import json
import re
import torch
from torch.utils.data import Dataset


# `<|mdm_mask|>` appears literally in decoded partial text
_MASK_PLACEHOLDER = '<|mdm_mask|>'


class GPT2EditDataset(Dataset):
    """
    Args:
        path: JSONL file path
        tokenizer: GPT-2 / BERT tokenizer (from transformers)
        max_length: pad/truncate summary sequences to this length
        masking_strategy: 'random' | 'text'
            'random' — ignore pre-existing partial; independently mask the
                       GPT-2-tokenized gold at `default_mask_rate`
            'text'   — decode summary_partial, replace <|mdm_mask|> with the
                       tokenizer mask token, re-tokenize with GPT-2
        default_mask_rate: used only when masking_strategy='random' and the
                           sample has no `mask_ratio` field
        use_context: if True, tokenize the source document and return
                     'context_ids' in each batch dict
        context_key: JSONL field name for the source document
                     (e.g. 'dialogue', 'article')
        context_max_length: pad/truncate context sequences to this length
    """

    def __init__(self, path: str, tokenizer,
                 max_length: int = 128,
                 masking_strategy: str = 'random',
                 default_mask_rate: float = 0.25,
                 use_context: bool = False,
                 context_key: str = 'dialogue',
                 context_max_length: int = 256):
        assert masking_strategy in ('random', 'text'), \
            "masking_strategy must be 'random' or 'text'"
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.masking_strategy = masking_strategy
        self.default_mask_rate = default_mask_rate
        self.use_context = use_context
        self.context_key = context_key
        self.context_max_length = context_max_length

        # Ensure there's a mask token
        if tokenizer.mask_token_id is None:
            tokenizer.add_special_tokens({'mask_token': '[MASK]'})
        self.mask_token_id = tokenizer.mask_token_id

        # Pad token fallback
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id

        with open(path) as f:
            self.samples = [json.loads(line) for line in f if line.strip()]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # --- Target: tokenize clean gold summary ---
        tgt_ids = self._encode_text(s['summary_gold'])

        # --- Source: create partial ---
        if self.masking_strategy == 'random':
            mask_rate = 1.0 - float(s.get('fill_frac', 1.0 - self.default_mask_rate))
            src_ids = self._apply_random_mask(tgt_ids.tolist(), mask_rate)
        else:  # 'text'
            partial_text = s.get('summary_partial', s['summary_gold'])
            # Replace <|mdm_mask|> with the tokenizer's mask token string
            mask_str = self.tokenizer.mask_token  # e.g. '[MASK]'
            partial_text = partial_text.replace(_MASK_PLACEHOLDER, mask_str)
            src_ids = self._encode_text(partial_text)

        # Attention mask: 1 at real content positions in target (before padding)
        real_len = min(len(self._raw_encode(s['summary_gold'])), self.max_length)
        attn_mask = torch.zeros(self.max_length, dtype=torch.long)
        attn_mask[:real_len] = 1

        item = {
            'source_ids': src_ids,
            'target_ids': tgt_ids,
            'attention_mask': attn_mask,
        }

        if self.use_context:
            doc_text = s.get(self.context_key, '')
            item['context_ids'] = self._encode_context(doc_text)

        return item

    # ------------------------------------------------------------------

    def _encode_context(self, text: str):
        ids = self.tokenizer.encode(
            text,
            add_special_tokens=True,
            max_length=self.context_max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )[0]
        return ids

    def _raw_encode(self, text: str):
        return self.tokenizer.encode(text, add_special_tokens=True)

    def _encode_text(self, text: str):
        ids = self.tokenizer.encode(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )[0]
        return ids

    def _apply_random_mask(self, ids: list, mask_rate: float):
        """Randomly replace `mask_rate` fraction of tokens with mask_token_id."""
        ids = ids[:self.max_length]
        result = ids[:]
        for i in range(len(ids)):
            # Don't mask BOS/EOS/PAD
            if ids[i] in (self.tokenizer.bos_token_id,
                          self.tokenizer.eos_token_id,
                          self.pad_token_id):
                continue
            if torch.rand(1).item() < mask_rate:
                result[i] = self.mask_token_id
        # Pad
        if len(result) < self.max_length:
            result += [self.pad_token_id] * (self.max_length - len(result))
        return torch.tensor(result, dtype=torch.long)


if __name__ == '__main__':
    import sys
    from transformers import AutoTokenizer

    path = sys.argv[1] if len(sys.argv) > 1 else \
        '/mnt/swordfish-pool2/ck3255/flm-edit/data/dialogsum_train_maskdisc_partial_12460.jsonl'

    tok = AutoTokenizer.from_pretrained('gpt2')
    ds = GPT2EditDataset(path, tok, max_length=128, masking_strategy='random')
    print(f'Loaded {len(ds)} samples')
    item = ds[0]
    for k, v in item.items():
        print(f'  {k}: shape={v.shape}')
    mask_count = (item['source_ids'] == ds.mask_token_id).sum().item()
    print(f'  mask tokens in source: {mask_count} / {ds.max_length}')
    print(f'  source[:10]: {item["source_ids"][:10].tolist()}')
    print(f'  target[:10]: {item["target_ids"][:10].tolist()}')
