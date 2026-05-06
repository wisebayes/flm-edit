"""Dataset for (source, target, instruction) editing triples.

Each line of the JSONL file must contain:
    {"source": "...", "target": "...", "instruction": "..."}  # instruction is optional
"""
import json
import torch
from torch.utils.data import Dataset


class EditDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=128,
                 instruction_max_length=64):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.instruction_max_length = instruction_max_length
        with open(path) as f:
            self.samples = [json.loads(line) for line in f if line.strip()]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        src = self._encode(s['source'], self.max_length)
        tgt = self._encode(s['target'], self.max_length)
        inst = self._encode(s.get('instruction', ''), self.instruction_max_length)
        return {
            'source_ids': src,
            'target_ids': tgt,
            'instruction_ids': inst,
            # attention_mask over target tokens (non-pad)
            'attention_mask': (tgt != self.tokenizer.pad_token_id).long(),
        }

    def _encode(self, text, max_len):
        tok = self.tokenizer
        # Some tokenizers lack a pad token; fall back to eos
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id
        ids = tok.encode(
            text,
            add_special_tokens=True,
            max_length=max_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )[0]
        return ids
