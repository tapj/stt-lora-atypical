import os
import sys
from types import SimpleNamespace

import numpy as np
import torch


# Ensure project root is importable when tests are run from repository root.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class DummyTokenizer:
    def __init__(self):
        self.pad_token_id = 0

    def __call__(self, text, return_tensors=None):
        _ = text
        ids = [1, 2, 3]
        if return_tensors == "pt":
            return SimpleNamespace(input_ids=torch.tensor([ids], dtype=torch.long))
        return SimpleNamespace(input_ids=ids)

    def decode(self, ids, skip_special_tokens=True):
        _ = (ids, skip_special_tokens)
        return "decoded-text"

    def batch_decode(self, batch_ids, skip_special_tokens=True):
        return [self.decode(ids, skip_special_tokens=skip_special_tokens) for ids in batch_ids]

    def pad(self, inputs, return_tensors=None):
        _ = return_tensors
        max_len = max(len(i["input_ids"]) for i in inputs)
        padded = []
        mask = []
        for i in inputs:
            ids = list(i["input_ids"])
            ids = ids[:max_len]
            ids = ids + [0] * (max_len - len(ids))
            padded.append(ids)
            mask.append([1 if v != 0 else 0 for v in ids])
        return {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }


class DummyFeatureExtractor:
    def __call__(self, audio, sampling_rate, return_tensors=None):
        _ = (audio, sampling_rate)
        feats = np.zeros((1, 2, 2), dtype=np.float32)
        out = {"input_features": torch.tensor(feats)}
        return out

    def pad(self, inputs, return_tensors=None):
        _ = return_tensors
        stacked = torch.stack([torch.tensor(i["input_features"]) for i in inputs])
        return {"input_features": stacked}


class DummyProcessor:
    def __init__(self):
        self.feature_extractor = DummyFeatureExtractor()
        self.tokenizer = DummyTokenizer()

    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        return cls()

    def get_decoder_prompt_ids(self, language=None, task=None):
        _ = (language, task)
        return [[0, 1]]


class DummyModel:
    def __init__(self):
        self.generation_config = SimpleNamespace(forced_decoder_ids=None)
        self.saved_path = None
        self.gradient_checkpointing = False

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def generate(self, feats, **kwargs):
        _ = feats
        self.last_kwargs = kwargs
        return torch.tensor([[4, 5, 6]], dtype=torch.long)

    def save_pretrained(self, path):
        self.saved_path = path

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing = True

    def print_trainable_parameters(self):
        return None

    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        return cls()
