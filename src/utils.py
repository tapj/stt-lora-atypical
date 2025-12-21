import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinism: note that full determinism can reduce performance on GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def mixed_precision_flags(mode: str) -> Tuple[bool, bool]:
    mode = (mode or "no").lower()
    if mode == "fp16":
        return True, False
    if mode == "bf16":
        return False, True
    return False, False


@dataclass
class RunPaths:
    output_dir: str
    adapter_dir: str
    best_dir: str
    logs_jsonl: str

    @staticmethod
    def from_output_dir(output_dir: str) -> "RunPaths":
        adapter_dir = os.path.join(output_dir, "adapter")
        best_dir = os.path.join(output_dir, "best")
        logs_jsonl = os.path.join(output_dir, "metrics.jsonl")
        return RunPaths(output_dir, adapter_dir, best_dir, logs_jsonl)
