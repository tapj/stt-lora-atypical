from typing import Dict, List

import numpy as np
from jiwer import cer, wer


def compute_wer_cer(pred_texts: List[str], ref_texts: List[str]) -> Dict[str, float]:
    # jiwer expects strings; it normalizes internally if you add transforms.
    w = wer(ref_texts, pred_texts)
    c = cer(ref_texts, pred_texts)
    return {"wer": float(w), "cer": float(c)}
