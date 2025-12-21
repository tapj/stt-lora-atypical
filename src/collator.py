from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from transformers import WhisperProcessor


@dataclass
class DataCollatorSpeechSeq2Seq:
    processor: WhisperProcessor
    max_label_length: int = 256

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        label_features = [{"input_ids": f["labels"][: self.max_label_length]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(labels_batch["attention_mask"].ne(1), -100)

        batch["labels"] = labels
        return batch
