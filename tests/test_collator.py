import torch

from src.collator import DataCollatorSpeechSeq2Seq
from tests.conftest import DummyProcessor


def test_data_collator_pads_and_truncates_labels():
    processor = DummyProcessor()
    collator = DataCollatorSpeechSeq2Seq(processor=processor, max_label_length=2)

    features = [
        {"input_features": [[1, 2]], "labels": [9, 8, 7]},
        {"input_features": [[3, 4]], "labels": [6]},
    ]

    batch = collator(features)
    assert "input_features" in batch
    assert batch["labels"].shape == (2, 2)
    assert torch.equal(batch["labels"][0], torch.tensor([9, 8]))
