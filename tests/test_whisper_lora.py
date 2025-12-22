import torch

from src.whisper_lora import (
    maybe_enable_gradient_checkpointing,
    set_forced_decoder_ids,
)
from tests.conftest import DummyModel, DummyProcessor


def test_set_forced_decoder_ids_disable():
    model = DummyModel()
    processor = DummyProcessor()
    set_forced_decoder_ids(model, processor, language="en", task="transcribe", enable=False)
    assert model.generation_config.forced_decoder_ids is None


def test_set_forced_decoder_ids_enable_language():
    model = DummyModel()
    processor = DummyProcessor()
    set_forced_decoder_ids(model, processor, language="en", task="transcribe", enable=True)
    assert model.generation_config.forced_decoder_ids == [[0, 1]]


def test_maybe_enable_gradient_checkpointing():
    model = DummyModel()
    maybe_enable_gradient_checkpointing(model, True)
    assert model.gradient_checkpointing is True
