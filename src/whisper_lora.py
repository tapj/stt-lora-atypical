from typing import List, Optional

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def load_processor(base_model_name: str) -> WhisperProcessor:
    return WhisperProcessor.from_pretrained(base_model_name)


def load_base_model(base_model_name: str) -> WhisperForConditionalGeneration:
    return WhisperForConditionalGeneration.from_pretrained(base_model_name)


def apply_lora(
    model: WhisperForConditionalGeneration,
    r: int,
    alpha: int,
    dropout: float,
    target_modules: List[str],
) -> WhisperForConditionalGeneration:
    cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return model


def set_forced_decoder_ids(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    language: Optional[str],
    task: str,
    enable: bool,
) -> None:
    # If language is set, force Whisper to use that language token and task token.
    if not enable:
        model.generation_config.forced_decoder_ids = None
        return

    if language:
        forced = processor.get_decoder_prompt_ids(language=language, task=task)
        model.generation_config.forced_decoder_ids = forced
    else:
        # Auto language. Keep task token if desired, but simplest is no forced ids.
        model.generation_config.forced_decoder_ids = None


def maybe_enable_gradient_checkpointing(model: torch.nn.Module, enabled: bool) -> None:
    if enabled and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
