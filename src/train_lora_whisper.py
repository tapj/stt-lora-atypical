#!/usr/bin/env python3
"""
Train Whisper with LoRA adapters from a CSV manifest.

CSV columns required:
  path, transcript

Fixes:
1) PEFT may pass input_ids into Whisper.forward -> patched Whisper forward accepts/ignores input_ids.
2) Seq2SeqTrainer evaluation calls generate() with 'labels' in generation inputs -> custom trainer drops it.

Outputs:
- outputs/.../best/  (LoRA adapter weights + processor)
- outputs/.../metrics.jsonl
"""

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from datasets import Audio, load_dataset
from jiwer import cer, wer
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)


# -------------------------
# Determinism
# -------------------------

def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------
# Logging
# -------------------------

class JsonlLoggerCallback(TrainerCallback):
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        rec = {"step": int(state.global_step), **logs}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# -------------------------
# Whisper patch for PEFT input_ids leak
# -------------------------

class WhisperForConditionalGenerationPatched(WhisperForConditionalGeneration):
    def forward(
        self,
        input_features=None,
        input_ids=None,          # ignored
        inputs_embeds=None,      # ignored
        **kwargs,
    ):
        # If someone accidentally gave input_ids instead of input_features, accept it.
        if input_features is None and input_ids is not None:
            input_features = input_ids

        kwargs.pop("input_ids", None)
        kwargs.pop("inputs_embeds", None)
        return super().forward(input_features=input_features, **kwargs)


# -------------------------
# Data collator
# -------------------------

@dataclass
class DataCollatorSpeechSeq2Seq:
    processor: WhisperProcessor
    max_label_length: int = 256

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = torch.tensor([f["input_features"] for f in features], dtype=torch.float32)

        label_features = [{"input_ids": f["labels"][: self.max_label_length]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch["attention_mask"].ne(1), -100)

        return {"input_features": input_features, "labels": labels}


# -------------------------
# Custom trainer: drop labels before generate()
# -------------------------

class WhisperSeq2SeqTrainer(Seq2SeqTrainer):
    """
    Fix evaluation crash:
    transformers Seq2SeqTrainer may pass `labels` into generate().
    Whisper generate() rejects it. We remove labels in generation_inputs.
    """

    def prediction_step(
        self,
        model,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Let the parent prepare inputs, then strip labels for generation.
        # We do not want to affect the loss computation.
        if self.args.predict_with_generate and not prediction_loss_only:
            # Parent implementation builds generation_inputs from `inputs` internally.
            # We patch by removing labels early.
            if "labels" in inputs:
                inputs = dict(inputs)
                inputs.pop("labels", None)

        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)


# -------------------------
# Metrics
# -------------------------

def compute_wer_cer(pred_texts: List[str], ref_texts: List[str]) -> Dict[str, float]:
    return {"wer": float(wer(ref_texts, pred_texts)), "cer": float(cer(ref_texts, pred_texts))}


# -------------------------
# Manifest resolution
# -------------------------

def resolve_manifest_paths(manifest_csv: str) -> str:
    src = Path(manifest_csv)
    out = src.with_name(src.stem + ".resolved.csv")
    base = src.parent

    with src.open("r", encoding="utf-8") as fin, out.open("w", encoding="utf-8", newline="") as fout:
        reader = csv.DictReader(fin)
        if not reader.fieldnames or "path" not in reader.fieldnames:
            raise ValueError(f"CSV must contain column 'path'. Found: {reader.fieldnames}")
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row in reader:
            p = (row.get("path") or "").strip()
            if not p:
                continue
            if not os.path.isabs(p):
                row["path"] = str((base / p).resolve())
            writer.writerow(row)

    return str(out)


# -------------------------
# LoRA
# -------------------------

def apply_lora(model, r: int, alpha: int, dropout: float, targets: List[str]):
    cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=targets,
        bias="none",
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return model


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_csv", required=True)
    ap.add_argument("--output_dir", default="outputs/run1")
    ap.add_argument("--base_model", default="openai/whisper-small")
    ap.add_argument("--language", default="fr", help="Empty = auto language")
    ap.add_argument("--task", default="transcribe")

    ap.add_argument("--val_split", type=float, default=0.1)
    ap.add_argument("--max_audio_seconds", type=float, default=20.0)

    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup_steps", type=int, default=50)
    ap.add_argument("--epochs", type=float, default=10.0)
    ap.add_argument("--max_steps", type=int, default=-1)

    ap.add_argument("--eval_steps", type=int, default=25)
    ap.add_argument("--save_steps", type=int, default=25)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--save_total_limit", type=int, default=3)

    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lora_targets", type=str, default="q_proj,v_proj")

    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    logs_path = os.path.join(args.output_dir, "metrics.jsonl")

    resolved_csv = resolve_manifest_paths(args.manifest_csv)
    ds = load_dataset("csv", data_files=resolved_csv)["train"]

    if "path" not in ds.column_names or "transcript" not in ds.column_names:
        raise SystemExit(f"CSV must have columns path, transcript. Found: {ds.column_names}")

    ds = ds.cast_column("path", Audio(sampling_rate=16000))

    split = ds.train_test_split(test_size=float(args.val_split), seed=int(args.seed))
    train_ds, val_ds = split["train"], split["test"]

    processor = WhisperProcessor.from_pretrained(args.base_model)
    base = WhisperForConditionalGenerationPatched.from_pretrained(args.base_model)

    base.config.use_cache = False
    base.gradient_checkpointing_enable()

    lang = (args.language or "").strip()
    if lang:
        base.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=lang, task=args.task)
    else:
        base.generation_config.forced_decoder_ids = None

    targets = [t.strip() for t in args.lora_targets.split(",") if t.strip()]
    model = apply_lora(base, args.lora_r, args.lora_alpha, args.lora_dropout, targets)

    max_len = int(16000 * float(args.max_audio_seconds))

    def preprocess(batch: Dict[str, Any]) -> Dict[str, Any]:
        audio = batch["path"]
        arr = audio["array"]
        sr = int(audio["sampling_rate"])
        if arr.shape[0] > max_len:
            arr = arr[:max_len]
        feats = processor.feature_extractor(arr, sampling_rate=sr)
        return {
            "input_features": feats["input_features"][0],
            "labels": processor.tokenizer((batch.get("transcript") or "").strip()).input_ids,
        }

    train_ds = train_ds.map(preprocess, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(preprocess, remove_columns=val_ds.column_names)

    collator = DataCollatorSpeechSeq2Seq(processor=processor, max_label_length=256)

    def compute_metrics(eval_pred):
        pred_ids = eval_pred.predictions
        label_ids = eval_pred.label_ids
        label_ids = np.where(label_ids == -100, processor.tokenizer.pad_token_id, label_ids)

        pred_texts = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        ref_texts = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return compute_wer_cer(pred_texts, ref_texts)

    use_fp16 = torch.cuda.is_available()

    train_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=int(args.batch_size),
        per_device_eval_batch_size=int(args.batch_size),
        gradient_accumulation_steps=int(args.grad_accum),
        learning_rate=float(args.lr),
        warmup_steps=int(args.warmup_steps),
        num_train_epochs=float(args.epochs),
        max_steps=int(args.max_steps),
        evaluation_strategy="steps",
        eval_steps=int(args.eval_steps),
        save_strategy="steps",
        save_steps=int(args.save_steps),
        logging_steps=int(args.logging_steps),
        save_total_limit=int(args.save_total_limit),
        predict_with_generate=False,          # IMPORTANT
        fp16=use_fp16,
        report_to=[],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",    # IMPORTANT
        greater_is_better=False,
        seed=int(args.seed),
    )


    trainer = WhisperSeq2SeqTrainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        #compute_metrics=compute_metrics,
        callbacks=[JsonlLoggerCallback(logs_path)],
    )

    trainer.train()

    best_dir = os.path.join(args.output_dir, "best")
    os.makedirs(best_dir, exist_ok=True)
    trainer.model.save_pretrained(best_dir)
    processor.save_pretrained(best_dir)

    print(f"Saved adapter+processor to: {best_dir}")
    print(f"Metrics log: {logs_path}")


if __name__ == "__main__":
    main()
