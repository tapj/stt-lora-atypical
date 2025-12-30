import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml
from datasets import Audio, Dataset, DatasetDict, load_dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    WhisperProcessor,
)

from src.collator import DataCollatorSpeechSeq2Seq
from src.metrics import compute_wer_cer
from src.utils import RunPaths, mixed_precision_flags, set_seed
from src.whisper_lora import (
    apply_lora,
    load_base_model,
    load_processor,
    maybe_enable_gradient_checkpointing,
    set_forced_decoder_ids,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_data(cfg: Dict[str, Any]) -> DatasetDict:
    dcfg = cfg["data"]
    if dcfg.get("manifest_csv"):
        manifest_path = Path(dcfg["manifest_csv"]).expanduser()
        if not manifest_path.is_absolute():
            manifest_path = (REPO_ROOT / manifest_path).resolve()
        else:
            manifest_path = manifest_path.resolve()
        data_dir = manifest_path.parent
        ds = load_dataset("csv", data_files=str(manifest_path), data_dir=str(data_dir))
        ds = ds["train"]

    audio_col = dcfg.get("audio_column", "path")

    # 🔧 FIX: convert relative audio paths to absolute paths BEFORE Audio decoding
    def _make_abs_path(ex):
        ex[audio_col] = str((data_dir / ex[audio_col]).resolve())
        return ex

    ds = ds.map(_make_abs_path, num_proc=1)

    # Ensure audio is loaded and resampled to 16k
    ds = ds.cast_column(audio_col, Audio(sampling_rate=16000))

    #     ds = load_dataset("csv", data_files=str(manifest_path), data_dir=str(data_dir))
    #     ds = ds["train"]
    # elif dcfg.get("pairs_dir"):
    #     raise ValueError("pairs_dir is supported via data/prepare_manifest.py. Provide manifest_csv.")
    # else:
    #     raise ValueError("Provide data.manifest_csv or data.pairs_dir.")
    # 
    # # Ensure audio is loaded and resampled to 16k
    # audio_col = dcfg.get("audio_column", "path")
    # ds = ds.cast_column(audio_col, Audio(sampling_rate=16000))

    # Split train/val deterministically
    val_split = float(dcfg.get("val_split", 0.1))
    dd = ds.train_test_split(test_size=val_split, seed=int(cfg["seed"]))
    return DatasetDict(train=dd["train"], validation=dd["test"])


def _optional_augment(audio: np.ndarray, sr: int, cfg: Dict[str, Any], rng: np.random.RandomState) -> np.ndarray:
    acfg = (cfg.get("data", {}).get("augment", {}) or {})
    if not acfg.get("enabled", False):
        return audio

    out = audio.astype(np.float32)

    # Conservative noise mix: gaussian hiss (no external files).
    if rng.rand() < float(acfg.get("noise_mix_prob", 0.0)):
        noise = rng.randn(out.shape[0]).astype(np.float32)
        noise = noise / (np.std(noise) + 1e-6)
        snr_db = 25.0  # mild
        sig_power = np.mean(out**2) + 1e-12
        noise_power = sig_power / (10 ** (snr_db / 10.0))
        out = out + noise * np.sqrt(noise_power)

    # Small time-stretch via resampling (approx). Avoid librosa dependency.
    if rng.rand() < float(acfg.get("time_stretch_prob", 0.0)):
        lo = float(acfg.get("time_stretch_min", 0.95))
        hi = float(acfg.get("time_stretch_max", 1.05))
        rate = float(rng.uniform(lo, hi))
        new_len = int(round(out.shape[0] / rate))
        idx = np.linspace(0, out.shape[0] - 1, new_len).astype(np.float32)
        out = np.interp(idx, np.arange(out.shape[0], dtype=np.float32), out).astype(np.float32)

    # Tiny pitch shift is risky for atypical speech. Here we do a very small spectral tilt approximation: skip true pitch shift.
    # Keep the hook but no-op by default.
    # If you want real pitch shift, add librosa and keep semitone range within [-1, 1].

    return out


def _normalize_loudness(audio: np.ndarray) -> np.ndarray:
    # Simple RMS normalization to a target RMS.
    target_rms = 0.08
    rms = float(np.sqrt(np.mean(audio**2) + 1e-12))
    if rms < 1e-6:
        return audio
    gain = target_rms / rms
    return (audio * gain).astype(np.float32)


def _prepare_features(cfg: Dict[str, Any], processor: WhisperProcessor):
    dcfg = cfg["data"]
    audio_col = dcfg.get("audio_column", "path")
    text_col = dcfg.get("text_column", "transcript")
    lang_col = dcfg.get("language_column", "language")

    max_s = float(dcfg.get("max_audio_seconds", 20))
    normalize = bool(dcfg.get("normalize_loudness", False))

    seed = int(cfg["seed"])

    def fn(batch: Dict[str, Any]) -> Dict[str, Any]:
        audio = batch[audio_col]
        arr = audio["array"]
        sr = int(audio["sampling_rate"])

        # Truncate overly long audio to keep training stable.
        max_len = int(sr * max_s)
        if arr.shape[0] > max_len:
            arr = arr[:max_len]

        audio_obj = batch.get(audio_col, "")
        if isinstance(audio_obj, dict):
            audio_id = audio_obj.get("path", "")
        else:
            audio_id = audio_obj
        rng = np.random.RandomState(seed + (hash(str(audio_id)) % 10_000))
        arr = _optional_augment(arr, sr, cfg, rng)
        if normalize:
            arr = _normalize_loudness(arr)

        inputs = processor.feature_extractor(arr, sampling_rate=sr)
        batch["input_features"] = inputs["input_features"][0]

        text = (batch.get(text_col) or "").strip()
        # Optional: if per-row language exists, you can prepend hints. Keep simple.
        batch_lang = (batch.get(lang_col) or "").strip() if lang_col in batch else ""
        _ = batch_lang  # reserved for future use

        labels = processor.tokenizer(text).input_ids
        batch["labels"] = labels
        return batch

    return fn


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


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    args = ap.parse_args()

    cfg_path = Path(args.config).expanduser()
    cfg = _read_yaml(cfg_path)
    set_seed(int(cfg["seed"]))

    paths = RunPaths.from_output_dir(cfg["output_dir"])
    os.makedirs(paths.output_dir, exist_ok=True)

    processor = load_processor(cfg["model"]["base_model_name"])
    ds = _load_data(cfg)

    prep = _prepare_features(cfg, processor)
    num_workers = int(cfg["data"].get("num_workers", 4))

    ds = ds.map(prep, remove_columns=ds["train"].column_names, num_proc=num_workers)

    model = load_base_model(cfg["model"]["base_model_name"])
    if cfg["train"].get("gradient_checkpointing", False):
        maybe_enable_gradient_checkpointing(model, True)

    # Force language/task if configured
    set_forced_decoder_ids(
        model,
        processor,
        language=cfg["model"].get("language"),
        task=cfg["model"].get("task", "transcribe"),
        enable=bool(cfg["model"].get("forced_decoder_ids", True)),
    )

    # LoRA only
    if cfg["lora"].get("enabled", True):
        model = apply_lora(
            model,
            r=int(cfg["lora"]["r"]),
            alpha=int(cfg["lora"]["alpha"]),
            dropout=float(cfg["lora"]["dropout"]),
            target_modules=list(cfg["lora"]["target_modules"]),
        )

    fp16, bf16 = mixed_precision_flags(cfg["train"].get("mixed_precision", "no"))

    training_args = Seq2SeqTrainingArguments(
        output_dir=paths.output_dir,
        per_device_train_batch_size=int(cfg["train"]["batch_size"]),
        per_device_eval_batch_size=int(cfg["train"]["batch_size"]),
        gradient_accumulation_steps=int(cfg["train"]["gradient_accumulation_steps"]),
        learning_rate=float(cfg["train"]["learning_rate"]),
        warmup_steps=int(cfg["train"]["warmup_steps"]),
        num_train_epochs=float(cfg["train"]["num_train_epochs"]),
        max_steps=int(cfg["train"]["max_steps"]) if cfg["train"].get("max_steps") else -1,
        weight_decay=float(cfg["train"].get("weight_decay", 0.0)),
        logging_steps=int(cfg["train"]["logging_steps"]),
        evaluation_strategy="steps",
        eval_steps=int(cfg["train"]["eval_steps"]),
        save_strategy="steps",
        save_steps=int(cfg["train"]["save_steps"]),
        save_total_limit=int(cfg["train"]["save_total_limit"]),
        predict_with_generate=True,
        generation_max_length=int(cfg["model"].get("max_label_length", 256)),
        fp16=fp16,
        bf16=bf16,
        dataloader_num_workers=num_workers,
        max_grad_norm=float(cfg["train"].get("max_grad_norm", 1.0)),
        report_to=[],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        seed=int(cfg["seed"]),
    )

    collator = DataCollatorSpeechSeq2Seq(
        processor=processor,
        max_label_length=int(cfg["model"].get("max_label_length", 256)),
    )

    def compute_metrics(eval_pred):
        pred_ids = eval_pred.predictions
        label_ids = eval_pred.label_ids

        # Replace -100 with pad token for decoding
        label_ids = np.where(label_ids == -100, processor.tokenizer.pad_token_id, label_ids)

        pred_texts = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        ref_texts = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        m = compute_wer_cer(pred_texts, ref_texts)
        return m

    class WhisperSeq2SeqTrainer(Seq2SeqTrainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            # Whisper expects audio features, not text encoder inputs
            inputs.pop("input_ids", None)
            inputs.pop("attention_mask", None)
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

    trainer = WhisperSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[JsonlLoggerCallback(paths.logs_jsonl)],
    )

    trainer.train()

    # Save best adapter and processor
    os.makedirs(paths.best_dir, exist_ok=True)
    if hasattr(trainer.model, "save_pretrained"):
        trainer.model.save_pretrained(paths.best_dir)
    processor.save_pretrained(paths.best_dir)

    print(f"Saved best checkpoint to {paths.best_dir}")
    print(f"Metrics log: {paths.logs_jsonl}")


if __name__ == "__main__":
    main()
