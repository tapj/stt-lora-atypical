import argparse
from pathlib import Path

import numpy as np
import torch
import torchaudio
import yaml
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from src.metrics import compute_wer_cer
from src.postprocess import apply_corrections, build_initial_prompt
from src.utils import get_device

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}


def read_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_audio_16k_mono(path: str) -> np.ndarray:
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav.squeeze(0).numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter_dir", type=str, required=True)
    ap.add_argument("--pairs_dir", type=str, required=True)
    ap.add_argument("--config", type=str, default="config.yaml")
    args = ap.parse_args()

    cfg = read_yaml(args.config)
    device = get_device(prefer_gpu=True)

    processor = WhisperProcessor.from_pretrained(args.adapter_dir)
    base = cfg["model"]["base_model_name"]
    model = WhisperForConditionalGeneration.from_pretrained(base)
    model = PeftModel.from_pretrained(model, args.adapter_dir).to(device).eval()

    prompt = build_initial_prompt(cfg.get("personalization", {}).get("phrase_list", []))
    correction_dict = cfg.get("personalization", {}).get("correction_dict", {})

    language = cfg["model"].get("language")
    task = cfg["model"].get("task", "transcribe")
    forced = processor.get_decoder_prompt_ids(language=language, task=task) if language else None

    pred_texts = []
    ref_texts = []

    pairs_dir = Path(args.pairs_dir)
    for a in sorted(pairs_dir.rglob("*")):
        if not a.is_file() or a.suffix.lower() not in AUDIO_EXTS:
            continue
        t = a.with_suffix(".txt")
        if not t.exists():
            continue
        ref = t.read_text(encoding="utf-8").strip()
        if not ref:
            continue

        audio = load_audio_16k_mono(str(a))
        feats = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt")["input_features"].to(device)

        gen_kwargs = dict(num_beams=int(cfg["decode"].get("beam_size", 5)), temperature=float(cfg["decode"].get("temperature", 0.0)))
        if forced is not None:
            gen_kwargs["forced_decoder_ids"] = forced
        if prompt:
            gen_kwargs["prompt_ids"] = processor.tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        with torch.no_grad():
            pred = model.generate(feats, **gen_kwargs)

        hyp = processor.tokenizer.decode(pred[0], skip_special_tokens=True).strip()
        hyp = apply_corrections(hyp, correction_dict)

        pred_texts.append(hyp)
        ref_texts.append(ref)

    m = compute_wer_cer(pred_texts, ref_texts)
    print(f"Samples: {len(ref_texts)}")
    print(f"WER: {m['wer']:.4f}")
    print(f"CER: {m['cer']:.4f}")


if __name__ == "__main__":
    main()
