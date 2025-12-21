import argparse
import json
import os
from typing import Any, Dict, Optional

import numpy as np
import torch
import torchaudio
import yaml
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from src.postprocess import apply_corrections, build_initial_prompt
from src.utils import get_device


def read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_audio_16k_mono(path: str, max_seconds: Optional[float] = None) -> np.ndarray:
    wav, sr = torchaudio.load(path)
    # mono
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    wav = wav.squeeze(0)
    if max_seconds:
        wav = wav[: int(16000 * max_seconds)]
    return wav.numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--adapter_dir", type=str, required=True, help="outputs/runX/best or outputs/runX/adapter")
    ap.add_argument("--audio", type=str, required=True)
    ap.add_argument("--out_json", type=str, default=None)
    ap.add_argument("--device", type=str, default=None, help="cpu or cuda")
    args = ap.parse_args()

    cfg = read_yaml(args.config)
    device = torch.device(args.device) if args.device else get_device(prefer_gpu=True)

    processor = WhisperProcessor.from_pretrained(args.adapter_dir)
    base_name = cfg["model"]["base_model_name"]
    model = WhisperForConditionalGeneration.from_pretrained(base_name)
    model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.to(device).eval()

    decode_cfg = cfg.get("decode", {})
    beam = int(decode_cfg.get("beam_size", 5))
    temp = float(decode_cfg.get("temperature", 0.0))

    prompt = build_initial_prompt(cfg.get("personalization", {}).get("phrase_list", []))

    audio = load_audio_16k_mono(args.audio, max_seconds=cfg["data"].get("max_audio_seconds"))
    inputs = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
    input_features = inputs["input_features"].to(device)

    gen_kwargs = dict(
        num_beams=beam,
        temperature=temp,
    )

    # Language control
    language = cfg["model"].get("language")
    task = cfg["model"].get("task", "transcribe")
    if language:
        forced = processor.get_decoder_prompt_ids(language=language, task=task)
        gen_kwargs["forced_decoder_ids"] = forced

    if prompt:
        prompt_ids = processor.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        gen_kwargs["prompt_ids"] = prompt_ids

    with torch.no_grad():
        pred_ids = model.generate(input_features, **gen_kwargs)

    text = processor.tokenizer.decode(pred_ids[0], skip_special_tokens=True).strip()
    text = apply_corrections(text, cfg.get("personalization", {}).get("correction_dict", {}))

    result = {"audio": args.audio, "text": text}
    print(text)

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
