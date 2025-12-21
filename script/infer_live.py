import argparse
import queue
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import sounddevice as sd
import torch
import yaml
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from src.postprocess import apply_corrections, build_initial_prompt
from src.utils import get_device
from src.vad import VADConfig, build_vad


@dataclass
class LiveState:
    buf: np.ndarray
    last_speech_t: float
    in_speech: bool


def read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--adapter_dir", type=str, required=True)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    cfg = read_yaml(args.config)
    device = torch.device(args.device) if args.device else get_device(prefer_gpu=True)

    processor = WhisperProcessor.from_pretrained(args.adapter_dir)
    model = WhisperForConditionalGeneration.from_pretrained(cfg["model"]["base_model_name"])
    model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.to(device).eval()

    prompt = build_initial_prompt(cfg.get("personalization", {}).get("phrase_list", []))
    correction_dict = cfg.get("personalization", {}).get("correction_dict", {})

    live_cfg = cfg.get("live", {})
    sr = int(live_cfg.get("sample_rate", 16000))
    chunk_ms = int(live_cfg.get("chunk_ms", 30))
    chunk_samples = int(sr * chunk_ms / 1000)

    vad_cfg_raw = live_cfg.get("vad", {}) or {}
    vad_cfg = VADConfig(
        backend=vad_cfg_raw.get("backend", "silero"),
        silero_threshold=float(vad_cfg_raw.get("silero_threshold", 0.5)),
        min_speech_ms=int(vad_cfg_raw.get("min_speech_ms", 250)),
        min_silence_ms=int(vad_cfg_raw.get("min_silence_ms", 500)),
    )
    vad = build_vad(vad_cfg, device=str(device))

    min_speech_s = vad_cfg.min_speech_ms / 1000.0
    min_silence_s = vad_cfg.min_silence_ms / 1000.0
    max_utt_s = float(vad_cfg_raw.get("max_utterance_s", 20))

    decode_cfg = cfg.get("decode", {})
    beam = int(decode_cfg.get("beam_size", 5))
    temp = float(decode_cfg.get("temperature", 0.0))

    language = cfg["model"].get("language")
    task = cfg["model"].get("task", "transcribe")
    forced_ids = processor.get_decoder_prompt_ids(language=language, task=task) if language else None

    q = queue.Queue()

    def audio_cb(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        q.put(indata.copy())

    state = LiveState(buf=np.zeros((0,), dtype=np.float32), last_speech_t=time.time(), in_speech=False)

    print("Live mode. Ctrl+C to stop.")
    print("Partial output updates per utterance. Final transcript printed after silence.\n")

    with sd.InputStream(channels=1, samplerate=sr, blocksize=chunk_samples, dtype="float32", callback=audio_cb):
        try:
            while True:
                chunk = q.get()
                x = chunk.reshape(-1).astype(np.float32)
                state.buf = np.concatenate([state.buf, x], axis=0)

                # Evaluate VAD on a rolling window (last 1.0s) for responsiveness.
                window = state.buf[-sr:] if state.buf.shape[0] > sr else state.buf
                speech = vad.has_speech(window)

                now = time.time()
                if speech:
                    state.last_speech_t = now
                    if not state.in_speech:
                        state.in_speech = True

                # If in speech and buffer too long, force utterance cut
                if state.in_speech and (state.buf.shape[0] / sr) >= max_utt_s:
                    do_flush = True
                else:
                    # End utterance after enough silence
                    do_flush = state.in_speech and ((now - state.last_speech_t) >= min_silence_s)

                # Require minimum speech duration
                if do_flush:
                    utt = state.buf.copy()
                    dur = utt.shape[0] / sr
                    state.buf = np.zeros((0,), dtype=np.float32)
                    state.in_speech = False

                    if dur < min_speech_s:
                        continue

                    # Transcribe utterance
                    inputs = processor.feature_extractor(utt, sampling_rate=sr, return_tensors="pt")
                    feats = inputs["input_features"].to(device)

                    gen_kwargs = dict(num_beams=beam, temperature=temp)
                    if forced_ids is not None:
                        gen_kwargs["forced_decoder_ids"] = forced_ids
                    if prompt:
                        prompt_ids = processor.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
                        gen_kwargs["prompt_ids"] = prompt_ids

                    with torch.no_grad():
                        pred = model.generate(feats, **gen_kwargs)

                    text = processor.tokenizer.decode(pred[0], skip_special_tokens=True).strip()
                    text = apply_corrections(text, correction_dict)

                    ts = time.strftime("%Y-%m-%d %H:%M:%S")
                    print(f"[{ts}] {text}")

        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
