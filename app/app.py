import io  # Needed for BytesIO when loading WAV from bytes
import json
import os
import subprocess
import tempfile
import time
import wave
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from src.postprocess import apply_corrections, build_initial_prompt
from src.utils import get_device


def read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def guess_audio_format(filename: Optional[str], content_type: Optional[str]) -> Optional[str]:
    if content_type:
        main_type = content_type.split(";", maxsplit=1)[0].strip().lower()
        if "/" in main_type:
            subtype = main_type.split("/", maxsplit=1)[1]
            if subtype:
                return subtype

    if filename:
        ext = os.path.splitext(filename)[1].lower().lstrip(".")
        if ext:
            return ext

    return None


def normalize_to_wav16k_mono(upload_bytes: bytes, filename: str) -> bytes:
    """
    Return WAV bytes (mono, 16 kHz, pcm_s16le) by invoking ffmpeg.
    """
    suffix = Path(filename or "").suffix.lower() or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as src:
        src.write(upload_bytes)
        src_path = src.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as dst:
        dst_path = dst.name

    env = os.environ.copy()
    env["PATH"] = f"bin{os.pathsep}" + env.get("PATH", "")

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        src_path,
        "-ac",
        "1",
        "-ar",
        "16000",
        "-acodec",
        "pcm_s16le",
        "-f",
        "wav",
        dst_path,
    ]
    try:
        subprocess.run(cmd, check=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        with open(dst_path, "rb") as f:
            wav_bytes = f.read()
        return wav_bytes
    finally:
        for p in (src_path, dst_path):
            try:
                os.remove(p)
            except OSError:
                pass


def load_wav_bytes_to_float32_mono_16k(data: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(data), "rb") as wf:
        if wf.getnchannels() != 1:
            raise ValueError("Expected mono WAV input")
        if wf.getframerate() != 16000:
            raise ValueError("Expected 16 kHz WAV input")
        if wf.getsampwidth() != 2:
            raise ValueError("Expected 16-bit PCM WAV input")
        frames = wf.readframes(wf.getnframes())
    wav = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return wav


class STTService:
    def __init__(self, cfg_path: str, adapter_dir: str, device_str: Optional[str] = None):
        self.cfg = read_yaml(cfg_path)
        self.device = torch.device(device_str) if device_str else get_device(prefer_gpu=True)

        self.processor = WhisperProcessor.from_pretrained(adapter_dir)
        base = self.cfg["model"]["base_model_name"]
        model = WhisperForConditionalGeneration.from_pretrained(base)
        self.model = PeftModel.from_pretrained(model, adapter_dir)
        self.model.to(self.device).eval()

        self.prompt = build_initial_prompt(self.cfg.get("personalization", {}).get("phrase_list", []))
        self.corrections = self.cfg.get("personalization", {}).get("correction_dict", {})

    def transcribe(self, audio_16k: np.ndarray, language: Optional[str], beam: int, temperature: float) -> str:
        inputs = self.processor.feature_extractor(audio_16k, sampling_rate=16000, return_tensors="pt")
        feats = inputs["input_features"].to(self.device)

        gen_kwargs = dict(num_beams=int(beam), temperature=float(temperature))

        task = self.cfg["model"].get("task", "transcribe")
        if language:
            forced = self.processor.get_decoder_prompt_ids(language=language, task=task)
            gen_kwargs["forced_decoder_ids"] = forced

        if self.prompt:
            prompt_ids = self.processor.tokenizer(self.prompt, return_tensors="pt").input_ids.to(self.device)
            gen_kwargs["prompt_ids"] = prompt_ids

        with torch.no_grad():
            pred = self.model.generate(feats, **gen_kwargs)

        text = self.processor.tokenizer.decode(pred[0], skip_special_tokens=True).strip()
        text = apply_corrections(text, self.corrections)
        return text


app = FastAPI()

BASE_DIR = os.path.dirname(__file__)
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# Lazy init on first request
SERVICE: Optional[STTService] = None

# Load environment variables for config paths, adapter directory, and any API tokens (e.g., HUGGINGFACE_HUB_TOKEN)
load_dotenv()
DEFAULT_CONFIG_PATH = os.getenv("STT_CONFIG_PATH", "config.yaml")
DEFAULT_ADAPTER_DIR = os.getenv("STT_ADAPTER_DIR", "outputs/run1/best")


def get_service(adapter_dir: str, config_path: str, device: Optional[str]) -> STTService:
    global SERVICE
    if SERVICE is None:
        SERVICE = STTService(config_path, adapter_dir, device_str=device)
    return SERVICE


@app.exception_handler(Exception)
async def all_exception_handler(request: Request, exc: Exception):
    return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "adapter_dir_default": DEFAULT_ADAPTER_DIR},
    )


@app.post("/api/transcribe")
async def api_transcribe(
    audio: UploadFile = File(...),
    adapter_dir: str = Form(...),
    config_path: str = Form(DEFAULT_CONFIG_PATH),
    device: Optional[str] = Form(None),  # "cpu" or "cuda"
    language: Optional[str] = Form(None),
    beam_size: int = Form(5),
    temperature: float = Form(0.0),
):
    upload = audio
    try:
        if not hasattr(upload, "read"):
            raise TypeError(f"Unsupported audio type: {type(upload)}")

        audio_bytes = await upload.read()
        filename = upload.filename or "audio.webm"
        wav_bytes = normalize_to_wav16k_mono(audio_bytes, filename)
        audio_16k = load_wav_bytes_to_float32_mono_16k(wav_bytes)

        svc = get_service(adapter_dir=adapter_dir, config_path=config_path, device=device)
        text = svc.transcribe(audio_16k, language=language, beam=beam_size, temperature=temperature)
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        return JSONResponse({"timestamp": ts, "text": text})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    finally:
        for path in (input_path, output_path):
            if path and path.exists():
                path.unlink()


@app.post("/api/download_model")
async def api_download_model(model_id: str = Form(...)):
    """
    Download and cache a HF model. Respects HF_HOME env var.
    """
    try:
        cache_dir = os.environ.get("HF_HOME")
        snapshot_download(repo_id=model_id, cache_dir=cache_dir, token=os.environ.get("HUGGINGFACE_HUB_TOKEN"))
        return JSONResponse({"status": "ok", "model_id": model_id})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
