import io
import json
import os
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import torchaudio
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from huggingface_hub import snapshot_download
from peft import PeftModel
from starlette.requests import Request
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


def load_wav_bytes_to_16k_mono(data: bytes, fmt: Optional[str] = None) -> np.ndarray:
    bio = io.BytesIO(data)
    wav, sr = torchaudio.load(bio, format=fmt)
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    wav = wav.squeeze(0)
    return wav.numpy().astype(np.float32)


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
    try:
        data = await audio.read()
        fmt = guess_audio_format(audio.filename, audio.content_type)
        audio_16k = load_wav_bytes_to_16k_mono(data, fmt=fmt)
        svc = get_service(adapter_dir=adapter_dir, config_path=config_path, device=device)
        text = svc.transcribe(audio_16k, language=language, beam=beam_size, temperature=temperature)
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        return JSONResponse({"timestamp": ts, "text": text})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.post("/api/download_model")
async def api_download_model(model_id: str = Form(...)):
    """
    Download and cache a HF model. Respects HF_HOME and TRANSFORMERS_CACHE env vars.
    """
    try:
        cache_dir = os.environ.get("TRANSFORMERS_CACHE")
        snapshot_download(repo_id=model_id, cache_dir=cache_dir, token=os.environ.get("HUGGINGFACE_HUB_TOKEN"))
        return JSONResponse({"status": "ok", "model_id": model_id})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
