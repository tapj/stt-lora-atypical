import io  # Needed for BytesIO when loading WAV from bytes
import csv
import json
import os
import shutil
import subprocess
import tempfile
import time
import wave
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, Form, UploadFile, Request
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
    def __init__(self, cfg_path: str, adapter_dir: Optional[str], device_str: Optional[str] = None):
        self.cfg = read_yaml(cfg_path)
        self.device = torch.device(device_str) if device_str else get_device(prefer_gpu=True)

        base_model_name = self.cfg["model"].get("base_model_name", "openai/whisper-small")

        adapter_path = Path(adapter_dir) if adapter_dir else None
        adapter_exists = adapter_path is not None and adapter_path.exists()

        processor_src = adapter_path if (adapter_exists and (adapter_path / "preprocessor_config.json").exists()) else base_model_name
        self.processor = WhisperProcessor.from_pretrained(processor_src)

        base_model = WhisperForConditionalGeneration.from_pretrained(base_model_name)
        if adapter_exists:
            self.model = PeftModel.from_pretrained(base_model, str(adapter_path))
        else:
            self.model = base_model
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
REPO_ROOT = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# Lazy init on first request
SERVICE_CACHE: Dict[str, STTService] = {}

# Load environment variables for config paths, adapter directory, and any API tokens (e.g., HUGGINGFACE_HUB_TOKEN)
load_dotenv()
DEFAULT_CONFIG_PATH = os.getenv("STT_CONFIG_PATH", "config.yaml")
DEFAULT_ADAPTER_DIR = os.getenv("STT_ADAPTER_DIR", "outputs/run1/best")
LORA_DIR = (REPO_ROOT / "outputs" / "lora").resolve()
DATA_DIR = (REPO_ROOT / "data" / "uploaded").resolve()
TRAIN_DIR = (REPO_ROOT / "outputs" / "train_runs").resolve()

for _dir in (LORA_DIR, DATA_DIR, TRAIN_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(Path(".hf")))
os.environ.setdefault("HF_DATASETS_CACHE", str(Path(".hf") / "datasets"))
os.environ.setdefault("TORCH_HOME", str(Path(".torch")))
Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["HF_DATASETS_CACHE"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["TORCH_HOME"]).mkdir(parents=True, exist_ok=True)


def get_service(adapter_dir: str, config_path: str, device: Optional[str]) -> STTService:
    key = adapter_dir or "__base__"
    if key not in SERVICE_CACHE:
        SERVICE_CACHE[key] = STTService(config_path, adapter_dir or None, device_str=device)
    return SERVICE_CACHE[key]


@app.exception_handler(Exception)
async def all_exception_handler(request: Request, exc: Exception):
    return JSONResponse({"error": str(exc)}, status_code=500)


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
    adapter_dir: Optional[str] = Form(None),
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


def _is_lora_dir(p: Path) -> bool:
    if not p.is_dir():
        return False
    if not (p / "adapter_config.json").exists():
        return False
    if (p / "adapter_model.safetensors").exists():
        return True
    if (p / "adapter_model.bin").exists():
        return True
    return False


def _list_lora_adapters() -> List[Dict[str, str]]:
    adapters: List[Dict[str, str]] = []

    roots = [LORA_DIR, REPO_ROOT / "outputs"]

    seen = set()

    for root in roots:
        if not root.exists():
            continue

        for p in root.iterdir():
            if _is_lora_dir(p):
                key = str(p.resolve())
                if key not in seen:
                    seen.add(key)
                    adapters.append({"name": p.name, "path": key})

        for p in root.glob("*/best"):
            if _is_lora_dir(p):
                key = str(p.resolve())
                if key not in seen:
                    seen.add(key)
                    adapters.append({"name": p.parent.name, "path": key})

        for p in root.glob("train_runs/*/best"):
            if _is_lora_dir(p):
                key = str(p.resolve())
                if key not in seen:
                    seen.add(key)
                    adapters.append({"name": p.parent.name, "path": key})

    adapters.sort(key=lambda x: x["name"])
    return adapters


@app.get("/api/lora/list")
async def api_lora_list():
    return JSONResponse({"lora": _list_lora_adapters()})


def _safe_extract_zip(zip_path: Path, dest_dir: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            extracted_path = dest_dir / member.filename
            if not extracted_path.resolve().is_relative_to(dest_dir.resolve()):
                raise ValueError(f"Unsafe zip path: {member.filename}")
        zf.extractall(dest_dir)


def _read_manifest_csv(manifest_path: Path, extracted_dir: Path) -> Tuple[List[Dict[str, str]], str, str]:
    rows: List[Dict[str, str]] = []
    audio_col = "path"
    text_col = "transcript"

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = {fn.lower(): fn for fn in reader.fieldnames or []}
        if "audio_path" in fieldnames:
            audio_col = fieldnames["audio_path"]
        elif "path" in fieldnames:
            audio_col = fieldnames["path"]
        if "text" in fieldnames:
            text_col = fieldnames["text"]
        elif "transcript" in fieldnames:
            text_col = fieldnames["transcript"]

        for row in reader:
            audio_rel = Path(row.get(audio_col, "")).expanduser()
            candidate = (extracted_dir / audio_rel).resolve() if not audio_rel.is_absolute() else audio_rel.resolve()
            if not candidate.exists():
                continue
            if not candidate.resolve().is_relative_to(extracted_dir.resolve()):
                continue
            text_val = row.get(text_col, "") or ""
            rows.append(
                {
                    "path": candidate.resolve().relative_to(extracted_dir.resolve()).as_posix(),
                    "transcript": text_val.strip(),
                }
            )
    return rows, "path", "transcript"


def _build_manifest_from_pairs(extracted_dir: Path) -> List[Dict[str, str]]:
    exts = {".wav", ".mp3", ".m4a", ".webm"}
    rows: List[Dict[str, str]] = []
    for audio_file in extracted_dir.rglob("*"):
        if audio_file.suffix.lower() not in exts or not audio_file.is_file():
            continue
        txt_path = audio_file.with_suffix(".txt")
        if not txt_path.exists():
            continue
        with open(txt_path, "r", encoding="utf-8") as tf:
            transcript = tf.read().strip()
        rows.append(
            {
                "path": audio_file.resolve().relative_to(extracted_dir.resolve()).as_posix(),
                "transcript": transcript,
            }
        )
    return rows


def _prepare_manifest(extracted_dir: Path) -> Tuple[Path, int, str, str]:
    manifest_path = (extracted_dir / "manifest.csv").resolve()
    rows: List[Dict[str, str]] = []
    audio_col = "path"
    text_col = "transcript"

    if manifest_path.exists():
        rows, audio_col, text_col = _read_manifest_csv(manifest_path, extracted_dir)
    else:
        rows = _build_manifest_from_pairs(extracted_dir)

    if not rows:
        raise ValueError("No audio/text pairs found in dataset.")

    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "transcript"])
        writer.writeheader()
        writer.writerows(rows)

    return manifest_path, len(rows), audio_col, text_col


@app.post("/api/lora/upload_zip")
async def api_lora_upload_zip(dataset_name: str = Form(...), zipfile: UploadFile = File(...)):
    name = dataset_name.strip()
    if not name:
        return JSONResponse({"error": "dataset_name is required"}, status_code=400)

    target_dir = (DATA_DIR / name).resolve()
    if not target_dir.is_relative_to(DATA_DIR):
        return JSONResponse({"error": "Invalid dataset name"}, status_code=400)
    target_dir.mkdir(parents=True, exist_ok=True)

    zip_path = target_dir / "dataset.zip"
    with open(zip_path, "wb") as f:
        f.write(await zipfile.read())

    extracted_dir = (target_dir / "extracted").resolve()
    if extracted_dir.exists():
        shutil.rmtree(extracted_dir)
    extracted_dir.mkdir(parents=True, exist_ok=True)

    try:
        _safe_extract_zip(zip_path, extracted_dir)
        manifest_path, n_pairs, audio_col, text_col = _prepare_manifest(extracted_dir)
        return JSONResponse(
            {
                "status": "ok",
                "dataset_dir": target_dir.as_posix(),
                "manifest": manifest_path.as_posix(),
                "n_pairs": n_pairs,
                "audio_column": audio_col,
                "text_column": text_col,
            }
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


def _update_status(status_path: Path, data: Dict[str, Any]) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _run_training_job(
    base_model: str,
    dataset_name: str,
    run_name: str,
    language: Optional[str],
    epochs: int,
    learning_rate: float,
    batch_size: int,
    status_path: Path,
    log_path: Path,
    manifest_path: Path,
    audio_column: str,
    text_column: str,
    config_template_path: str,
) -> None:
    status = {"state": "queued", "run_name": run_name, "dataset": dataset_name}
    _update_status(status_path, status)
    try:
        src_dir = REPO_ROOT / "src"
        if not src_dir.exists():
            raise RuntimeError(f"Expected repository source directory missing: {src_dir}")

        status.update({"state": "running"})
        _update_status(status_path, status)

        cfg = read_yaml(config_template_path)
        cfg.setdefault("data", {})
        cfg.setdefault("model", {})
        cfg.setdefault("lora", {})
        cfg.setdefault("train", {})

        output_dir = TRAIN_DIR / run_name
        cfg["output_dir"] = str(output_dir)

        cfg["data"] = {
            "manifest_csv": manifest_path.as_posix(),
            "audio_column": audio_column,
            "text_column": text_column,
            "val_split": cfg.get("data", {}).get("val_split", 0.1),
            "num_workers": cfg.get("data", {}).get("num_workers", 4),
            "max_audio_seconds": cfg.get("data", {}).get("max_audio_seconds", 20),
        }
        cfg["model"]["base_model_name"] = base_model or "openai/whisper-small"
        cfg["model"]["language"] = language or None
        cfg["lora"]["enabled"] = True
        cfg["train"]["num_train_epochs"] = epochs
        cfg["train"]["learning_rate"] = learning_rate

        if not torch.cuda.is_available():
            cfg["train"]["mixed_precision"] = "no"
            cfg["train"]["batch_size"] = max(1, min(int(batch_size), 2))
            cfg["train"]["gradient_accumulation_steps"] = 1
        else:
            cfg["train"]["batch_size"] = batch_size
            grad_accum = cfg["train"].get("gradient_accumulation_steps")
            cfg["train"]["gradient_accumulation_steps"] = grad_accum if grad_accum else 1

        run_dir = TRAIN_DIR / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = run_dir / "config.generated.yaml"
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f)

        env = os.environ.copy()
        env["PATH"] = f"bin{os.pathsep}" + env.get("PATH", "")
        env.setdefault("HF_HOME", str(Path(".hf")))
        env.setdefault("HF_DATASETS_CACHE", str(Path(".hf") / "datasets"))
        env.setdefault("TORCH_HOME", str(Path(".torch")))
        env["PYTHONPATH"] = "."

        with open(log_path, "w", encoding="utf-8") as log_file:
            proc = subprocess.Popen(
                ["python", "-m", "script.train", "--config", cfg_path.as_posix()],
                stdout=log_file,
                stderr=log_file,
                cwd=REPO_ROOT,
                env=env,
            )
            proc.wait()
            if proc.returncode != 0:
                raise RuntimeError(f"Training failed with code {proc.returncode}")

        best_dir = (TRAIN_DIR / run_name / "best").resolve()
        dest_dir = (LORA_DIR / run_name).resolve()
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        if best_dir.exists():
            shutil.copytree(best_dir, dest_dir)

        status.update({"state": "done", "adapter_dir": dest_dir.as_posix(), "log_path": log_path.as_posix()})
        _update_status(status_path, status)
    except Exception as e:
        status.update({"state": "error", "error": str(e)})
        _update_status(status_path, status)


@app.post("/api/lora/train")
async def api_lora_train(
    background_tasks: BackgroundTasks,
    base_model: str = Form("openai/whisper-small"),
    dataset_name: str = Form(...),
    run_name: str = Form(...),
    language: Optional[str] = Form(None),
    epochs: int = Form(3),
    lr: float = Form(1e-4),
    batch_size: int = Form(8),
):
    dataset_dir = (DATA_DIR / dataset_name).resolve()
    if not dataset_dir.is_relative_to(DATA_DIR):
        return JSONResponse({"error": "Invalid dataset name"}, status_code=400)
    extracted_dir = dataset_dir / "extracted"
    manifest_path = extracted_dir / "manifest.csv"
    if not manifest_path.exists():
        return JSONResponse({"error": "Dataset not found or manifest missing."}, status_code=400)

    status_path = TRAIN_DIR / run_name / "status.json"
    log_path = TRAIN_DIR / run_name / "train.log"

    _, _, audio_col, text_col = _prepare_manifest(extracted_dir)

    background_tasks.add_task(
        _run_training_job,
        base_model,
        dataset_name,
        run_name,
        language,
        int(epochs),
        float(lr),
        int(batch_size),
        status_path,
        log_path,
        manifest_path,
        audio_col,
        text_col,
        DEFAULT_CONFIG_PATH,
    )

    return JSONResponse(
        {
            "status": "started",
            "run_name": run_name,
            "log_path": log_path.as_posix(),
            "status_path": status_path.as_posix(),
        }
    )


@app.get("/api/lora/status")
async def api_lora_status(run_name: str, tail: int = 50):
    run_dir = TRAIN_DIR / run_name
    status_path = run_dir / "status.json"
    log_path = run_dir / "train.log"
    status_data: Dict[str, Any] = {}
    if status_path.exists():
        with open(status_path, "r", encoding="utf-8") as f:
            status_data = json.load(f)
    else:
        status_data = {"state": "unknown", "run_name": run_name}

    lines: List[str] = []
    if log_path.exists():
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()[-tail:]

    return JSONResponse({"status": status_data, "log_tail": lines})
