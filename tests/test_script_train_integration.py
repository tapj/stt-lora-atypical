import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml
from huggingface_hub import snapshot_download


def _is_offline() -> bool:
    return os.environ.get("HF_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"


def _model_available(model_id: str) -> bool:
    model_path = Path(model_id)
    if model_path.exists() and model_path.is_dir():
        return True
    try:
        snapshot_download(repo_id=model_id, local_files_only=True)
        return True
    except Exception:
        return False


def test_script_train_with_resolved_manifest(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]

    if _is_offline():
        pytest.skip("Offline environment; skip training invocation")

    base_model = os.environ.get("STT_TEST_BASE_MODEL", "openai/whisper-tiny")
    if not _model_available(base_model):
        pytest.skip("Base model not available locally; pre-seed cache or set STT_TEST_BASE_MODEL")

    workdir = tmp_path / "work"
    workdir.mkdir(parents=True, exist_ok=True)

    pairs_src = repo_root / "data" / "example_pairs"
    pairs_dir = workdir / "example_pairs"
    shutil.copytree(pairs_src, pairs_dir)

    manifest_csv = workdir / "manifest.csv"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""

    cache_root = tmp_path / "hf_home"
    env["HF_HOME"] = str(cache_root)
    env["TORCH_HOME"] = str(tmp_path / "torch_home")
    env["HF_DATASETS_CACHE"] = str(tmp_path / "hf_datasets_cache")
    env["XDG_CACHE_HOME"] = str(tmp_path / "xdg_cache_home")

    cmd_manifest = [
        "python",
        "-m",
        "data.prepare_manifest",
        "--pairs_dir",
        str(pairs_dir),
        "--out_csv",
        str(manifest_csv),
    ]
    r_manifest = subprocess.run(cmd_manifest, cwd=str(repo_root), env=env, capture_output=True, text=True)
    assert r_manifest.returncode == 0, f"Manifest prep failed:\n{r_manifest.stdout}\n{r_manifest.stderr}"

    output_dir = workdir / "train_outputs"
    cfg = {
        "seed": 42,
        "output_dir": str(output_dir),
        "data": {
            "manifest_csv": str(manifest_csv),
            "audio_column": "path",
            "text_column": "transcript",
            "val_split": 0.5,
            "max_audio_seconds": 20,
            "num_workers": 1,
        },
        "model": {
            "base_model_name": base_model,
            "language": None,
            "task": "transcribe",
            "forced_decoder_ids": False,
            "max_label_length": 128,
        },
        "lora": {"enabled": True, "r": 4, "alpha": 8, "dropout": 0.05, "target_modules": ["q_proj", "v_proj"]},
        "train": {
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 1e-4,
            "warmup_steps": 0,
            "num_train_epochs": 1,
            "max_steps": 1,
            "logging_steps": 1,
            "eval_steps": 1,
            "save_steps": 1,
            "save_total_limit": 1,
            "mixed_precision": "no",
        },
    }

    cfg_path = workdir / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    cmd_train = ["python", "-m", "script.train", "--config", str(cfg_path)]
    r_train = subprocess.run(cmd_train, cwd=str(repo_root), env=env, capture_output=True, text=True)
    assert r_train.returncode == 0, f"Training failed:\n{r_train.stdout}\n{r_train.stderr}"

    best_dir = next(output_dir.rglob("best"), None)
    assert best_dir is not None, "Missing 'best' directory"
    assert (best_dir / "adapter_config.json").exists()
    assert (best_dir / "adapter_model.safetensors").exists() or (best_dir / "adapter_model.bin").exists()
    assert (best_dir / "preprocessor_config.json").exists() or (best_dir / "processor_config.json").exists()
