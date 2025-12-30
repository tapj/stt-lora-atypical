import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml


def _is_offline_env() -> bool:
    return os.environ.get("HF_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"


def _has_network_block(output: str) -> bool:
    text = output.lower()
    patterns = [
        "403",
        "forbidden",
        "proxy",
        "huggingface.co",
        "maxretryerror",
        "connectionerror",
        "tunnel connection failed",
        "blocked",
    ]
    return any(pat in text for pat in patterns)


def test_train_smoke_example_pairs(tmp_path: Path):
    """
    Functional smoke test:
    - prepare manifest from data/example_pairs
    - run a tiny LoRA train for 1-2 steps on CPU
    - verify output artifacts exist
    """

    repo_root = Path(__file__).resolve().parents[1]

    # Copy example_pairs into a temp workdir to keep writes isolated
    workdir = tmp_path / "work"
    workdir.mkdir(parents=True, exist_ok=True)

    pairs_src = repo_root / "data" / "example_pairs"
    assert pairs_src.exists(), f"Missing {pairs_src}"

    pairs_dir = workdir / "example_pairs"
    shutil.copytree(pairs_src, pairs_dir)

    # Manifest output location
    manifest_csv = workdir / "manifest.csv"

    # Determine model + cache locations
    model_env = os.environ.get("STT_TEST_BASE_MODEL", "openai/whisper-tiny")
    model_path = Path(model_env)
    model_is_local = model_path.exists() and model_path.is_dir()

    offline_env = _is_offline_env()
    offline = offline_env or model_is_local

    cache_base_env = os.environ.get("STT_TEST_HF_HOME")
    cache_base = Path(cache_base_env) if cache_base_env else tmp_path / "hf_home"
    cache_base.mkdir(parents=True, exist_ok=True)

    # HF caches redirected into tmp or provided cache
    hf_home = cache_base
    torch_home = tmp_path / "torch_home"
    hf_datasets_cache = tmp_path / "hf_datasets_cache"
    xdg_cache_home = tmp_path / "xdg_cache_home"
    torch_home.mkdir(parents=True, exist_ok=True)
    hf_datasets_cache.mkdir(parents=True, exist_ok=True)
    xdg_cache_home.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["HF_HOME"] = str(hf_home)
    env["TORCH_HOME"] = str(torch_home)
    env["HF_DATASETS_CACHE"] = str(hf_datasets_cache)
    env["XDG_CACHE_HOME"] = str(xdg_cache_home)
    env["CUDA_VISIBLE_DEVICES"] = ""
    if offline:
        env["HF_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"

    # 1) Prepare manifest (paths in manifest will be relative to pairs_dir parent)
    cmd_manifest = [
        "python",
        "-m",
        "data.prepare_manifest",
        "--pairs_dir",
        str(pairs_dir),
        "--out_csv",
        str(manifest_csv),
    ]
    r = subprocess.run(cmd_manifest, cwd=str(repo_root), env=env, capture_output=True, text=True)
    assert r.returncode == 0, (
        "prepare_manifest failed:\n"
        f"STDOUT:\n{r.stdout}\n"
        f"STDERR:\n{r.stderr}"
    )
    assert manifest_csv.exists()

    # Offline but remote repo id -> skip
    if offline and not model_is_local:
        pytest.skip(
            "HF download blocked/offline and STT_TEST_BASE_MODEL is not a local path; "
            "set STT_TEST_BASE_MODEL to a local Whisper checkout or pre-seed HF cache."
        )

    # 2) Write a minimal config for CPU + tiny model + very short run
    out_dir = workdir / "outputs_smoke"
    cfg = {
        "seed": 1337,
        "output_dir": str(out_dir),
        "data": {
            "manifest_csv": str(manifest_csv),
            "pairs_dir": None,
            "audio_column": "path",
            "text_column": "transcript",
            "language_column": "language",
            "speaker_column": "speaker",
            "val_split": 0.34,  # small dataset -> still get at least 1 val example
            "max_audio_seconds": 20,
            "num_workers": 1,
            "normalize_loudness": False,
            "augment": {"enabled": False},
        },
        "model": {
            "base_model_name": model_env if not model_is_local else str(model_path),
            "language": None,
            "task": "transcribe",
            "forced_decoder_ids": False,
            "max_label_length": 128,
        },
        "lora": {
            "enabled": True,
            "r": 4,
            "alpha": 8,
            "dropout": 0.05,
            "target_modules": ["q_proj", "v_proj"],
        },
        "train": {
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 1e-4,
            "warmup_steps": 0,
            "num_train_epochs": 1,
            "max_steps": 2,  # keep very short
            "weight_decay": 0.0,
            "logging_steps": 1,
            "eval_steps": 1,
            "save_steps": 1,
            "save_total_limit": 1,
            "mixed_precision": "no",
            "gradient_checkpointing": False,
            "max_grad_norm": 1.0,
        },
        "decode": {"beam_size": 1, "temperature": 0.0, "no_repeat_ngram_size": 0},
        "personalization": {"phrase_list": [], "correction_dict": {}},
        "live": {"sample_rate": 16000, "chunk_ms": 30, "vad": {"backend": "energy"}},
    }

    cfg_path = workdir / "smoke_config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    # 3) Run train
    cmd_train = ["python", "-m", "script.train", "--config", str(cfg_path)]
    r = subprocess.run(cmd_train, cwd=str(repo_root), env=env, capture_output=True, text=True)

    if r.returncode != 0:
        combined = f"{r.stdout}\n{r.stderr}"
        if _has_network_block(combined):
            pytest.skip(
                "HF download blocked in this environment; set STT_TEST_BASE_MODEL to a local path "
                "or pre-seed HF cache via STT_TEST_HF_HOME."
            )
        assert r.returncode == 0, (
            "train failed:\n"
            f"STDOUT:\n{r.stdout}\n"
            f"STDERR:\n{r.stderr}"
        )

    # 4) Verify artifacts exist
    assert out_dir.exists(), "output_dir missing"
    best_dirs = list(out_dir.rglob("best"))
    assert best_dirs, f"No 'best' dir found under {out_dir}"

    best_dir = best_dirs[0]
    # LoRA adapters typically produce adapter_model.safetensors and adapter_config.json
    assert (best_dir / "adapter_config.json").exists()
    assert (best_dir / "adapter_model.safetensors").exists() or (best_dir / "adapter_model.bin").exists()

    # Processor files
    assert (best_dir / "preprocessor_config.json").exists() or (best_dir / "processor_config.json").exists()
