import argparse
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import yaml

from src.manifest_utils import resolve_manifest_paths


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_path(path_value: str, base_dir: Path) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _require(section: Dict[str, Any], key: str, context: str) -> Any:
    if key not in section or section.get(key) is None:
        raise ValueError(f"Missing required config field: {context}.{key}")
    return section.get(key)


def _get_manifest_path(cfg: Dict[str, Any], cfg_path: Path) -> Path:
    data = cfg.get("data", {}) or {}
    manifest = data.get("manifest_csv") or data.get("manifest")
    if not manifest:
        raise ValueError("Provide data.manifest_csv (or data.manifest) in the YAML config.")
    return _resolve_path(str(manifest), cfg_path.parent)


def _build_command(cfg: Dict[str, Any], resolved_manifest: str) -> List[str]:
    lora_cfg = cfg.get("lora", {}) or {}
    if not lora_cfg.get("enabled", True):
        raise ValueError("LoRA disabled but YAML workflow uses LoRA trainer; enable lora.enabled or implement non-LoRA path")

    data_cfg = cfg.get("data", {}) or {}
    model_cfg = cfg.get("model", {}) or {}
    train_cfg = cfg.get("train", {}) or {}

    language = model_cfg.get("language")
    language_val = language if language is not None else ""
    task = model_cfg.get("task") or "transcribe"
    max_steps = train_cfg.get("max_steps")
    max_steps_val = max_steps if max_steps is not None else -1

    target_modules = lora_cfg.get("target_modules") or []
    if not target_modules:
        raise ValueError("lora.target_modules is required for LoRA training")
    targets_csv = ",".join(target_modules)

    cmd = [
        "python",
        "-u",
        "src/train_lora_whisper.py",
        "--manifest_csv",
        resolved_manifest,
        "--output_dir",
        str(_require(cfg, "output_dir", "root")),
        "--base_model",
        str(_require(model_cfg, "base_model_name", "model")),
        "--language",
        str(language_val),
        "--task",
        str(task),
        "--val_split",
        str(_require(data_cfg, "val_split", "data")),
        "--max_audio_seconds",
        str(_require(data_cfg, "max_audio_seconds", "data")),
        "--seed",
        str(_require(cfg, "seed", "root")),
        "--batch_size",
        str(_require(train_cfg, "batch_size", "train")),
        "--grad_accum",
        str(_require(train_cfg, "gradient_accumulation_steps", "train")),
        "--lr",
        str(_require(train_cfg, "learning_rate", "train")),
        "--warmup_steps",
        str(_require(train_cfg, "warmup_steps", "train")),
        "--epochs",
        str(_require(train_cfg, "num_train_epochs", "train")),
        "--max_steps",
        str(max_steps_val),
        "--eval_steps",
        str(_require(train_cfg, "eval_steps", "train")),
        "--save_steps",
        str(_require(train_cfg, "save_steps", "train")),
        "--logging_steps",
        str(_require(train_cfg, "logging_steps", "train")),
        "--save_total_limit",
        str(_require(train_cfg, "save_total_limit", "train")),
        "--lora_r",
        str(_require(lora_cfg, "r", "lora")),
        "--lora_alpha",
        str(_require(lora_cfg, "alpha", "lora")),
        "--lora_dropout",
        str(_require(lora_cfg, "dropout", "lora")),
        "--lora_targets",
        targets_csv,
    ]
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    args = ap.parse_args()

    cfg_path = Path(args.config).expanduser()
    cfg = _read_yaml(cfg_path)

    manifest_path = _get_manifest_path(cfg, cfg_path)
    resolved_manifest = resolve_manifest_paths(str(manifest_path))

    output_dir = _resolve_path(str(cfg["output_dir"]), cfg_path.parent)
    cfg = dict(cfg)
    cfg["output_dir"] = str(output_dir)

    cmd = _build_command(cfg, resolved_manifest)

    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


if __name__ == "__main__":
    main()
