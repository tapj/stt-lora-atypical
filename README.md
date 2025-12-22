# Local STT LoRA for atypical speech (Whisper + PEFT)

## Goals
- Local-first supervised fine-tuning with LoRA adapters.
- Real-time mic inference (utterance-level streaming with VAD).
- Batch/file inference.
- Reproducible runs: config, pinned deps, logs, checkpoints.

## Setup
Python 3.11+ recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows (Git Bash):

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

Environment variables

Copy `.env.example` to `.env` and adjust as needed:

```bash
cp .env.example .env
```

- `STT_CONFIG_PATH` points to the YAML config used by the app (defaults to `config.yaml`).
- `STT_ADAPTER_DIR` sets the default adapter directory shown in the web UI (defaults to `outputs/run1/best`).
- `HUGGINGFACE_HUB_TOKEN` is optional and used when downloading gated models (e.g., Whisper checkpoints hosted on Hugging Face). The app uses `openai/whisper-small` by default and does **not** require an OpenAI API key.

GPU (optional):

Install a CUDA-enabled torch build that matches your system if the pinned torch does not.

Data formats
Option A: CSV manifest

CSV columns:

path, transcript, language(optional), speaker(optional)

Example: data/example_manifest.csv

Downloadable example (for quick smoke tests, configurable language/dataset):

```bash
python data/save_hf_subset_pairs.py --dataset mozilla-foundation/common_voice_11_0 --config fr --split train --n 50 --out_dir data/example_pairs
```

The script pulls a subset from Hugging Face Common Voice (defaults to French) and saves WAV + `.txt` pairs to `data/example_pairs`. Afterward, create a manifest with the provided helper:

```bash
python data/prepare_manifest.py --pairs_dir data/example_pairs --out_csv data/example_manifest.csv
```

These small samples are for wiring sanity checks; accuracy is not the goal.

Option B: audio + txt pairs

If you have:

sample.wav

sample.txt

Create a manifest:

```python
python data/prepare_manifest.py --pairs_dir /path/to/pairs --out_csv data/manifest.csv
```

Then set data.manifest_csv: data/manifest.csv in config.yaml.

Training

Edit config.yaml then run:

```python
python scripts/train.py --config config.yaml
```

Outputs:

outputs/runX/best/ (best LoRA adapter by WER + processor config)

outputs/runX/metrics.jsonl

Evaluate a held-out folder

Create a folder with audio+txt pairs and run:


```python
python scripts/eval_folder.py --adapter_dir outputs/run1/best --pairs_dir /path/to/heldout --config config.yaml

```

Inference (file)

```python
python scripts/infer_file.py --adapter_dir outputs/run1/best --audio path/to/audio.wav --config config.yaml

```

Inference (live mic)

```python
python scripts/infer_live.py --adapter_dir outputs/run1/best --config config.yaml
```


```bash
uvicorn app.app:app --host 127.0.0.1 --port 8000

```

Open http://127.0.0.1:8000

Buttons: Record, Stop, Transcribe, Copy, Save transcript.

Troubleshooting
1) Web UI audio decode fails

Browser records audio/webm by default. If torchaudio cannot decode webm on your OS:

- Use file mode (wav/flac/mp3) as primary.
- Or switch to a desktop UI.
- Or install ffmpeg and ensure torchaudio has backend support.

2) CUDA OOM

- Lower batch_size.
- Increase gradient_accumulation_steps.
- Reduce max_audio_seconds.
- Use fp16 (or bf16 on supported GPUs).

3) Accuracy is worse after fine-tuning

Common causes:

- Too aggressive augmentation.
- Too little atypical-speech coverage.
- Transcripts not matching what is spoken.
- Mixed languages without language control.
