#!/usr/bin/env bash
set -euo pipefail

. .venv/bin/activate

export PATH="./bin:${PATH}"
export HF_HOME="./.hf"
export HF_DATASETS_CACHE="./.hf/datasets"
export TORCH_HOME="./.torch"

mkdir -p ./.hf/datasets ./.torch

python -m uvicorn app.app:app
