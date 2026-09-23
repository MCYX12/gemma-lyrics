#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$ROOT_DIR/data"
MODEL_DIR="$ROOT_DIR/model/gemma4_mlx"
ADAPTER_DIR="$ROOT_DIR/adapter/lora"
PREPARE_SCRIPT="$ROOT_DIR/scripts/prepare_mlx_data.py"
VENV_PYTHON="$ROOT_DIR/venv/bin/python"

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "venv python not found at $VENV_PYTHON"
  exit 1
fi

"$VENV_PYTHON" "$PREPARE_SCRIPT" \
  --source "$DATA_DIR/train_original.jsonl" \
  --output-dir "$DATA_DIR"

"$VENV_PYTHON" -m mlx_lm.lora \
  --model "$MODEL_DIR" \
  --train \
  --data "$DATA_DIR" \
  --fine-tune-type lora \
  --batch-size 1 \
  --iters 300 \
  --learning-rate 5e-5 \
  --max-seq-length 512 \
  --adapter-path "$ADAPTER_DIR"
