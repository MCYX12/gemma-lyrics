#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="$ROOT_DIR/venv/bin/python"
MODEL_DIR="$ROOT_DIR/model/gemma4_mlx"
ADAPTER_DIR="$ROOT_DIR/adapter/lora"
RAW_INPUT="${1:-写一段雨夜孤独的歌词}"

if [[ "$RAW_INPUT" == *"简谱"* || "$RAW_INPUT" == *"数字简谱"* || "$RAW_INPUT" == *"数字谱"* ]]; then
  CLEAN_INPUT="${RAW_INPUT//数字简谱/}"
  CLEAN_INPUT="${CLEAN_INPUT//数字谱/}"
  CLEAN_INPUT="${CLEAN_INPUT//简谱/}"
  CLEAN_INPUT="${CLEAN_INPUT//带对应歌词/}"
  CLEAN_INPUT="${CLEAN_INPUT//对应歌词/}"

  if [[ "$RAW_INPUT" == *"四句"* || "$RAW_INPUT" == *"几句"* || "$RAW_INPUT" == *"片段"* || "$RAW_INPUT" == *"一段"* ]]; then
    PROMPT="写一段完整中文歌词，控制在4到8行。${CLEAN_INPUT}"
    MIN_LINES=4
    MAX_TOKENS=192
    STRUCTURE_TEMPLATE=short
  else
    PROMPT="写一整首完整中文歌词，严格按5段输出，中间空一行。第1段主歌4行，第2段主歌4行，第3段副歌4行，第4段桥段4行，第5段收束4行。副歌最多重复1句，其余句子不要重复。${CLEAN_INPUT}"
    MIN_LINES=16
    MAX_TOKENS=448
    STRUCTURE_TEMPLATE=full_song
  fi
elif [[ "$RAW_INPUT" == *"风格："* || "$RAW_INPUT" == *"情绪："* || "$RAW_INPUT" == *"场景："* || "$RAW_INPUT" == *"结构："* || "$RAW_INPUT" == *"意象："* ]]; then
  PROMPT="写一整首完整中文歌词。${RAW_INPUT}"
else
  PROMPT="$RAW_INPUT"
fi

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "venv python not found at $VENV_PYTHON"
  exit 1
fi

if [[ "$RAW_INPUT" == *"简谱"* || "$RAW_INPUT" == *"数字简谱"* || "$RAW_INPUT" == *"数字谱"* ]]; then
  LYRICS="$(
    "$VENV_PYTHON" "$ROOT_DIR/scripts/generate_lyrics.py" \
      --model "$MODEL_DIR" \
      --adapter-path "$ADAPTER_DIR" \
      --prompt "$PROMPT" \
      --quiet \
      --structure-template "${STRUCTURE_TEMPLATE:-none}" \
      --min-lines "$MIN_LINES" \
      --max-attempts 3 \
      --max-tokens "$MAX_TOKENS"
  )"

  printf '%s\n' "$LYRICS" | "$VENV_PYTHON" "$ROOT_DIR/scripts/lyrics_to_jianpu.py"
  exit 0
fi

CMD=(
  "$VENV_PYTHON"
  "$ROOT_DIR/scripts/generate_lyrics.py"
  --model "$MODEL_DIR"
  --adapter-path "$ADAPTER_DIR"
  --prompt "$PROMPT"
)

"${CMD[@]}"
