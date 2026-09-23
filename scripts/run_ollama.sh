#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="$ROOT_DIR/venv/bin/python"
RAW_INPUT="${1:-风格：冷淡克制 情绪：孤独 场景：夜晚城市}"

if [[ "$RAW_INPUT" == *"简谱"* || "$RAW_INPUT" == *"数字简谱"* || "$RAW_INPUT" == *"数字谱"* ]]; then
  CLEAN_INPUT="${RAW_INPUT//数字简谱/}"
  CLEAN_INPUT="${CLEAN_INPUT//数字谱/}"
  CLEAN_INPUT="${CLEAN_INPUT//简谱/}"
  CLEAN_INPUT="${CLEAN_INPUT//带对应歌词/}"
  CLEAN_INPUT="${CLEAN_INPUT//对应歌词/}"

  if [[ "$RAW_INPUT" == *"四句"* || "$RAW_INPUT" == *"几句"* || "$RAW_INPUT" == *"片段"* || "$RAW_INPUT" == *"一段"* ]]; then
    PROMPT="写一段完整中文歌词，控制在4到8行。${CLEAN_INPUT}"
  else
    PROMPT="写一整首完整中文歌词，严格按5段输出，中间空一行。第1段主歌4行，第2段主歌4行，第3段副歌4行，第4段桥段4行，第5段收束4行。副歌最多重复1句，其余句子不要重复。${CLEAN_INPUT}"
  fi
elif [[ "$RAW_INPUT" == *"风格："* || "$RAW_INPUT" == *"情绪："* || "$RAW_INPUT" == *"场景："* || "$RAW_INPUT" == *"结构："* || "$RAW_INPUT" == *"意象："* ]]; then
  PROMPT="写一整首完整中文歌词。${RAW_INPUT}"
else
  PROMPT="$RAW_INPUT"
fi

if [[ "$RAW_INPUT" == *"简谱"* || "$RAW_INPUT" == *"数字简谱"* || "$RAW_INPUT" == *"数字谱"* ]]; then
  LYRICS="$(ollama run gemma-lyrics --think=false "$PROMPT")"
  printf '%s\n' "$LYRICS" | "$VENV_PYTHON" "$ROOT_DIR/scripts/lyrics_to_jianpu.py"
  exit 0
fi

ollama run gemma-lyrics --think=false "$PROMPT"
