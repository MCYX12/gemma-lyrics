#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ollama create gemma-lyrics -f "$ROOT_DIR/Modelfile"
