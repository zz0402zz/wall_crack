#!/bin/zsh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

export BASE_DATA_DIR="$ROOT_DIR/.label_studio"
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="$ROOT_DIR"
export LATEST_VERSION_CHECK=false
export LABEL_STUDIO_USERNAME="${LABEL_STUDIO_USERNAME:-admin@example.com}"
export LABEL_STUDIO_PASSWORD="${LABEL_STUDIO_PASSWORD:-admin123456}"

exec "$ROOT_DIR/.venv/bin/label-studio" start \
  --internal-host 127.0.0.1 \
  --port 8080 \
  --no-browser \
  --init \
  --username "$LABEL_STUDIO_USERNAME" \
  --password "$LABEL_STUDIO_PASSWORD"
