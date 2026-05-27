#!/bin/bash
# AskMe — 直接启动（不用 Docker，轻量化）
set -e

cd "$(dirname "$0")/.."
source .venv/bin/activate

PORT=${PORT:-8765}
echo "AskMe starting on port $PORT..."
exec python -m uvicorn server.main:app --host 0.0.0.0 --port $PORT
