#!/bin/bash
# AskMe — 一键重建知识库索引
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
python server/indexer.py
echo ""
echo "重建完成。重启 AskMe 服务使索引生效:"
echo "  docker-compose restart askme"
