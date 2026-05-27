# AskMe — 简历AI聊天机器人

基于 LLM-Wiki 知识库的 RAG 简历问答系统。HR 扫码即可与 AI 对话，了解候选人背景。

## 架构

- **节点A（树莓派）**：FastAPI + Docker，FRP 穿透
- **节点B（外网兜底）**：Cloudflare Pages + Workers

## 快速开始

```bash
# 生成知识库索引
python server/indexer.py

# 启动服务
docker-compose up -d
```

## 文档

详见 [001_PRD.md](./001_PRD.md)
