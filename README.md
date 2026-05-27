# AskMe

简历 AI 聊天机器人。简历上放置二维码，面试官/HR/猎头扫码后与 AI 对话，基于个人知识库准确回答职业经历、项目经验、技能背景等相关问题。

## 架构

```
简历二维码 → 域名
                  │
            反向代理
                  │
            内网穿透
                  │
            ┌─────┴─────┐
            │ FastAPI   │
            │ ├ 聊天前端  │
            │ ├ RAG 引擎 │
            │ └ LLM 调用 │
            └─────┬─────┘
                  │
        ┌─────────┼─────────┐
        │         │         │
   知识库文件   Embedding   LLM API
   (markdown)   (bge-m3)  (GLM-5-turbo)
```

## 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| 前端 | Vanilla HTML + Tailwind CDN | 单文件，无构建步骤，移动端优先 |
| 后端 | Python FastAPI + Uvicorn | 异步 SSE 流式输出 |
| RAG | bge-m3 向量检索 + 全量 context 注入 | 数据量小（~8K tokens），全量塞入 128K context |
| LLM | GLM-5-turbo via Z.AI | 流式 SSE，过滤 reasoning_content |
| Embedding | BAAI/bge-m3 via SiliconFlow | 1024 维向量，免费额度 |
| 向量存储 | JSON 文件（内存加载） | 10 个文件，< 300KB，无需向量数据库 |
| 部署 | systemd 服务 | 开机自启，崩溃自动恢复 |

## 项目结构

```
AskMe/
├── server/                   # 后端
│   ├── main.py               # FastAPI 入口：路由 + 中间件
│   ├── rag.py                # RAG 引擎：向量加载 + 检索 + context 组装
│   ├── llm.py                # LLM 调用：GLM API 流式封装
│   ├── indexer.py            # 索引器：知识库 → 向量 JSON
│   ├── requirements.txt      # Python 依赖
│   └── Dockerfile            # 容器化（备选部署方式）
├── web/
│   └── index.html            # 聊天前端页面
├── data/
│   ├── knowledge/            # 知识库源文件（本地，不上传）
│   │   ├── 01_公司A/
│   │   ├── 02_公司B/
│   │   ├── 03_公司C/
│   │   ├── 04_个人/
│   │   ├── 05_简历/
│   │   └── 06_方法论/
│   └── embeddings.json       # 向量索引（自动生成，不上传）
├── scripts/
│   ├── reindex.sh            # 一键重建索引
│   ├── start.sh              # 手动启动服务
│   └── askme.service         # systemd 服务单元
├── docker-compose.yml        # Docker Compose 部署（备选）
├── .env                      # API 密钥（不上传）
└── .gitignore
```

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 聊天前端页面 |
| POST | `/api/chat` | RAG + LLM 流式返回（SSE） |
| GET | `/api/health` | 健康检查 |
| POST | `/api/reindex` | 重建知识库索引 |
| GET | `/robots.txt` | 禁止搜索引擎收录 |

### POST /api/chat

请求：
```json
{
  "message": "介绍一下工作经历",
  "history": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
}
```

响应：SSE 流
```
data: {"content": "在"}
data: {"content": "公司A"}
...
data: [DONE]
```

## 安全措施

| 措施 | 说明 |
|------|------|
| 敏感信息脱敏 | 知识库中手机号、邮箱等替换为占位符 |
| System Prompt 约束 | 禁止透露联系方式，只回答简历相关问题 |
| 对话轮次限制 | 每 IP 最多 20 轮 |
| 请求限流 | 每 IP 每 60 秒最多 20 次请求 |
| robots.txt | 禁止所有搜索引擎收录 |
| API 密钥隔离 | 密钥存储在 .env 文件中，不进入版本控制 |

## 数据流

```
知识库准备（一次性 / 更新时重跑）：
  markdown 文件 → indexer.py → bge-m3 Embedding API → embeddings.json

在线查询（每次对话）：
  用户问题 → Embedding → 余弦相似度 top-K 检索
           → system prompt + 全量知识库 + 安全约束
           → GLM-5-turbo 流式生成 → SSE 推送到前端
```

## 知识库更新

1. 替换 `data/knowledge/` 下的 markdown 文件
2. 运行 `bash scripts/reindex.sh`
3. `sudo systemctl restart askme`

QR 码指向域名，知识库更新不影响二维码。

## 环境配置

`.env` 文件（需手动创建）：

```
SILICONFLOW_API_KEY=sk-xxx      # SiliconFlow bge-m3 embedding
GLM_API_KEY=xxx                  # Z.AI GLM-5-turbo
GLM_BASE_URL=https://api.z.ai/api/coding/paas/v4
```

## 依赖

- Python 3.12+
- fastapi, uvicorn, httpx, numpy, scikit-learn, python-dotenv
- Python venv（`.venv/`，不上传）

## 许可

[MIT License](LICENSE)
