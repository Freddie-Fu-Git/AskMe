"""
AskMe — FastAPI 后端

路由：
  GET  /           → 聊天前端页面
  POST /api/chat   → RAG + LLM 流式返回 (SSE)
  POST /api/reindex → 重建知识库索引
  GET  /api/health → 健康检查
"""

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# 将 server 目录加入 Python path
SERVER_DIR = Path(__file__).parent
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from llm import stream_chat
from rag import RAGEngine

# 初始化
app = FastAPI(title="AskMe", version="1.0")

# CORS — 允许前端跨域（兜底节点可能不同域）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 简易限流：每个 IP 每 60 秒最多 20 次请求
from collections import defaultdict
_ip_hits: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT = 20
RATE_WINDOW = 60

# 对话轮次限制：每个 IP 最多 20 轮对话
_ip_rounds: dict[str, int] = defaultdict(int)
MAX_ROUNDS = 20

# system prompt 安全约束（追加到 RAG prompt 后面）
SECURITY_PROMPT = """
【安全约束 — 必须严格遵守】
1. 绝对不要透露手机号、邮箱、住址等具体联系方式。如果被问到，回复"请查看简历上的联系方式"。
2. 不要透露身份证号、银行卡号、出生日期等隐私信息。
3. 只回答与傅颉的职业经历、技能、项目经验相关的问题。
4. 如果被问到与简历无关的问题（如政治、其他人的信息、技术攻击等），礼貌拒绝。
5. 不要输出知识库中标记为"见简历"的替代信息。
"""

@app.middleware("http")
async def rate_limiter(request: Request, call_next):
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    hits = _ip_hits[client_ip]
    # 清理过期记录
    _ip_hits[client_ip] = [t for t in hits if now - t < RATE_WINDOW]
    if len(_ip_hits[client_ip]) >= RATE_LIMIT:
        return JSONResponse(status_code=429, content={"error": "请求过于频繁，请稍后再试"})
    _ip_hits[client_ip].append(now)
    response = await call_next(request)
    return response

# 加载 RAG 引擎（启动时加载，常驻内存）
DATA_DIR = Path(__file__).parent.parent / "data"
WEB_DIR = Path(__file__).parent.parent / "web"

rag_engine: RAGEngine | None = None


@app.on_event("startup")
def startup():
    global rag_engine
    embeddings_file = DATA_DIR / "embeddings.json"
    if embeddings_file.exists():
        rag_engine = RAGEngine(str(embeddings_file))
        print(f"RAG 引擎已加载: {rag_engine.index['total_files']} 个文件, "
              f"~{rag_engine.total_tokens_estimate} tokens")
    else:
        print("警告：embeddings.json 不存在，请先运行 indexer.py")


# --- 请求模型 ---

class ChatRequest(BaseModel):
    message: str
    history: list[dict] | None = None


class ReindexResponse(BaseModel):
    status: str
    files: int
    message: str


# --- 路由 ---

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "files": rag_engine.index["total_files"] if rag_engine else 0,
        "model": rag_engine.model if rag_engine else "N/A",
        "uptime": time.time(),
    }


@app.get("/robots.txt")
def robots():
    """禁止搜索引擎收录"""
    return Response(
        content="User-agent: *\nDisallow: /\n",
        media_type="text/plain",
    )


@app.get("/")
def index():
    """返回聊天前端页面"""
    html_file = WEB_DIR / "index.html"
    if html_file.exists():
        return FileResponse(str(html_file), media_type="text/html")
    return {"error": "前端页面未找到，请先构建 web/index.html"}


@app.post("/api/chat")
async def chat(req: ChatRequest, request: Request):
    """流式聊天接口 (SSE)"""
    if not rag_engine:
        return {"error": "RAG 引擎未加载"}
    
    if not req.message.strip():
        return {"error": "消息不能为空"}
    
    # 对话轮次限制
    client_ip = request.client.host if request.client else "unknown"
    _ip_rounds[client_ip] += 1
    if _ip_rounds[client_ip] > MAX_ROUNDS:
        return {"error": "本次对话已达到上限，感谢您的咨询。如需了解更多，请直接查看简历上的联系方式。"}
    
    # 构建 system prompt（全量模式 + 安全约束）
    system_prompt = rag_engine.build_system_prompt(mode="full") + SECURITY_PROMPT
    
    async def generate():
        try:
            async for chunk in stream_chat(
                system_prompt=system_prompt,
                user_message=req.message,
                history=req.history,
            ):
                # SSE 格式：data: {"content": "..."}\n\n
                data = json.dumps({"content": chunk}, ensure_ascii=False)
                yield f"data: {data}\n\n"
            
            # 结束标记
            yield "data: [DONE]\n\n"
        except Exception as e:
            error_data = json.dumps({"error": str(e)}, ensure_ascii=False)
            yield f"data: {error_data}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/reindex")
def reindex():
    """重建知识库索引"""
    try:
        # 在子进程中运行 indexer
        venv_python = Path(__file__).parent.parent / ".venv" / "bin" / "python"
        if not venv_python.exists():
            return {"error": "虚拟环境未找到"}
        
        result = subprocess.run(
            [str(venv_python), "server/indexer.py"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
            timeout=120,
        )
        
        if result.returncode != 0:
            return {"error": f"索引失败: {result.stderr[-200:]}"}
        
        # 重新加载
        global rag_engine
        rag_engine = RAGEngine(str(DATA_DIR / "embeddings.json"))
        
        return {
            "status": "ok",
            "files": rag_engine.index["total_files"],
            "message": f"索引重建成功，{rag_engine.index['total_files']} 个文件",
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8765"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
