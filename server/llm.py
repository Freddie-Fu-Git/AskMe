"""
AskMe LLM Client — Z.AI GLM-5-turbo 流式调用

OpenAI 兼容接口，SSE 流式返回。
过滤 reasoning_content（思考过程），只输出正文。
"""

import json
import os
from pathlib import Path
from typing import AsyncGenerator

import httpx

# 默认配置
DEFAULT_BASE_URL = "https://api.z.ai/api/coding/paas/v4"
DEFAULT_MODEL = "glm-5-turbo"
MAX_TOKENS = 1024


def _load_env():
    """从 .env 文件加载配置"""
    env_file = Path(__file__).parent.parent / ".env"
    env_vars = {}
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env_vars[k.strip()] = v.strip()
    return env_vars


def get_config() -> dict:
    """获取 LLM 配置"""
    env = _load_env()
    return {
        "api_key": os.environ.get("GLM_API_KEY", env.get("GLM_API_KEY", "")),
        "base_url": os.environ.get("GLM_BASE_URL", env.get("GLM_BASE_URL", DEFAULT_BASE_URL)),
        "model": os.environ.get("GLM_MODEL", env.get("GLM_MODEL", DEFAULT_MODEL)),
    }


async def stream_chat(
    system_prompt: str,
    user_message: str,
    history: list[dict] | None = None,
) -> AsyncGenerator[str, None]:
    """
    流式调用 GLM API，yield 文本片段。
    
    过滤 reasoning_content，只输出 content 部分。
    """
    config = get_config()
    
    if not config["api_key"]:
        yield "错误：未配置 GLM API Key"
        return
    
    # 组装消息
    messages = [{"role": "system", "content": system_prompt}]
    
    if history:
        # 只保留最近 6 轮（12 条消息），控制 context 长度
        recent = history[-12:]
        messages.extend(recent)
    
    messages.append({"role": "user", "content": user_message})
    
    payload = {
        "model": config["model"],
        "messages": messages,
        "stream": True,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.7,
    }
    
    url = f"{config['base_url']}/chat/completions"
    headers = {
        "Authorization": f"Bearer {config['api_key']}",
        "Content-Type": "application/json",
    }
    
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream("POST", url, json=payload, headers=headers) as resp:
            if resp.status_code != 200:
                error_text = await resp.aread()
                yield f"错误：API 调用失败 ({resp.status_code})"
                return
            
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue
                
                data_str = line[6:]  # 去掉 "data: "
                if data_str == "[DONE]":
                    break
                
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                
                choices = chunk.get("choices", [])
                if not choices:
                    continue
                
                delta = choices[0].get("delta", {})
                
                # 跳过 reasoning_content（思考过程）
                # 只输出 content
                content = delta.get("content", "")
                if content:
                    yield content


async def chat(
    system_prompt: str,
    user_message: str,
    history: list[dict] | None = None,
) -> str:
    """非流式调用，返回完整回复"""
    result = []
    async for chunk in stream_chat(system_prompt, user_message, history):
        result.append(chunk)
    return "".join(result)


if __name__ == "__main__":
    import asyncio
    
    async def test():
        print("测试 GLM 流式调用...")
        async for chunk in stream_chat(
            "你是一个简洁的助手，只回答一句话。",
            "你好"
        ):
            print(chunk, end="", flush=True)
        print("\n完成")
    
    asyncio.run(test())
