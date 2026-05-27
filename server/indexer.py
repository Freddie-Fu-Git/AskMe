#!/usr/bin/env python3
"""
AskMe Indexer — 从 knowledge 目录生成向量索引

使用硅基流动 BAAI/bge-m3 embedding 模型（1024维）。
数据量极小（10个文件，~8K tokens），向量索引文件 < 1MB。
"""

import json
import os
import sys
import time
from pathlib import Path

import httpx

# 配置
SILICONFLOW_BASE = "https://api.siliconflow.cn/v1"
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# 从 .env 或环境变量读取
API_KEY = os.environ.get("SILICONFLOW_API_KEY", "")


def get_api_key() -> str:
    """获取 API key"""
    if API_KEY:
        return API_KEY
    # 尝试从 .env 文件读取
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("SILICONFLOW_API_KEY="):
                return line.split("=", 1)[1].strip()
    print("错误：未找到 SILICONFLOW_API_KEY")
    print("请创建 AskMe/.env 文件，内容：SILICONFLOW_API_KEY=sk-xxx")
    sys.exit(1)


def embed_texts(texts: list[str], api_key: str) -> list[list[float]]:
    """调用硅基流动 embedding API，支持批量"""
    all_vectors = []
    batch_size = 10  # bge-m3 支持批量
    
    with httpx.Client(timeout=60) as client:
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            print(f"  Embedding {i+1}-{i+len(batch)} / {len(texts)} ...")
            
            resp = client.post(
                f"{SILICONFLOW_BASE}/embeddings",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": EMBEDDING_MODEL,
                    "input": batch,
                    "encoding_format": "float",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            
            # 按 index 排序确保顺序正确
            sorted_data = sorted(data["data"], key=lambda x: x["index"])
            for item in sorted_data:
                all_vectors.append(item["embedding"])
            
            # 避免 rate limit
            if i + batch_size < len(texts):
                time.sleep(0.5)
    
    return all_vectors


def load_knowledge(knowledge_dir: Path) -> list[dict]:
    """递归读取 knowledge 目录下所有 .md 文件"""
    documents = []
    
    for md_file in sorted(knowledge_dir.rglob("*.md")):
        relative = md_file.relative_to(knowledge_dir)
        category = str(relative.parent) if str(relative.parent) != "." else "root"
        
        text = md_file.read_text(encoding="utf-8").strip()
        if not text:
            continue
        
        documents.append({
            "file": str(relative),
            "title": md_file.stem,
            "category": category,
            "text": text,
            "char_count": len(text),
        })
    
    return documents


def main():
    project_dir = Path(__file__).parent.parent
    knowledge_dir = project_dir / "data" / "knowledge"
    output_file = project_dir / "data" / "embeddings.json"
    
    if not knowledge_dir.exists():
        print(f"错误：知识库目录不存在 {knowledge_dir}")
        sys.exit(1)
    
    api_key = get_api_key()
    
    print(f"读取知识库: {knowledge_dir}")
    documents = load_knowledge(knowledge_dir)
    print(f"加载 {len(documents)} 个文件:")
    
    total_chars = 0
    for doc in documents:
        print(f"  [{doc['category']}] {doc['title']} ({doc['char_count']} 字符)")
        total_chars += doc["char_count"]
    
    print(f"\n总计: {total_chars} 字符 ≈ {total_chars // 2} tokens")
    
    # 调用 embedding API
    print(f"\n调用 {EMBEDDING_MODEL} 生成向量 (dim={EMBEDDING_DIM})...")
    texts = [doc["text"] for doc in documents]
    vectors = embed_texts(texts, api_key)
    
    assert len(vectors) == len(documents), "向量数量与文件数量不匹配"
    
    # 组装
    for i, doc in enumerate(documents):
        doc["vector"] = vectors[i]
    
    # 全量 context
    full_context_parts = []
    for doc in documents:
        full_context_parts.append(f"## {doc['title']}\n\n{doc['text']}")
    full_context = "\n\n---\n\n".join(full_context_parts)
    
    index = {
        "version": "2.0",
        "model": EMBEDDING_MODEL,
        "dim": EMBEDDING_DIM,
        "files": documents,
        "full_context": full_context,
        "total_chars": total_chars,
        "total_files": len(documents),
    }
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    
    size_kb = output_file.stat().st_size / 1024
    print(f"\n向量索引已保存: {output_file}")
    print(f"文件大小: {size_kb:.1f} KB")
    print(f"向量维度: {EMBEDDING_DIM}")
    print("完成！")


if __name__ == "__main__":
    main()
