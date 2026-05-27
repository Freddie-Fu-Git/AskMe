#!/usr/bin/env python3
"""
AskMe Indexer — 从 knowledge 目录生成检索索引

方案：数据量极小（10个文件，~16K tokens），采用双层策略：
1. 全量模式：所有文件内容直接作为 context（128K context 够用）
2. 检索模式：TF-IDF 向量化 + 余弦相似度，按需 top-K 检索

不依赖外部 Embedding API，纯本地计算。
"""

import json
import math
import os
import re
import sys
from pathlib import Path

# 词频统计用的简单中文分词（按字+英文单词）
def tokenize(text: str) -> list[str]:
    """简单分词：保留中文单字和英文单词"""
    tokens = []
    # 英文单词
    tokens.extend(re.findall(r'[a-zA-Z]{2,}', text.lower()))
    # 中文字符
    tokens.extend(re.findall(r'[\u4e00-\u9fff]', text))
    return tokens


def compute_tfidf(documents: list[dict]) -> list[dict]:
    """计算 TF-IDF 向量"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    corpus = [doc["text"] for doc in documents]
    
    # 用自定义分词器，支持中文
    vectorizer = TfidfVectorizer(
        tokenizer=tokenize,
        token_pattern=None,  # 禁用默认 pattern
        max_features=5000,
        lowercase=True,
    )
    
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    # 把稀疏矩阵转为 dense list 存 JSON
    for i, doc in enumerate(documents):
        vec = tfidf_matrix[i].toarray().flatten()
        doc["tfidf_vector"] = vec.tolist()
        doc["tfidf_norm"] = float(math.sqrt(sum(v*v for v in vec)))
    
    # 保存词汇表（查询时需要）
    vocabulary = vectorizer.vocabulary_
    
    return documents, vectorizer


def load_knowledge(knowledge_dir: str) -> list[dict]:
    """递归读取 knowledge 目录下所有 .md 文件"""
    knowledge_path = Path(knowledge_dir)
    documents = []
    
    for md_file in sorted(knowledge_path.rglob("*.md")):
        # 从路径提取分类（子目录名）
        relative = md_file.relative_to(knowledge_path)
        category = str(relative.parent) if str(relative.parent) != "." else "root"
        
        text = md_file.read_text(encoding="utf-8").strip()
        if not text:
            continue
        
        title = md_file.stem
        
        documents.append({
            "file": str(relative),
            "title": title,
            "category": category,
            "text": text,
            "char_count": len(text),
        })
    
    return documents


def main():
    # 路径
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    knowledge_dir = project_dir / "data" / "knowledge"
    output_file = project_dir / "data" / "index.json"
    
    if not knowledge_dir.exists():
        print(f"错误：知识库目录不存在 {knowledge_dir}")
        sys.exit(1)
    
    print(f"读取知识库: {knowledge_dir}")
    documents = load_knowledge(knowledge_dir)
    print(f"加载 {len(documents)} 个文件:")
    
    total_chars = 0
    for doc in documents:
        print(f"  [{doc['category']}] {doc['title']} ({doc['char_count']} 字符)")
        total_chars += doc["char_count"]
    
    print(f"\n总计: {total_chars} 字符 ≈ {total_chars // 2} tokens")
    
    # 计算 TF-IDF
    print("\n计算 TF-IDF 向量...")
    documents, vectorizer = compute_tfidf(documents)
    vocab_size = len(vectorizer.vocabulary_)
    print(f"词汇表大小: {vocab_size}")
    
    # 全量 context（拼成一个字符串，供全量模式使用）
    full_context_parts = []
    for doc in documents:
        full_context_parts.append(f"## {doc['title']}\n\n{doc['text']}")
    full_context = "\n\n---\n\n".join(full_context_parts)
    
    # 构建索引
    index = {
        "version": "1.0",
        "files": documents,
        "full_context": full_context,
        "total_chars": total_chars,
        "total_files": len(documents),
    }
    
    # 保存
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    
    print(f"\n索引已保存: {output_file}")
    print(f"文件大小: {output_file.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
