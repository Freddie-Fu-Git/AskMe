#!/usr/bin/env python3
"""快速测试检索质量"""
import json
import math
import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

def tokenize(text: str) -> list[str]:
    import re
    tokens = []
    tokens.extend(re.findall(r'[a-zA-Z]{2,}', text.lower()))
    tokens.extend(re.findall(r'[\u4e00-\u9fff]', text))
    return tokens

# 加载索引
index_file = Path(__file__).parent.parent / "data" / "index.json"
with open(index_file, encoding="utf-8") as f:
    index = json.load(f)

files = index["files"]
corpus = [doc["text"] for doc in files]
titles = [doc["title"] for doc in files]

vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, max_features=5000, lowercase=True)
tfidf_matrix = vectorizer.fit_transform(corpus)

def search(query: str, top_k: int = 3):
    q_vec = vectorizer.transform([query]).toarray().flatten()
    scores = []
    for i in range(len(files)):
        d_vec = tfidf_matrix[i].toarray().flatten()
        # 余弦相似度
        dot = sum(a*b for a, b in zip(q_vec, d_vec))
        norm_q = math.sqrt(sum(v*v for v in q_vec))
        norm_d = math.sqrt(sum(v*v for v in d_vec))
        if norm_q > 0 and norm_d > 0:
            sim = dot / (norm_q * norm_d)
        else:
            sim = 0
        scores.append((sim, i))
    scores.sort(reverse=True)
    
    print(f"\n查询: 「{query}」")
    print(f"{'排名':<4} {'相似度':<10} {'分类':<12} {'文件'}")
    print("-" * 50)
    for rank, (sim, idx) in enumerate(scores[:top_k], 1):
        print(f"{rank:<4} {sim:<10.4f} {files[idx]['category']:<12} {titles[idx]}")
    return scores[:top_k]

# 测试用例
test_queries = [
    "你在自如做了什么",
    "百融云创的工作经历",
    "企业文化是怎么做的",
    "你有什么项目经验",
    "AI相关的技能",
    "Tell me about your work experience",
    "STAR法则怎么用的",
    "译禾是做什么的",
]

for q in test_queries:
    search(q, top_k=3)

print("\n\n=== 检索测试完成 ===")
