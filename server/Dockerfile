FROM python:3.12-slim

WORKDIR /app

# 安装依赖
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY server/ .
COPY data/ ./data/
COPY web/ ./web/

# 加载 .env（docker-compose 会通过 env_file 注入）

EXPOSE 8765

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8765"]
