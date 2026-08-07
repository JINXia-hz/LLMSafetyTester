# LLMSEC 安全评估工具——一键容器（含聚类 + 预缓存 embedding 模型）
#
# 构建：
#   docker build -t llmsec .
#
# 运行（Web 面板，零预处理——.env 由 entrypoint 自动创建，看板 UI 配置）：
#   docker run -p 8080:8080 -v llmsec-data:/app/output llmsec
#
# output 卷持久化评估结果 + .env 配置（重启不丢）。

FROM python:3.11-slim AS base

# 系统依赖（sentence-transformers/torch 需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先装依赖（利用 Docker 层缓存：代码变了不重装）
COPY pyproject.toml llmsec/requirements-cluster.txt ./
RUN pip install --no-cache-dir -r requirements-cluster.txt

# 预缓存 embedding 模型（all-MiniLM-L6-v2，~90MB；避免运行时下载）
# 国内环境可设 HF_ENDPOINT=https://hf-mirror.com
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" \
    || echo "WARNING: embedding 模型预缓存失败（网络问题），运行时会自动重试下载"

# 装项目本身
COPY . .
RUN pip install --no-cache-dir -e .

# 创建数据目录
RUN mkdir -p /app/attacks /app/output /app/data

# 入口脚本：自动创建/恢复 .env（无需宿主机预处理）
COPY docker-entrypoint.sh /app/
RUN chmod +x /app/docker-entrypoint.sh

VOLUME ["/app/output", "/app/attacks"]

EXPOSE 8080

ENTRYPOINT ["/app/docker-entrypoint.sh"]
# 默认启动 Web 面板
CMD ["python", "-m", "uvicorn", "llmsec.server.dashboard_api:app", "--host", "0.0.0.0", "--port", "8080"]
