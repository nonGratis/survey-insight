FROM python:3.11-slim

WORKDIR /app

# curl потрібен для HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD curl -fsS "http://localhost:${PORT:-8501}/_stcore/health" || exit 1

# Порт параметризовано: Cloud Run передає його через env var $PORT (зазвичай
# 8080); локально/Docker — фолбек 8501. headless=true вже в .streamlit/config.toml.
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]
