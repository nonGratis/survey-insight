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
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

# TODO Week 3 (Cloud Run): замінити хардкод 8501 на $PORT.
# Cloud Run передає порт через env var $PORT (зазвичай 8080). Варіант:
# CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]
# Поки локально/Docker — фіксований 8501. headless=true вже в .streamlit/config.toml.
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
