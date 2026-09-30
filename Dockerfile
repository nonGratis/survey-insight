FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    SERVICE=web

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD if [ "$SERVICE" = "web" ]; then curl -fsS "http://localhost:${PORT:-8080}/_stcore/health" || exit 1; else curl -fsS "http://localhost:${PORT:-8080}/health" || exit 1; fi

CMD ["sh", "-c", "case \"$SERVICE\" in api) exec python -m uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080} ;; worker) exec python -m uvicorn worker.main:app --host 0.0.0.0 --port ${PORT:-8080} ;; web|*) exec streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-8080} --server.headless=true ;; esac"]
