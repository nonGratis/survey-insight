FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    SERVICE=web

WORKDIR /app

COPY requirements.txt constraints.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

# Only local Docker reads HEALTHCHECK; Cloud Run runs its own probes. Python is already in the
# image, so the check needs no curl and no apt layer.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import os, urllib.request as r; path = '/health' if os.environ.get('SERVICE') in ('api', 'worker') else '/_stcore/health'; r.urlopen('http://localhost:' + os.environ.get('PORT', '8080') + path, timeout=4)"

CMD ["sh", "-c", "case \"$SERVICE\" in api) exec python -m uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080} ;; worker) exec python -m uvicorn worker.main:app --host 0.0.0.0 --port ${PORT:-8080} ;; web|*) exec streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-8080} --server.headless=true ;; esac"]
