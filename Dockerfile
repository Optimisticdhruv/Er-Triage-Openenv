# Stage 1: builder
FROM python:3.10-slim as builder
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: production
FROM python:3.10-slim
WORKDIR /app
RUN useradd -m -u 1000 appuser
COPY --from=builder /root/.local /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH
COPY app/ ./app/
COPY scenarios/ ./scenarios/
COPY static/ ./static/
COPY openenv.yaml .
COPY inference.py .
RUN mkdir -p baseline/results && chown -R appuser:appuser /app
USER appuser
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=7860
ENV ENVIRONMENT=production
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
