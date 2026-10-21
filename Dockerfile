# syntax=docker/dockerfile:1.7
# Serving image for the API and the learner: ONNX Runtime, no PyTorch.
FROM python:3.13-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN useradd --uid 10001 --create-home app
WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY pyproject.toml ./
COPY src/ src/
RUN pip install --no-deps .
# model bundles are versioned directories; MCI_BUNDLE picks one (set by the Kubernetes ConfigMap)
COPY bundles/ /app/bundles/
USER 10001
ENV MCI_BUNDLE=/app/bundles/current MCI_PRECISION=int8 PORT=8000
EXPOSE 8000 9100
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/healthz')"
CMD ["sh", "-c", "mci serve --port ${PORT} --workers ${WEB_CONCURRENCY:-1}"]
