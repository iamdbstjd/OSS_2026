# syntax=docker/dockerfile:1.7@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e
FROM python:3.12.13-slim-bookworm@sha256:4766d8b510c428e595d74b9cc5bbb2fae8e26316fffb4adc89908d79aacd58a2

ARG UV_VERSION=0.11.29

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

RUN python -m pip install --no-cache-dir "uv==${UV_VERSION}" \
    && groupadd --system ragplan \
    && useradd --system --gid ragplan --home-dir /app ragplan

COPY pyproject.toml uv.lock README.md LICENSE docker-compose.yml ./
COPY src ./src
COPY configs ./configs
COPY benchmark/configs ./benchmark/configs
COPY benchmark/manifests ./benchmark/manifests
COPY benchmark/qrels ./benchmark/qrels
COPY benchmark/NOTICE.md ./benchmark/NOTICE.md

RUN uv sync --frozen --no-dev --group graph-extraction \
    && chown -R ragplan:ragplan /app

USER ragplan

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=5 \
  CMD [".venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"]

CMD [".venv/bin/uvicorn", "ragplan.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
