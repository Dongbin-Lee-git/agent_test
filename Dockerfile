# 빌드 스테이지: 의존성 설치
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# 빌드에 필요한 툴 + uv 설치
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential gcc libffi-dev && \
    pip install --no-cache-dir uv && \
    rm -rf /var/lib/apt/lists/*

# 의존성 파일만 먼저 복사
COPY pyproject.toml ./

# .venv에 의존성 설치 (dev 의존성 제외)
RUN uv sync --no-dev --no-cache

# 애플리케이션 코드 복사
# (테스트/샘플 파일이 많으면 필요한 디렉토리만 선택해서 COPY 해도 됨)
COPY . .

# -------------------------------------------------------
# backend 런타임 스테이지
# -------------------------------------------------------
FROM python:3.12-slim AS backend

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN useradd -m appuser

# 빌더에서 만든 가상환경만 복사
COPY --from=builder /app/.venv /app/.venv

# 애플리케이션 필요한 파일만 복사
COPY --from=builder /app/main.py /app/main.py
COPY --from=builder /app/app /app/app
COPY --from=builder /app/infra /app/infra
COPY --from=builder /app/pyproject.toml /app/pyproject.toml

RUN chown -R appuser:appuser /app

# venv를 기본 Python으로 사용
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

USER appuser

EXPOSE 8001

CMD ["/app/.venv/bin/uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]

# -------------------------------------------------------
# frontend 런타임 스테이지
# -------------------------------------------------------
FROM python:3.12-slim AS frontend

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN useradd -m appuser

# 빌더에서 만든 가상환경만 복사
COPY --from=builder /app/.venv /app/.venv

# 프론트엔드 필요한 파일만 복사
COPY --from=builder /app/infra /app/infra
COPY --from=builder /app/pyproject.toml /app/pyproject.toml

# venv를 기본 Python으로 사용
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

USER appuser

EXPOSE 8002

CMD ["/app/.venv/bin/streamlit", "run", "infra/frontend/ui.py", "--server.port", "8002", "--server.address", "0.0.0.0"]