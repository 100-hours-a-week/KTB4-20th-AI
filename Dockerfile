FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.12.16 /uv /bin/uv

WORKDIR /app

ENV UV_PYTHON_DOWNLOADS=never

COPY . .
RUN uv sync --locked --no-dev --python /usr/local/bin/python

ENV PATH="/app/.venv/bin:$PATH"

# 배포 이미지의 커밋 SHA. Sentry release 구분에 쓴다. 커밋마다 바뀌므로 의존성 설치 뒤에 둬서 빌드 캐시를 지킨다
ARG GIT_SHA=""
ENV GIT_SHA=$GIT_SHA

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
