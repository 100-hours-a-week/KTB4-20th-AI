FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.12.16 /uv /bin/uv

WORKDIR /app

ENV UV_PYTHON_DOWNLOADS=never

COPY . .
RUN uv sync --locked --no-dev --python /usr/local/bin/python

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
