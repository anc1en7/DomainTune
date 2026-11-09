FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md uv.lock ./
COPY src ./src
COPY app ./app
RUN pip install --no-cache-dir uv==0.11.8 && uv sync --locked --no-dev
EXPOSE 8000
CMD [".venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
