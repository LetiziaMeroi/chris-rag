FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-runtime.txt .

RUN pip install --no-cache-dir \
    -r requirements-runtime.txt

COPY src ./src

RUN mkdir -p /app/logs

EXPOSE 8000
EXPOSE 8501

CMD ["python", "-m", "src.query.query_orchestrator", "--help"]