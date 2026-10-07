FROM python:3.14-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements-evaluation.txt ./
RUN pip install --no-cache-dir -r requirements-evaluation.txt
COPY main.py ./
COPY scripts/benchmark.py scripts/apertus_nli.py ./scripts/
ENTRYPOINT ["python", "main.py"]
