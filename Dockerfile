# The service image intentionally installs only the base package plus the
# service extra. Pulling the `dense` extra would add roughly 2 GB of PyTorch to
# an image whose default configuration is BM25 and does not import it.
FROM python:3.12-slim

WORKDIR /app

# Copy metadata first so the dependency layer is cached independently of source
# changes: editing a retriever should not reinstall numpy.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN pip install --no-cache-dir ".[service]"

ENV RAG_DATASET=scifact \
    RAG_SYSTEM=bm25

EXPOSE 8000

# The corpus downloads on first start and is cached in the volume; the health
# check tolerates that by allowing a generous start period rather than a long
# interval, so a genuinely wedged container is still detected quickly.
HEALTHCHECK --interval=15s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health').read()"

CMD ["uvicorn", "rag_ablations.service:app", "--host", "0.0.0.0", "--port", "8000"]
