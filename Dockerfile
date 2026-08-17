# Build for amd64 explicitly so it can ship to AWS ECS from ARM Macs.
# (Lesson 11 will pin --platform linux/amd64 on the build command.)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Build args, not baked-in defaults: an unset PIP_INDEX_URL leaves pip on
# PyPI, so builds outside mainland China behave exactly as before. Builds
# inside China pass a mirror, where a direct pypi.org connection routinely
# times out rather than merely being slow.
ARG PIP_INDEX_URL=
ARG PIP_TRUSTED_HOST=
ENV PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST}

COPY requirements.txt .
RUN pip install -r requirements.txt

# Warm the Chroma default embedding model (~80 MB, all-MiniLM-L6-v2) into the
# image so students' first /rag-ask isn't a 30-60s HuggingFace download that
# often times out on flaky classroom wifi. Runs a throwaway add+query which
# forces the embedding function to fetch + cache the model files.
RUN python -c "import chromadb; c = chromadb.EphemeralClient(); \
    col = c.create_collection('warmup'); \
    col.add(documents=['warmup'], ids=['1']); \
    col.query(query_texts=['warmup'], n_results=1); \
    print('embedding model cached')"

COPY app/ ./app/
COPY data/ ./data/
COPY scripts/ ./scripts/

# Build the vector index into the image. The index is a ~12 MB binary sqlite
# file that is git-ignored, so it cannot be COPYed in — anyone who clones the
# repo would get an image with no index and every /rag-ask would 500. Building
# it here makes the image self-contained and reproducible from source alone.
# The embedding model is already cached by the warm-up step above, so this is
# CPU-only and offline.
RUN python scripts/build_index.py

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
