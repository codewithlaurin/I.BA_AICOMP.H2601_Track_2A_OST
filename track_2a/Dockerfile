# Judge image: linux/amd64, CPU only, no downloads at runtime.
#   docker build --platform linux/amd64 -t apertus-claimcheck .
#   docker run --rm -e BASE_URL -e API_KEY -e LLM_NAME -v ...:/data -v ...:/output \
#     apertus-claimcheck --input /data/cases.jsonl --output /output/predictions.jsonl
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Versions come from uv.lock. PyTorch is installed first from the CPU index: the default
# Linux wheel pulls about 3 GB of CUDA libraries that a CPU-only judge never uses.
COPY pyproject.toml uv.lock ./
RUN pip install uv==0.9.* \
 && uv export --frozen --no-dev --no-hashes --no-emit-project --format requirements-txt -o /tmp/constraints.txt \
 && TORCH=$(grep -E '^torch==' /tmp/constraints.txt | cut -d' ' -f1 | cut -d';' -f1) \
 && pip install --index-url https://download.pytorch.org/whl/cpu "$TORCH" \
 && pip uninstall -y uv

COPY src ./src
RUN pip install --constraint /tmp/constraints.txt . \
 && python -c "import torch; assert not torch.cuda.is_available() and '+cpu' in torch.__version__, torch.__version__"

# Page retrieval needs intfloat/multilingual-e5-small. It is baked into the image at build
# time; at runtime the Hugging Face libraries are switched to offline mode.
ENV EMBEDDING_MODEL=/opt/models/multilingual-e5-small
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('intfloat/multilingual-e5-small', device='cpu').save('$EMBEDDING_MODEL')" \
 && rm -rf /root/.cache/huggingface
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# Apertus is reached over the OpenAI-compatible API: BASE_URL, API_KEY and LLM_NAME are
# injected at `docker run`; no key is baked into the image.
ENTRYPOINT ["claimcheck"]
