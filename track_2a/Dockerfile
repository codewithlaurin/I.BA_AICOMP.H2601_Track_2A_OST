# Judge image: linux/amd64, CPU only, no downloads at runtime.
#   docker build --platform linux/amd64 -t my-solution:dev .
#   docker run --rm -e BASE_URL -e API_KEY -e LLM_NAME -v ...:/data -v ...:/output \
#     my-solution:dev --input /data/cases.jsonl --output /output/predictions.jsonl
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TMPDIR=/tmp \
    XDG_CACHE_HOME=/tmp/cache \
    HF_HOME=/tmp/huggingface \
    TORCH_HOME=/tmp/torch

WORKDIR /app

# Versions come from uv.lock. PyTorch is installed first from the CPU index: the default
# Linux wheel pulls about 3 GB of CUDA libraries that a CPU-only judge never uses.
COPY pyproject.toml uv.lock ./
RUN pip install uv==0.9.* \
 && uv export --frozen --no-dev --no-hashes --no-emit-project --format requirements-txt -o /tmp/constraints.txt \
 && TORCH=$(grep -E '^torch==' /tmp/constraints.txt | cut -d' ' -f1 | cut -d';' -f1) \
 && pip install --index-url https://download.pytorch.org/whl/cpu "$TORCH" \
 && pip uninstall -y uv

COPY src/claimcheck ./src/claimcheck
RUN pip install --constraint /tmp/constraints.txt . \
 && python -c "import torch; import importlib.metadata as m; assert torch.version.cuda is None and '+cpu' in torch.__version__, torch.__version__; assert not any(d.metadata['Name'].lower().startswith(('nvidia-', 'triton')) for d in m.distributions())" \
 && pip check

# Page retrieval needs intfloat/multilingual-e5-small. It is baked into the image at build
# time; at runtime the Hugging Face libraries are switched to offline mode.
ENV EMBEDDING_MODEL=/opt/models/multilingual-e5-small
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('intfloat/multilingual-e5-small', device='cpu').save('$EMBEDDING_MODEL')" \
 && rm -rf /tmp/huggingface
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

RUN mkdir -p /data /output /tmp/cache /tmp/huggingface /tmp/torch \
 && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('$EMBEDDING_MODEL', device='cpu', local_files_only=True); assert model.encode(['build-time offline check']).shape[0] == 1"

# Apertus is reached over the OpenAI-compatible API: BASE_URL, API_KEY and LLM_NAME are
# injected at `docker run`; no key is baked into the image.
ENTRYPOINT ["claimcheck"]
