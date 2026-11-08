# FinLens image. Works locally and as a Hugging Face "Docker" Space (which expects port 7860
# and a non-root user with uid 1000). The sample corpus is generated and indexed at build time,
# so the container starts instantly and needs no external services: it runs in rule-based mode
# (FINLENS_LLM_PROVIDER=none). Point it at an Ollama server to enable the local-LLM helpers.
FROM python:3.12-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    FINLENS_LLM_PROVIDER=none

WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .
RUN python scripts/make_sample_data.py && python scripts/ingest.py

# Hosts like Render inject $PORT; Hugging Face Spaces and local Docker use the 7860 default.
EXPOSE 7860
CMD ["sh", "-c", "uvicorn finlens.api:app --host 0.0.0.0 --port ${PORT:-7860}"]
