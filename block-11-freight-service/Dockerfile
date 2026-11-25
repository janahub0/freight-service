FROM python:3.12-slim

WORKDIR /app

# packages first, so docker can reuse this layer when only the code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rag_core.py rag_pipeline.py agent.py app.py mcp_server.py sample_corpus.md ./

# the embedding cache and the memory file are saved here
ENV DATA_DIR=/data
RUN mkdir -p /data

# hosts like Render and Azure give the port as $PORT
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
