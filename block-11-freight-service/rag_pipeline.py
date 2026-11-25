# Day 66 — First end-to-end RAG pipeline
# ingest -> chunk -> embed -> store (Qdrant) -> retrieve -> generate an answer WITH sources.
# Also shows that the model refuses when the context doesn't contain the answer.
# Other days import build_index / retrieve / answer from this file.
# Needs: rag_core.py, sample_corpus.md, .env in the same folder.

import os
from qdrant_client import QdrantClient, models
from rag_core import *

SYSTEM_RAG = (
    "You answer questions using ONLY the numbered context passages provided. "
    "Cite the passages you used like [1] or [2]. "
    "If the context does not contain the answer, reply exactly: "
    "\"I can't answer that from the provided documents.\" Do not use outside knowledge."
)


class RagIndex:
    """Chunks + their Qdrant collection, built once and reused."""

    def __init__(self, chunks, client, collection):
        self.chunks, self.client, self.collection = chunks, client, collection


def build_index(path=None, strategy="structure", collection="rag_docs"):
    text = load_corpus(path)
    sections = split_sections(text)
    chunks = chunk_with_strategy(text, strategy, sections)
    for c in chunks:
        c["section"] = c["sections"][0] if c["sections"] else "Document"
    vectors = embed_texts([c["text"] for c in chunks], task_type="RETRIEVAL_DOCUMENT")

    # in docker compose the qdrant container is used, on your laptop it stays in memory
    url = os.getenv("QDRANT_URL")
    client = QdrantClient(url=url) if url else QdrantClient(":memory:")
    if url:
        # the qdrant container can need a few seconds to start, so try a few times
        for attempt in range(15):
            try:
                client.get_collections()
                break
            except Exception:
                print("  waiting for qdrant to start...")
                time.sleep(2)
    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(collection, vectors_config=models.VectorParams(
        size=EMBED_DIM, distance=models.Distance.COSINE))
    client.upsert(collection, points=[
        models.PointStruct(id=c["id"], vector=vectors[i].tolist(),
                           payload={"text": c["text"], "section": c["section"]})
        for i, c in enumerate(chunks)])
    return RagIndex(chunks, client, collection)


def retrieve(index, question, k=3):
    """Return the k most similar chunks as dicts with text, section, score."""
    qv = embed_texts([question], task_type="RETRIEVAL_QUERY")[0].tolist()
    res = index.client.query_points(index.collection, query=qv, limit=k)
    return [{"id": p.id, "text": p.payload["text"], "section": p.payload["section"],
             "score": p.score} for p in res.points]


def format_context(passages):
    return "\n\n".join(f"[{i}] ({p['section']})\n{p['text']}" for i, p in enumerate(passages, start=1))


def answer(question, passages):
    prompt = f"Context passages:\n\n{format_context(passages)}\n\nQuestion: {question}\n\nAnswer:"
    return generate(prompt, system=SYSTEM_RAG)


def ask(index, question, k=3, verbose=True):
    passages = retrieve(index, question, k)
    reply = answer(question, passages)
    if verbose:
        print(f"\nQ: {question}")
        print(f"A: {reply.strip()}")
        print("Sources retrieved:")
        for i, p in enumerate(passages, start=1):
            print(f"  [{i}] {p['section']}  (similarity {p['score']:.3f})")
    return reply, passages
