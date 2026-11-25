# rag_core.py — shared helpers for Block 7 (RAG), Days 62-70.
#
# Put this file in the SAME folder as the day scripts, sample_corpus.md and your .env
# (the one with GEMINI_API_KEY=...). The day scripts import from here.
#
# What lives here (plumbing you'd normally get from a library):
#   - Gemini calls with rate-limit handling:      embed_texts(), generate(), generate_json()
#   - an on-disk embedding cache (so reruns don't re-spend your free quota)
#   - three chunking strategies:                   chunk_fixed(), chunk_by_structure()
#   - cosine similarity, a small BM25, and rank fusion (RRF) for hybrid search
#   - the shared test questions, each with the corpus section that answers it

import os
import re
import json
import math
import time
import hashlib
from pathlib import Path

import numpy as np
import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import types, errors

load_dotenv()

# ---------------------------------------------------------------- settings ---
GEN_MODEL = "gemini-3.5-flash-lite"      # text generation + judging (this name worked for you in Block 6)
EMBED_MODEL = "gemini-embedding-001"     # if this 404s, rag_core tries to find a current embedding model itself
EMBED_DIM = 768                          # smaller vectors = faster + cheaper; we normalise them ourselves
CORPUS_PATH = "sample_corpus.md"         # swap in your own .md/.txt file here for the capstone
CACHE_PATH = "embedding_cache.json"
LLM_MIN_INTERVAL = 6.0                   # seconds between LLM calls, keeps you under free-tier limits

_client = None
_last_llm_call = 0.0
_cache = None


def get_client():
    """Create the Gemini client on first use (so importing this file never crashes)."""
    global _client
    if _client is None:
        key = os.getenv("GEMINI_API_KEY")
        if not key or key.startswith("paste_your"):
            raise ValueError("No GEMINI_API_KEY found. Put it in the .env file next to these scripts.")
        _client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=60000))  # 60s: fail instead of freezing
    return _client


# ------------------------------------------------------- retry / throttling ---
def _is_retryable(e):
    return e.code == 429 or (e.code is not None and e.code >= 500)


def _throttle():
    global _last_llm_call
    wait = LLM_MIN_INTERVAL - (time.time() - _last_llm_call)
    if wait > 0:
        time.sleep(wait)
    _last_llm_call = time.time()


def _with_retries(fn, what, max_attempts=6):
    """Call fn(); on rate limits / server errors wait and try again, otherwise raise."""
    for attempt in range(max_attempts):
        try:
            return fn()
        except errors.APIError as e:
            if _is_retryable(e) and attempt < max_attempts - 1:
                wait = 10 * (attempt + 1)
                print(f"  [{what}] API busy or rate-limited (status {e.code}), waiting {wait}s...")
                time.sleep(wait)
                continue
            raise
        except httpx.TimeoutException:
            if attempt < max_attempts - 1:
                print(f"  [{what}] request timed out, trying again...")
                continue
            raise


# --------------------------------------------------------------- generation ---
def generate(prompt, system=None, temperature=0.0):
    """Plain text generation."""
    def call():
        _throttle()
        resp = get_client().models.generate_content(
            model=GEN_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(system_instruction=system, temperature=temperature),
        )
        return resp.text or ""
    return _with_retries(call, "generate")


def generate_json(prompt, schema, system=None):
    """Generation that must match a Pydantic schema (structured output, same idea as Day 54)."""
    def call():
        _throttle()
        resp = get_client().models.generate_content(
            model=GEN_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                response_schema=schema,
                temperature=0.0,
            ),
        )
        parsed = resp.parsed
        return parsed if parsed is not None else schema.model_validate_json(resp.text)
    return _with_retries(call, "generate_json")


# --------------------------------------------------------------- embeddings ---
def _load_cache():
    global _cache
    if _cache is None:
        p = Path(CACHE_PATH)
        _cache = json.loads(p.read_text()) if p.exists() else {}
    return _cache


def _save_cache():
    Path(CACHE_PATH).write_text(json.dumps(_cache))


def _cache_key(text, task_type):
    return hashlib.sha256(f"{EMBED_MODEL}|{EMBED_DIM}|{task_type}|{text}".encode()).hexdigest()


def _pick_embedding_model():
    """Ask the API which models support embeddings (used only if the default name 404s)."""
    names = []
    for m in get_client().models.list():
        if "embedContent" in (getattr(m, "supported_actions", None) or []):
            names.append(m.name.replace("models/", ""))
    names.sort(key=lambda n: ("embedding" not in n, n))   # prefer names containing 'embedding'
    return names[0] if names else None


def _embed_batch(texts, task_type):
    global EMBED_MODEL
    switched = False
    for attempt in range(6):
        try:
            resp = get_client().models.embed_content(
                model=EMBED_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=EMBED_DIM),
            )
            return [list(e.values) for e in resp.embeddings]
        except errors.APIError as e:
            if e.code == 404 and not switched:
                new_name = _pick_embedding_model()
                if new_name and new_name != EMBED_MODEL:
                    print(f"  Embedding model '{EMBED_MODEL}' not available, switching to '{new_name}'")
                    EMBED_MODEL = new_name
                    switched = True
                    continue
            if _is_retryable(e) and attempt < 5:
                wait = 10 * (attempt + 1)
                print(f"  [embed] API busy or rate-limited (status {e.code}), waiting {wait}s...")
                time.sleep(wait)
                continue
            raise
        except httpx.TimeoutException:
            if attempt < 5:
                print("  [embed] request timed out, trying again...")
                continue
            raise


def normalize(arr):
    arr = np.asarray(arr, dtype=float)
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / np.maximum(norms, 1e-12)


def embed_texts(texts, task_type="RETRIEVAL_DOCUMENT", batch_size=50):
    """Embed a list of strings -> numpy array (n, dim), L2-normalised.

    task_type: RETRIEVAL_DOCUMENT for chunks you store, RETRIEVAL_QUERY for the user's
    question, SEMANTIC_SIMILARITY when you just compare sentences with each other.
    Results are cached on disk, so rerunning a script costs no extra API calls.
    """
    cache = _load_cache()
    keys = [_cache_key(t, task_type) for t in texts]
    missing = {}
    for k, t in zip(keys, texts):
        if k not in cache and k not in missing:
            missing[k] = t
    items = list(missing.items())
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        vectors = _embed_batch([t for _, t in batch], task_type)
        for (k, _), v in zip(batch, vectors):
            cache[k] = v
        _save_cache()
        time.sleep(1)
    # (keys were built with the model name at call time; if we switched models mid-call, rebuild)
    keys = [_cache_key(t, task_type) for t in texts]
    for k, t in zip(keys, texts):
        if k not in cache:
            cache[k] = _embed_batch([t], task_type)[0]
    return normalize([cache[k] for k in keys])


# -------------------------------------------------------------- similarity ---
def cosine_manual(a, b):
    """Cosine similarity with plain Python: dot(a, b) / (|a| * |b|)."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


def top_k_cosine(query_vec, matrix, k=3):
    """Both inputs already normalised, so cosine similarity == dot product."""
    scores = matrix @ query_vec
    order = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in order]


# ------------------------------------------------------------ corpus + chunks ---
def load_corpus(path=None):
    return Path(path or CORPUS_PATH).read_text(encoding="utf-8")


def split_sections(text):
    """Find each '## Heading' section: [{title, start, body_start, end}, ...]."""
    matches = list(re.finditer(r"^## (.+)$", text, flags=re.M))
    sections = []
    if not matches:
        return [{"title": "Document", "start": 0, "body_start": 0, "end": len(text)}]
    if matches[0].start() > 0:
        sections.append({"title": "Introduction", "start": 0, "body_start": 0, "end": matches[0].start()})
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append({"title": m.group(1).strip(), "start": m.start(), "body_start": m.end(), "end": end})
    return sections


def _sections_overlapping(sections, start, end):
    return [s["title"] for s in sections if s["start"] < end and start < s["end"]]


def _make_chunk(text, start, end, sections):
    return {"text": text, "start": start, "end": end, "sections": _sections_overlapping(sections, start, end)}


def chunk_fixed(text, sections, size=400, overlap=0):
    """Naive baseline: cut every `size` characters, stepping forward by size - overlap.
    Ignores sentence and section boundaries on purpose (that's the weakness to observe)."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    step = size - overlap
    chunks, i = [], 0
    while i < len(text):
        piece = text[i:i + size]
        if piece.strip():
            chunks.append(_make_chunk(piece, i, i + len(piece), sections))
        if i + size >= len(text):
            break
        i += step
    return chunks


def chunk_by_structure(text, sections, max_chars=900):
    """Structure-aware: stay inside one section, pack whole sentences up to max_chars,
    and put the section title on top of each chunk so the chunk explains itself."""
    chunks = []
    for sec in sections:
        body = text[sec["body_start"]:sec["end"]]
        offset = sec["body_start"]
        # sentence boundaries: whitespace after . ! ? followed by a capital letter
        cuts = [0] + [m.end() for m in re.finditer(r"(?<=[.!?])\s+(?=[A-Z\"(])", body)] + [len(body)]
        spans = []
        for a, b in zip(cuts, cuts[1:]):
            seg = body[a:b]
            if seg.strip():
                s = offset + a + (len(seg) - len(seg.lstrip()))
                e = offset + a + len(seg.rstrip())
                spans.append((s, e))
        prefix = f"{sec['title']}\n\n"
        current = []

        def flush():
            if current:
                s, e = current[0][0], current[-1][1]
                chunks.append(_make_chunk(prefix + text[s:e], s, e, sections))

        for s, e in spans:
            projected = len(prefix) + (e - (current[0][0] if current else s))
            if current and projected > max_chars:
                flush()
                current = []
            current.append((s, e))
        flush()
    return chunks


STRATEGIES = {
    "fixed_400":         lambda text, secs: chunk_fixed(text, secs, size=400, overlap=0),
    "fixed_400_overlap": lambda text, secs: chunk_fixed(text, secs, size=400, overlap=80),
    "structure":         lambda text, secs: chunk_by_structure(text, secs, max_chars=900),
}


def chunk_with_strategy(text, strategy, sections=None):
    sections = sections or split_sections(text)
    chunks = STRATEGIES[strategy](text, sections)
    for i, c in enumerate(chunks):
        c["id"] = i
        c["strategy"] = strategy
    return chunks


# ----------------------------------------------------- keyword search (BM25) ---
STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "is", "are", "was", "were", "be", "by",
    "it", "its", "that", "this", "with", "as", "at", "from", "can", "do", "does", "how", "what", "why",
    "which", "who", "when", "where", "if", "than", "then", "so", "such", "they", "their", "them", "not",
}


def tokenize(text):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


class BM25:
    """Classic keyword ranking. Rewards rare words that appear often in a short document."""

    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tokens = [tokenize(d) for d in docs]
        self.doc_len = np.array([len(t) for t in self.tokens], dtype=float)
        self.avg_len = self.doc_len.mean() if len(docs) else 0.0
        df = {}
        for toks in self.tokens:
            for w in set(toks):
                df[w] = df.get(w, 0) + 1
        n = len(docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.tf = []
        for toks in self.tokens:
            d = {}
            for w in toks:
                d[w] = d.get(w, 0) + 1
            self.tf.append(d)

    def scores(self, query):
        q = tokenize(query)
        out = np.zeros(len(self.tokens))
        for i, tf in enumerate(self.tf):
            for w in q:
                if w in tf:
                    f = tf[w]
                    denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avg_len)
                    out[i] += self.idf[w] * f * (self.k1 + 1) / denom
        return out


def rank_from_scores(scores):
    """Document indices, best first (documents with zero keyword score still get a rank)."""
    return [int(i) for i in np.argsort(-np.asarray(scores), kind="stable")]


def rrf_fuse(rankings, k=60):
    """Reciprocal Rank Fusion: each list votes 1/(k + rank) for every document it ranks.
    Returns document indices, best first. No score scaling needed, which is why it's popular."""
    fused = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            fused[doc] = fused.get(doc, 0.0) + 1.0 / (k + rank)
    return [d for d, _ in sorted(fused.items(), key=lambda kv: -kv[1])]


# ------------------------------------------------ shared test questions (with answers) ---
# (question, title of the corpus section that answers it) — lets us measure retrieval objectively.
TEST_QUESTIONS = [
    ("Why do cities flood more than rural areas after heavy rain?", "Why Cities Flood"),
    ("What is the difference between pluvial and fluvial flooding?", "Types of Flooding"),
    ("How can a city absorb stormwater using green infrastructure?", "Sustainable Drainage and Sponge Cities"),
    ("How do forecasters predict heavy rain a few hours ahead?", "Rainfall Nowcasting and Early Warning"),
    ("What does double materiality mean?", "CSRD, ESRS and Double Materiality"),
    ("What are Scope 3 emissions?", "Greenhouse Gas Scopes"),
    ("What are warning signs of greenwashing?", "Greenwashing"),
    ("How is an activity classified as environmentally sustainable in the EU?", "The EU Taxonomy"),
    ("Why might a rainfall model trained in one city fail in another?", "Rainfall Nowcasting and Early Warning"),
    ("Where do emission factors come from?", "Carbon Accounting Data Quality"),
]


def target_rank(chunks_in_rank_order, target_title):
    """1-based position of the first chunk that overlaps the target section, or None."""
    for pos, c in enumerate(chunks_in_rank_order, start=1):
        if target_title in c["sections"]:
            return pos
    return None


def summarize_ranks(ranks):
    """hit@1, hit@3 and mean reciprocal rank from a list of ranks (None = not found)."""
    n = len(ranks)
    hit1 = sum(1 for r in ranks if r == 1) / n
    hit3 = sum(1 for r in ranks if r is not None and r <= 3) / n
    mrr = sum((1 / r) for r in ranks if r) / n
    return hit1, hit3, mrr
