from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PDF_DIR = ROOT / "pdf"
MD_DIR = ROOT / "data" / "md"
INDEX_DIR = ROOT / "data" / "index"
EVAL_DIR = ROOT / "eval"

PAGE_MARKER = "<!-- page: {n} -->"

# Embedding models under test. Prefix decides the backend:
#   hf:<repo>      -> sentence-transformers (runs on Apple Silicon via MPS)
#   ollama:<name>  -> local Ollama server
EMBEDDING_MODELS: dict[str, str] = {
    "qwen3-0.6b": "hf:Qwen/Qwen3-Embedding-0.6B",
    "qwen3-4b": "hf:Qwen/Qwen3-Embedding-4B",
    "bge-m3": "hf:BAAI/bge-m3",
    "e5-large": "hf:intfloat/multilingual-e5-large-instruct",
}

# Query-side instructions for instruction-tuned models (document side stays raw).
QUERY_PREFIX: dict[str, str] = {
    "hf:Qwen/Qwen3-Embedding-0.6B": "Instruct: Given a question about a stove or fireplace manual, retrieve the passage that answers it\nQuery: ",
    "hf:Qwen/Qwen3-Embedding-4B": "Instruct: Given a question about a stove or fireplace manual, retrieve the passage that answers it\nQuery: ",
    "hf:intfloat/multilingual-e5-large-instruct": "Instruct: Given a question about a stove or fireplace manual, retrieve the passage that answers it\nQuery: ",
}

CHUNK_SIZE = 1200  # characters (~300 tokens); fits the 512-token limit of e5
CHUNK_OVERLAP = 150

# Parent documents (structure-aware): a chapter is one parent when it fits PARENT_MAX_TOKENS,
# otherwise its sections become parents. Tiny parents are merged, oversize ones are split.
PARENT_MAX_TOKENS = 2000
PARENT_MIN_TOKENS = 100
PARENT_TOKENIZER = "Qwen/Qwen3-Embedding-4B"
# Cover page and table of contents list every section title: they attract lexical matches but never
# contain the answer, so their chunks are not indexed (the parent is still stored).
INDEX_FRONT_MATTER = False
