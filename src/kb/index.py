"""Build one Chroma index per (extractor, embedding model)."""
from __future__ import annotations

import argparse
import logging
import shutil
import time
from pathlib import Path

from langchain_chroma import Chroma

from kb.chunk import load_corpus
from kb.config import EMBEDDING_MODELS, INDEX_DIR
from kb.embeddings import get_embeddings

log = logging.getLogger("kb.index")


def index_path(extractor: str, model: str) -> Path:
    return INDEX_DIR / extractor / model.replace("/", "_").replace(":", "_")


def build(extractor: str, model: str, force: bool = False) -> Chroma:
    path = index_path(extractor, model)
    emb = get_embeddings(model)
    if path.exists() and not force:
        log.info("reuse index %s", path)
        return Chroma(persist_directory=str(path), embedding_function=emb)
    shutil.rmtree(path, ignore_errors=True)
    docs = load_corpus(extractor)
    t0 = time.time()
    store = Chroma.from_documents(docs, emb, persist_directory=str(path), collection_metadata={"hnsw:space": "cosine"})
    log.info("%s / %s: %d chunks embedded in %.0fs", extractor, model, len(docs), time.time() - t0)
    return store


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Build vector indexes")
    ap.add_argument("--extractor", default="docling")
    ap.add_argument("--model", action="append", help="model key or spec; repeatable (default: all configured)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    for m in args.model or list(EMBEDDING_MODELS):
        build(args.extractor, m, args.force)


if __name__ == "__main__":
    main()
