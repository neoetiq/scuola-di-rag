"""Interactive retrieval tester: type a question, see the top chunks, repeat; 'quit' to exit.

    uv run kb-query                       # docling corpus, bge-m3
    uv run kb-query --extractor mineru --model qwen3-4b -k 3
"""
from __future__ import annotations

import argparse
import logging
import textwrap

from kb.config import EMBEDDING_MODELS
from kb.index import build

EXIT_WORDS = {"quit", "exit", "q", "esci"}


def show_hits(hits, width: int = 100) -> None:
    for i, (doc, score) in enumerate(hits, 1):
        m = doc.metadata
        pages = f"p.{m['page_start']}" if m["page_start"] == m["page_end"] else f"p.{m['page_start']}-{m['page_end']}"
        print(f"\n{'=' * width}")
        print(f"#{i}  score={score:.3f}  [{m['language']}] {m['source']}  {pages}")
        print(f"    sezione: {m['section']}")
        print("-" * width)
        body = doc.page_content.split("\n", 1)[1] if "\n" in doc.page_content else doc.page_content
        for line in body.strip().splitlines():
            print(textwrap.fill(line, width) if not line.startswith("|") else line[:width])
    print(f"{'=' * width}\n")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Interactive retrieval tester")
    ap.add_argument("--extractor", default="docling", help="corpus to search (docling, mineru, pymupdf4llm, langchain)")
    ap.add_argument("--model", default="bge-m3", help=f"embedding model key or spec ({', '.join(EMBEDDING_MODELS)})")
    ap.add_argument("-k", type=int, default=3, help="number of chunks to show")
    ap.add_argument("--width", type=int, default=100)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    print(f"Carico l'indice {args.extractor} / {args.model} ...")
    store = build(args.extractor, args.model)
    n = store._collection.count()
    print(f"Pronto: {n} chunk indicizzati. Scrivi una domanda, oppure 'quit' per uscire.\n")

    while True:
        try:
            query = input("domanda> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query.lower() in EXIT_WORDS:
            break
        hits = store.similarity_search_with_relevance_scores(query, k=args.k)
        show_hits(hits, args.width)
    print("Ciao.")


if __name__ == "__main__":
    main()
