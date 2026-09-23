"""Ad-hoc semantic search against a built index: uv run kb-search "domanda" --model bge-m3"""
from __future__ import annotations

import argparse
import logging

from kb.index import build


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Query a vector index")
    ap.add_argument("query")
    ap.add_argument("--extractor", default="docling")
    ap.add_argument("--model", default="bge-m3")
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    store = build(args.extractor, args.model)
    for i, (doc, score) in enumerate(store.similarity_search_with_relevance_scores(args.query, k=args.k), 1):
        m = doc.metadata
        print(f"\n#{i}  score={score:.3f}  {m['doc_code']} p{m['page_start']}-{m['page_end']}  [{m['section']}]")
        print(doc.page_content[:600])


if __name__ == "__main__":
    main()
