"""Interactive retrieval tester: type a question, see the top results, repeat; 'quit' to exit.

    uv run kb-query                               # Qdrant, BGE-M3 dense+sparse (A1), top 3 chunks
    uv run kb-query --parents                     # same search, results aggregated into parent documents
    uv run kb-query --pipeline B1 --doc H072521IT0
    uv run kb-query --engine chroma --model bge-m3
"""
from __future__ import annotations

import argparse
import logging
import textwrap

EXIT_WORDS = {"quit", "exit", "q", "esci"}


def _print_body(text: str, width: int, max_chars: int | None = None) -> None:
    if max_chars and len(text) > max_chars:
        text = text[:max_chars] + f"\n[... troncato, {len(text) - max_chars} caratteri omessi]"
    for line in text.strip().splitlines():
        print(textwrap.fill(line, width) if not line.startswith("|") else line[:width])


def show_chunks(hits: list[tuple[dict, str, float | None]], width: int) -> None:
    for i, (m, text, score) in enumerate(hits, 1):
        pages = f"p.{m['page_start']}" if m["page_start"] == m["page_end"] else f"p.{m['page_start']}-{m['page_end']}"
        print(f"\n{'=' * width}")
        sc = f"score={score:.3f}  " if score is not None else ""
        print(f"#{i}  {sc}[{m.get('language', '')}] {m.get('source', '')}  {pages}")
        print(f"    sezione: {m.get('section', '')}")
        if m.get("parent_id"):
            print(f"    parent:  {m['parent_id']} ({m.get('parent_tokens', '?')} token)")
        print("-" * width)
        _print_body(text.split("\n", 1)[1] if "\n" in text else text, width)
    print(f"{'=' * width}\n")


def show_parents(parents: list[dict], width: int, max_chars: int) -> None:
    total = sum(p["tokens"] for p in parents)
    for i, p in enumerate(parents, 1):
        pages = f"p.{p['page_start']}" if p["page_start"] == p["page_end"] else f"p.{p['page_start']}-{p['page_end']}"
        print(f"\n{'=' * width}")
        print(f"PARENT #{i}  {p['parent_id']}  [{p['language']}] {pages}  {p['tokens']} token  "
              f"(chunk trovati: {p['hit_chunks']}, miglior rango: {p['best_rank']})")
        print(f"    {p['title']}")
        print("-" * width)
        _print_body(p["text"], width, max_chars)
    print(f"{'=' * width}\ncontesto totale: {len(parents)} parent, {total} token\n")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Interactive retrieval tester")
    ap.add_argument("--engine", choices=("qdrant", "chroma"), default="qdrant")
    ap.add_argument("--extractor", default="docling", help="corpus (docling, mineru, pymupdf4llm, langchain)")
    ap.add_argument("--pipeline", default="A1", help="qdrant: pipeline name prefix (A0, A1, A2, B0, B1, B1m, ...)")
    ap.add_argument("--reranker", choices=("bge", "qwen"), default="qwen", help="qdrant: reranker for A2/B2 pipelines")
    ap.add_argument("--model", default="bge-m3", help="chroma: embedding model key")
    ap.add_argument("--doc", help="qdrant: restrict the search to one manual (doc_code, e.g. H072521IT0)")
    ap.add_argument("--parents", action="store_true", help="qdrant: aggregate chunks into parent documents")
    ap.add_argument("-k", type=int, default=3, help="results to show (chunks, or parents with --parents)")
    ap.add_argument("--chunks", type=int, default=10, help="with --parents: chunks retrieved before aggregation")
    ap.add_argument("--max-chars", type=int, default=3000, help="with --parents: characters of each parent to print")
    ap.add_argument("--width", type=int, default=100)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    if args.engine == "chroma":
        from kb.index import build

        print(f"Carico l'indice Chroma {args.extractor} / {args.model} ...")
        store = build(args.extractor, args.model)

        def run(query: str) -> None:
            hits = store.similarity_search_with_relevance_scores(query, k=args.k)
            show_chunks([(d.metadata, d.page_content, s) for d, s in hits], args.width)
    else:
        from qdrant_client import QdrantClient, models

        from kb.hybrid import PIPELINES, QDRANT_PATH, Encoders, fetch_parents, search

        pipeline = next((p for p in PIPELINES if p.split()[0] == args.pipeline), None)
        if pipeline is None:
            ap.error(f"pipeline sconosciuta: {args.pipeline}. Disponibili: {', '.join(p.split()[0] for p in PIPELINES)}")
        client = QdrantClient(path=str(QDRANT_PATH / args.extractor))
        enc = Encoders(reranker=args.reranker)
        flt = None
        if args.doc:
            flt = models.Filter(must=[models.FieldCondition(key="doc_code", match=models.MatchValue(value=args.doc))])
        mode = f"parent (da {args.chunks} chunk)" if args.parents else "chunk"
        print(f"Qdrant {args.extractor}: {client.count('bge').count} chunk, {client.count('parents').count} parent. "
              f"Catena: {pipeline}. Risultati: {mode}." + (f" Filtro manuale: {args.doc}." if args.doc else ""))

        def run(query: str) -> None:
            hits = search(client, enc, query, pipeline, k=args.chunks if args.parents else args.k, doc_filter=flt)
            if args.parents:
                show_parents(fetch_parents(client, hits, max_parents=args.k), args.width, args.max_chars)
            else:
                show_chunks([(h, h["text"], None) for h in hits], args.width)

    print("Scrivi una domanda, oppure 'quit' per uscire.\n")
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
        run(query)
    print("Ciao.")


if __name__ == "__main__":
    main()
