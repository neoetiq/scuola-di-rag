"""Retrieval benchmark: for each embedding model, hit@k and MRR on eval/questions.jsonl.

A question is a JSON line: {"q": "...", "doc_code": "H072521IT0", "pages": [10], "lang": "it"}
A retrieved chunk is a hit when doc_code matches and its page range overlaps `pages`.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from kb.config import EMBEDDING_MODELS, EVAL_DIR
from kb.index import build

log = logging.getLogger("kb.eval")
KS = (1, 3, 5, 10)


def load_questions(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]


def is_hit(meta: dict, q: dict) -> bool:
    if meta.get("doc_code") != q["doc_code"]:
        return False
    return any(meta["page_start"] <= p <= meta["page_end"] for p in q["pages"])


def evaluate(extractor: str, model: str, questions: list[dict], k: int = 10) -> dict:
    store = build(extractor, model)
    ranks: list[int | None] = []
    t0 = time.time()
    for q in questions:
        hits = store.similarity_search(q["q"], k=k)
        rank = next((i + 1 for i, d in enumerate(hits) if is_hit(d.metadata, q)), None)
        ranks.append(rank)
        q["_rank"] = rank
        q["_top"] = [f"p{d.metadata['page_start']} {d.metadata['section'][:60]}" for d in hits[:3]]
    n = len(questions)
    res = {"model": model, "extractor": extractor, "n": n, "query_s": (time.time() - t0) / n}
    for kk in KS:
        res[f"hit@{kk}"] = sum(1 for r in ranks if r and r <= kk) / n
    res["mrr"] = sum(1 / r for r in ranks if r) / n
    return res


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Benchmark embedding models on the eval set")
    ap.add_argument("--extractor", default="docling")
    ap.add_argument("--model", action="append")
    ap.add_argument("--questions", type=Path, default=EVAL_DIR / "questions.jsonl")
    ap.add_argument("--verbose", action="store_true", help="print per-question ranks")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    questions = load_questions(args.questions)
    rows = []
    for m in args.model or list(EMBEDDING_MODELS):
        res = evaluate(args.extractor, m, questions)
        rows.append(res)
        if args.verbose:
            for q in questions:
                print(f"  rank={q['_rank']!s:>4}  [{q['lang']}] {q['q'][:70]}  -> {q['_top'][0]}")
    print(f"\n{'model':<12} {'extractor':<11} {'hit@1':>6} {'hit@3':>6} {'hit@5':>6} {'hit@10':>6} {'MRR':>6} {'s/query':>8}")
    for r in rows:
        print(f"{r['model']:<12} {r['extractor']:<11} {r['hit@1']:6.2f} {r['hit@3']:6.2f} {r['hit@5']:6.2f} {r['hit@10']:6.2f} {r['mrr']:6.3f} {r['query_s']:8.3f}")
    out = EVAL_DIR / "results.jsonl"
    with out.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r | {"ts": time.strftime("%Y-%m-%d %H:%M")}) + "\n")


if __name__ == "__main__":
    main()
