"""Hybrid retrieval benchmark on Qdrant (embedded, no server).

Two collections, one per dense model, each also holding every sparse representation:
  bge  : dense=BGE-M3            sparse={bge_sparse, bm25, splade}
  qwen : dense=Qwen3-Embedding-4B sparse={bge_sparse, bm25, splade}
A pipeline = dense branch + sparse branches fused with RRF (+ optional reranker).
"""
from __future__ import annotations

import argparse
import json
import logging
import time
import uuid
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models

from kb.chunk import build_corpus
from kb.config import EVAL_DIR, ROOT
from kb.embeddings import get_embeddings
from kb.evaluate import is_hit, load_questions
from kb.sparse import BGEM3Encoder, FastembedSparse, Qwen3Reranker, Reranker, SparseVec

log = logging.getLogger("kb.hybrid")
QDRANT_PATH = ROOT / "data" / "qdrant"
SPARSE_NAMES = ("bge_sparse", "bm25", "splade")
PARENTS = "parents"  # payload-only collection: the parent-document docstore
SPLADE = "prithivida/Splade_PP_en_v1"

# name -> (dense collection, sparse branches, rerank)
PIPELINES: dict[str, tuple[str, tuple[str, ...], bool]] = {
    "A0 bge dense": ("bge", (), False),
    "A1 bge dense+sparse RRF": ("bge", ("bge_sparse",), False),
    "A2 A1 + reranker": ("bge", ("bge_sparse",), True),
    "B0 qwen dense": ("qwen", (), False),
    "B1 qwen+bm25+splade RRF": ("qwen", ("bm25", "splade"), False),
    "B2 B1 + reranker": ("qwen", ("bm25", "splade"), True),
    "B1m qwen+bm25+bge_sparse RRF": ("qwen", ("bm25", "bge_sparse"), False),
    "B2m B1m + reranker": ("qwen", ("bm25", "bge_sparse"), True),
    "S1 bge_sparse only": ("bge", ("bge_sparse",), None),
    "S2 bm25 only": ("bge", ("bm25",), None),
    "S3 splade only": ("bge", ("splade",), None),
}


def _sv(v: SparseVec) -> models.SparseVector:
    return models.SparseVector(indices=v.indices, values=v.values)


class Encoders:
    """Encoders are loaded lazily, so a BGE-only pipeline never loads Qwen3-4B (and vice versa)."""

    def __init__(self, reranker: str = "bge"):
        self.reranker_name = reranker
        self._cache: dict[str, object] = {}

    def _get(self, key: str, factory):
        if key not in self._cache:
            self._cache[key] = factory()
        return self._cache[key]

    @property
    def bge(self) -> BGEM3Encoder:
        return self._get("bge", BGEM3Encoder)

    @property
    def qwen(self):
        return self._get("qwen", lambda: get_embeddings("qwen3-4b"))

    @property
    def bm25(self) -> FastembedSparse:
        return self._get("bm25", lambda: FastembedSparse("Qdrant/bm25"))

    @property
    def splade(self) -> FastembedSparse:
        return self._get("splade", lambda: FastembedSparse(SPLADE))

    @property
    def reranker(self):
        return self._get("reranker", lambda: Qwen3Reranker() if self.reranker_name == "qwen" else Reranker())

    def release(self, key: str) -> None:
        """Drop a loaded model and give its memory back (it is reloaded lazily if needed again)."""
        import gc

        import torch

        self._cache.pop(key, None)
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def encode_query(self, q: str, needed: set[str]) -> dict:
        out: dict = {}
        if needed & {"bge", "bge_sparse"}:
            d, sp = self.bge.encode([q])
            out["bge"], out["bge_sparse"] = d[0].tolist(), _sv(sp[0])
        if "qwen" in needed:
            out["qwen"] = self.qwen.embed_query(q)
        if "bm25" in needed:
            out["bm25"] = _sv(self.bm25.encode([q], query=True)[0])
        if "splade" in needed:
            out["splade"] = _sv(self.splade.encode([q])[0])
        return out


def parent_point_id(parent_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, parent_id))


def build_collections(client: QdrantClient, enc: Encoders, extractor: str, force: bool = False) -> list:
    docs, parents = build_corpus(extractor)
    texts = [d.page_content for d in docs]
    existing = {c.name for c in client.get_collections().collections}
    if {"bge", "qwen", PARENTS} <= existing and not force:
        log.info("reusing Qdrant collections (%d chunks)", client.count("bge").count)
        return docs
    # One model in memory at a time: sparse (ONNX, CPU) first, then each dense model, released after use.
    t0 = time.time()
    bm25 = enc.bm25.encode(texts)
    splade = enc.splade.encode(texts)
    enc.release("splade")
    log.info("bm25 + splade: %.0fs", time.time() - t0); t0 = time.time()
    bge_dense, bge_sparse = enc.bge.encode(texts)
    enc.release("bge")
    log.info("bge-m3 dense+sparse: %.0fs", time.time() - t0); t0 = time.time()
    qwen_dense = np.array(enc.qwen.embed_documents(texts))
    enc.release("qwen")
    log.info("qwen3-4b dense: %.0fs", time.time() - t0)
    for name, dense in (("bge", bge_dense), ("qwen", qwen_dense)):
        if name in existing:
            client.delete_collection(name)
        client.create_collection(
            name,
            vectors_config={"dense": models.VectorParams(size=dense.shape[1], distance=models.Distance.COSINE)},
            sparse_vectors_config={s: models.SparseVectorParams(modifier=models.Modifier.IDF if s == "bm25" else None) for s in SPARSE_NAMES},
        )
        points = [
            models.PointStruct(
                id=i,
                vector={"dense": dense[i].tolist(), "bge_sparse": _sv(bge_sparse[i]), "bm25": _sv(bm25[i]), "splade": _sv(splade[i])},
                payload=d.metadata | {"text": d.page_content},
            )
            for i, d in enumerate(docs)
        ]
        for j in range(0, len(points), 64):
            client.upsert(name, points[j : j + 64])
        log.info("collection %s: %d points", name, len(points))

    # parent-document docstore: no vectors, looked up by id (uuid5 of parent_id)
    if PARENTS in existing:
        client.delete_collection(PARENTS)
    client.create_collection(PARENTS, vectors_config={})
    ppoints = [models.PointStruct(id=parent_point_id(p.parent_id), vector={}, payload=p.payload()) for p in parents]
    for j in range(0, len(ppoints), 64):
        client.upsert(PARENTS, ppoints[j : j + 64])
    log.info("collection %s: %d parents", PARENTS, len(ppoints))
    return docs


def fetch_parents(client: QdrantClient, hits: list[dict], max_parents: int | None = None) -> list[dict]:
    """Aggregate retrieved chunks into their parent documents, best-ranked parent first.
    Each parent payload gets ``hit_chunks`` (how many retrieved chunks fall in it) and ``best_rank``."""
    order: dict[str, dict] = {}
    for rank, h in enumerate(hits, 1):
        pid = h.get("parent_id")
        if not pid:
            continue
        info = order.setdefault(pid, {"best_rank": rank, "hit_chunks": 0})
        info["hit_chunks"] += 1
    ids = list(order)[:max_parents] if max_parents else list(order)
    found = {p.payload["parent_id"]: p.payload for p in client.retrieve(PARENTS, ids=[parent_point_id(i) for i in ids])}
    return [found[i] | order[i] for i in ids if i in found]


def search(
    client: QdrantClient, enc: Encoders, q: str, pipeline: str, k: int = 10, prefetch: int = 20,
    doc_filter: models.Filter | None = None,
) -> list[dict]:
    coll, sparse_branches, rerank = PIPELINES[pipeline]
    qv = enc.encode_query(q, ({coll} if rerank is not None else set()) | set(sparse_branches))
    branches = []
    if rerank is not None:  # dense branch included (None = sparse-only pipelines)
        branches.append(models.Prefetch(query=qv[coll], using="dense", limit=prefetch, filter=doc_filter))
    branches += [models.Prefetch(query=qv[s], using=s, limit=prefetch, filter=doc_filter) for s in sparse_branches]
    if len(branches) == 1:
        res = client.query_points(coll, query=branches[0].query, using=branches[0].using, limit=prefetch if rerank else k, query_filter=doc_filter)
    else:
        res = client.query_points(coll, prefetch=branches, query=models.FusionQuery(fusion=models.Fusion.RRF), limit=prefetch)
    hits = [p.payload for p in res.points]
    if rerank:
        scores = enc.reranker.rerank(q, [h["text"] for h in hits])
        hits = [h for _, h in sorted(zip(scores, hits), key=lambda x: -x[0])]
    return hits[:k]


def is_hit_family(meta: dict, q: dict) -> bool:
    """Same manual in any language (doc code without the language suffix) and overlapping pages."""
    fam = lambda c: (c or "")[:-3]
    return fam(meta.get("doc_code")) == fam(q["doc_code"]) and any(
        meta["page_start"] <= p <= meta["page_end"] for p in q["pages"]
    )


def evaluate(client: QdrantClient, enc: Encoders, questions: list[dict], pipeline: str, k: int = 10, mode: str = "plain") -> dict:
    rerank = PIPELINES[pipeline][2]
    """mode: plain = question as written; product = product name appended to the query
    (user names the stove); filter = Qdrant payload filter on the manual (user selected it)."""
    ranks, ranks_fam, t0 = [], [], time.time()
    for q in questions:
        text = q["q"] if mode != "product" else f"{q['q']} ({q.get('product', '')})"
        flt = None
        if mode == "filter":
            flt = models.Filter(must=[models.FieldCondition(key="doc_code", match=models.MatchValue(value=q["doc_code"]))])
        hits = search(client, enc, text, pipeline, k=k, doc_filter=flt)
        rank = next((i + 1 for i, h in enumerate(hits) if is_hit(h, q)), None)
        ranks.append(rank)
        ranks_fam.append(next((i + 1 for i, h in enumerate(hits) if is_hit_family(h, q)), None))
        q.setdefault("_ranks", {})[pipeline] = rank
    n = len(questions)
    sub = lambda pred: [r for r, q in zip(ranks, questions) if pred(q)]
    mrr = lambda rs: (sum(1 / r for r in rs if r) / len(rs)) if rs else float("nan")
    return {
        "pipeline": pipeline + (" [qwen-rr]" if rerank and enc.reranker_name == "qwen" else ""),
        "mode": mode,
        "hit@1": sum(1 for r in ranks if r == 1) / n,
        "hit@3": sum(1 for r in ranks if r and r <= 3) / n,
        "hit@5": sum(1 for r in ranks if r and r <= 5) / n,
        "mrr": mrr(ranks),
        "mrr_fam": mrr(ranks_fam),
        "mrr_code": mrr(sub(lambda q: q.get("type") == "code")),
        "mrr_cross": mrr(sub(lambda q: q.get("cross"))),
        "mrr_table": mrr(sub(lambda q: q.get("type") == "table")),
        "s/query": (time.time() - t0) / n,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Hybrid retrieval benchmark on Qdrant")
    ap.add_argument("--extractor", default="docling")
    ap.add_argument("--pipeline", action="append", help="pipeline name prefix (e.g. A2); default: all")
    ap.add_argument("--questions", type=Path, default=EVAL_DIR / "questions.jsonl")
    ap.add_argument("--force", action="store_true", help="rebuild collections")
    ap.add_argument("--mode", choices=("plain", "product", "filter"), default="plain")
    ap.add_argument("--reranker", choices=("bge", "qwen"), default="bge", help="bge-reranker-v2-m3 or Qwen3-Reranker-0.6B")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "transformers", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    client = QdrantClient(path=str(QDRANT_PATH / args.extractor))
    enc = Encoders(reranker=args.reranker)
    build_collections(client, enc, args.extractor, args.force)
    questions = load_questions(args.questions)
    selected = [p for p in PIPELINES if not args.pipeline or any(p.startswith(x) for x in args.pipeline)]
    rows = [evaluate(client, enc, questions, p, mode=args.mode) for p in selected]
    print(f"\nmode={args.mode}  corpus={client.count('bge').count} chunks  questions={len(questions)}")

    hdr = f"{'pipeline':<32} {'hit@1':>6} {'hit@3':>6} {'hit@5':>6} {'MRR':>6} {'fam':>6} {'code':>6} {'cross':>6} {'table':>6} {'s/q':>6}"
    print("\n" + hdr)
    for r in rows:
        print(f"{r['pipeline']:<32} {r['hit@1']:6.2f} {r['hit@3']:6.2f} {r['hit@5']:6.2f} {r['mrr']:6.3f} {r['mrr_fam']:6.3f} {r['mrr_code']:6.3f} {r['mrr_cross']:6.3f} {r['mrr_table']:6.3f} {r['s/query']:6.2f}")
    if args.verbose:
        print("\nper-question ranks (columns = pipelines in the order above):")
        for q in questions:
            print("  " + " ".join(f"{str(q['_ranks'].get(p, '-')):>4}" for p in selected) + f"  [{q.get('type','')}{' x' if q.get('cross') else ''}] {q['q'][:60]}")
    with (EVAL_DIR / "results_hybrid.jsonl").open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r | {"extractor": args.extractor, "ts": time.strftime("%Y-%m-%d %H:%M")}) + "\n")
    client.close()


if __name__ == "__main__":
    main()
