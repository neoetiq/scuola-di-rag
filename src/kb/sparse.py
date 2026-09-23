"""Encoders for hybrid retrieval: BGE-M3 (dense + learned sparse in one pass), BM25 and SPLADE
via fastembed, and a cross-encoder reranker. Everything runs locally."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger("kb.sparse")


@dataclass
class SparseVec:
    indices: list[int]
    values: list[float]


class BGEM3Encoder:
    """Dense (CLS, normalized) + sparse (ReLU(linear) max-pooled per token id), as in FlagEmbedding."""

    def __init__(self, repo: str = "BAAI/bge-m3", device: str = "mps", max_length: int = 2048, batch_size: int = 8):
        from huggingface_hub import hf_hub_download
        from transformers import AutoModel, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(repo)
        self.model = AutoModel.from_pretrained(repo, torch_dtype=torch.float16).to(device).eval()
        self.sparse_linear = torch.nn.Linear(self.model.config.hidden_size, 1)
        state = torch.load(hf_hub_download(repo, "sparse_linear.pt"), map_location="cpu")
        self.sparse_linear.load_state_dict(state)
        self.sparse_linear = self.sparse_linear.to(device).half().eval()
        self.device, self.max_length, self.batch_size = device, max_length, batch_size
        self.skip_ids = {self.tok.cls_token_id, self.tok.eos_token_id, self.tok.pad_token_id, self.tok.unk_token_id}

    @torch.no_grad()
    def encode(self, texts: list[str]) -> tuple[np.ndarray, list[SparseVec]]:
        dense_all, sparse_all = [], []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            enc = self.tok(batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
            hidden = self.model(**enc).last_hidden_state
            dense = torch.nn.functional.normalize(hidden[:, 0].float(), dim=-1).cpu().numpy()
            weights = torch.relu(self.sparse_linear(hidden)).squeeze(-1).float().cpu().numpy()
            ids = enc["input_ids"].cpu().numpy()
            for row_ids, row_w in zip(ids, weights):
                d: dict[int, float] = {}
                for tid, w in zip(row_ids.tolist(), row_w.tolist()):
                    if tid in self.skip_ids or w <= 0:
                        continue
                    if w > d.get(tid, 0.0):
                        d[tid] = w
                sparse_all.append(SparseVec(list(d.keys()), list(d.values())))
            dense_all.append(dense)
        return np.concatenate(dense_all), sparse_all


class FastembedSparse:
    """fastembed sparse encoders: 'Qdrant/bm25' (no stemming, language agnostic) or SPLADE."""

    def __init__(self, name: str):
        from fastembed import SparseTextEmbedding

        kw = {"disable_stemmer": True} if name == "Qdrant/bm25" else {}
        self.model = SparseTextEmbedding(model_name=name, **kw)
        self.is_bm25 = name == "Qdrant/bm25"

    def encode(self, texts: list[str], query: bool = False) -> list[SparseVec]:
        it = self.model.query_embed(texts) if (query and self.is_bm25) else self.model.embed(texts, batch_size=32)
        return [SparseVec(e.indices.tolist(), e.values.tolist()) for e in it]


class Reranker:
    def __init__(self, repo: str = "BAAI/bge-reranker-v2-m3", device: str = "mps", max_length: int = 1024):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(repo, device=device, max_length=max_length, model_kwargs={"torch_dtype": torch.float16})

    def rerank(self, query: str, docs: list[str]) -> list[float]:
        return self.model.predict([(query, d) for d in docs], batch_size=8).tolist()


class Qwen3Reranker:
    """Qwen3-Reranker: causal LM judged on P('yes') for (instruction, query, document)."""

    _PREFIX = (
        "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the "
        'Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    )
    _SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    _INSTRUCTION = "Given a question about a stove or fireplace manual, judge whether the passage answers it"

    def __init__(self, repo: str = "Qwen/Qwen3-Reranker-0.6B", device: str = "mps", max_length: int = 2048, batch_size: int = 4):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(repo, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float16).to(device).eval()
        self.yes_id = self.tok.convert_tokens_to_ids("yes")
        self.no_id = self.tok.convert_tokens_to_ids("no")
        self.device, self.max_length, self.batch_size = device, max_length, batch_size

    @torch.no_grad()
    def rerank(self, query: str, docs: list[str]) -> list[float]:
        texts = [f"{self._PREFIX}<Instruct>: {self._INSTRUCTION}\n<Query>: {query}\n<Document>: {d}{self._SUFFIX}" for d in docs]
        scores: list[float] = []
        for i in range(0, len(texts), self.batch_size):
            enc = self.tok(texts[i : i + self.batch_size], padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
            logits = self.model(**enc).logits[:, -1, :]
            pair = torch.stack([logits[:, self.no_id], logits[:, self.yes_id]], dim=1).float()
            scores += torch.log_softmax(pair, dim=1)[:, 1].exp().tolist()
        return scores
