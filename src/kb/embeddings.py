"""Embedding factory. Spec format: 'hf:<repo>' or 'ollama:<model>'."""
from __future__ import annotations

from langchain_core.embeddings import Embeddings

from kb.config import EMBEDDING_MODELS, QUERY_PREFIX


def resolve(name_or_spec: str) -> str:
    return EMBEDDING_MODELS.get(name_or_spec, name_or_spec)


def get_embeddings(name_or_spec: str, device: str = "mps", batch_size: int = 16) -> Embeddings:
    spec = resolve(name_or_spec)
    backend, _, model = spec.partition(":")
    if backend == "hf":
        from langchain_huggingface import HuggingFaceEmbeddings

        big = any(s in model for s in ("4B", "8B", "7B"))
        model_kwargs: dict = {"device": device, "trust_remote_code": True}
        if big:
            import torch

            model_kwargs["model_kwargs"] = {"torch_dtype": torch.float16}
        prefix = QUERY_PREFIX.get(spec, "")
        emb = HuggingFaceEmbeddings(
            model_name=model,
            model_kwargs=model_kwargs,
            encode_kwargs={"normalize_embeddings": True, "batch_size": batch_size},
            query_encode_kwargs={"normalize_embeddings": True, "prompt": prefix} if prefix else {"normalize_embeddings": True},
        )
        emb._client.max_seq_length = min(emb._client.max_seq_length or 2048, 2048)
        return emb
    if backend == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(model=model)
    raise ValueError(f"unknown embedding spec: {spec}")
