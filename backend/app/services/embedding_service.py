"""Nemotron embedding service — batch embed texts via nemotron-3.5-embed-1b (2048-dim).

Single client instance reused across calls. Supports batched input for efficiency
(one network request per batch, up to 64 texts/request).

Asymmetric model: requires input_type parameter ("passage" for documents, "query" for queries).
"""
from __future__ import annotations

from typing import List, Optional

from openai import OpenAI

from app.core.config import settings


# Reuse one client (cheap, thread-safe).
_client: Optional[OpenAI] = None
EMBED_MODEL = "nvidia/nemotron-3.5-embed-8b"
EMBED_DIM = 4096
MAX_BATCH = 64


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=settings.nvidia_api_key,
            base_url=settings.nvidia_base_url,
            timeout=30.0,  # 30s timeout - embedding service can be slow under load
        )
    return _client


from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from openai import APIError, APIStatusError, APITimeoutError, RateLimitError

# Retry on timeout, 503, rate limit, and transient errors
_embed_retry_config = {
    "wait": wait_exponential_jitter(initial=0.5, max=4),
    "stop": stop_after_attempt(3),
    "retry": retry_if_exception_type((TimeoutError, ConnectionError, APIError, APIStatusError, APITimeoutError, RateLimitError)),
    "reraise": True,
}

@retry(**_embed_retry_config)
def _embed_batch(batch: List[str], input_type: str) -> List[List[float]]:
    """Embed a batch with the required input_type parameter. Retries on transient errors."""
    client = _get_client()
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=batch,
        extra_body={"input_type": input_type}
    )
    sorted_data = sorted(resp.data, key=lambda d: d.index)
    return [d.embedding for d in sorted_data]


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a list of document passages in batches of MAX_BATCH; return vector list."""
    if not texts:
        return []
    out: List[List[float]] = []
    for i in range(0, len(texts), MAX_BATCH):
        batch = texts[i:i + MAX_BATCH]
        out.extend(_embed_batch(batch, "passage"))
    return out


def embed_query(text: str) -> List[float]:
    """Embed a single query text — used for retrieval queries."""
    return _embed_batch([text], "query")[0]