
"""FAISS-based cosine-similarity retrieval."""

import faiss
import numpy as np


def build_index(gallery_embeddings: np.ndarray):
    """Build an exact inner-product index for normalized embeddings."""
    embeddings = np.asarray(gallery_embeddings, dtype=np.float32)

    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("Gallery embeddings must be a non-empty 2D array.")

    if not np.isfinite(embeddings).all():
        raise ValueError("Gallery embeddings contain non-finite values.")

    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError("Gallery embeddings must be L2-normalized.")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.ascontiguousarray(embeddings))
    return index


def search_index(
    index,
    query_embeddings: np.ndarray,
    top_k: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Return similarity scores and gallery indices for each query."""
    queries = np.asarray(query_embeddings, dtype=np.float32)

    if queries.ndim != 2 or queries.shape[0] == 0:
        raise ValueError("Query embeddings must be a non-empty 2D array.")

    if queries.shape[1] != index.d:
        raise ValueError("Query and gallery embedding dimensions differ.")

    if not np.isfinite(queries).all():
        raise ValueError("Query embeddings contain non-finite values.")

    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    norms = np.linalg.norm(queries, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError("Query embeddings must be L2-normalized.")

    k = min(top_k, index.ntotal)
    if k == 0:
        raise ValueError("The FAISS index is empty.")

    scores, indices = index.search(
        np.ascontiguousarray(queries),
        k,
    )
    return scores, indices
