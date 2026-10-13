
"""Reciprocal-neighbor and query-adaptive reranking utilities."""

import numpy as np

from satellite_retrieval.retrieval import build_index, search_index


def build_reciprocal_sets(
    gallery_embeddings: np.ndarray,
    reciprocal_k: int = 20,
) -> list[set[int]]:
    """Build reciprocal gallery-neighbor sets, excluding each item itself."""
    gallery = np.asarray(gallery_embeddings, dtype=np.float32)

    if gallery.ndim != 2 or gallery.shape[0] < 2:
        raise ValueError("Gallery must contain at least two embeddings.")
    if reciprocal_k <= 0:
        raise ValueError("reciprocal_k must be positive.")
    if not np.isfinite(gallery).all():
        raise ValueError("Gallery embeddings must be finite.")

    index = build_index(gallery)
    neighbor_k = min(reciprocal_k + 1, len(gallery))
    _, neighbor_indices = search_index(index, gallery, top_k=neighbor_k)

    neighbor_sets = [
        {int(j) for j in row if int(j) != i}
        for i, row in enumerate(neighbor_indices)
    ]

    return [
        {j for j in neighbors if i in neighbor_sets[j]}
        for i, neighbors in enumerate(neighbor_sets)
    ]


def _validate_inputs(scores, indices, reciprocal_sets):
    scores = np.asarray(scores, dtype=np.float32)
    indices = np.asarray(indices, dtype=np.int64)

    if scores.ndim != 2 or indices.ndim != 2 or scores.shape != indices.shape:
        raise ValueError("Scores and indices must be matching 2D arrays.")
    if scores.shape[0] == 0 or scores.shape[1] < 2:
        raise ValueError("At least one query and two candidates are required.")
    if not np.isfinite(scores).all():
        raise ValueError("Scores must be finite.")
    if indices.min() < 0 or indices.max() >= len(reciprocal_sets):
        raise ValueError("Candidate index is outside reciprocal_sets.")

    return scores, indices


def _rerank_with_weights(
    scores: np.ndarray,
    indices: np.ndarray,
    reciprocal_sets: list[set[int]],
    query_alphas: np.ndarray,
    query_neighbor_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the same neighborhood score with a per-query alpha."""
    if query_neighbor_k <= 0:
        raise ValueError("query_neighbor_k must be positive.")

    output_scores = np.empty_like(scores, dtype=np.float32)
    output_indices = np.empty_like(indices, dtype=np.int64)

    for q in range(scores.shape[0]):
        query_neighbors = set(
            map(int, indices[q, :min(query_neighbor_k, indices.shape[1])])
        )
        candidate_scores = []

        for rank, candidate in enumerate(indices[q]):
            reciprocal = reciprocal_sets[int(candidate)]
            union = query_neighbors | reciprocal

            overlap = (
                len(query_neighbors & reciprocal) / len(union)
                if union
                else 0.0
            )

            alpha = float(query_alphas[q])
            candidate_scores.append(
                alpha * float(scores[q, rank])
                + (1.0 - alpha) * overlap
            )

        order = np.argsort(candidate_scores)[::-1]
        output_scores[q] = np.asarray(candidate_scores, dtype=np.float32)[order]
        output_indices[q] = indices[q, order]

    return output_scores, output_indices


def reciprocal_rerank_from_sets(
    initial_scores: np.ndarray,
    initial_indices: np.ndarray,
    reciprocal_sets: list[set[int]],
    alpha: float = 0.7,
    query_neighbor_k: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-weight reciprocal-neighbor reranking."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1.")

    scores, indices = _validate_inputs(
        initial_scores, initial_indices, reciprocal_sets
    )

    return _rerank_with_weights(
        scores,
        indices,
        reciprocal_sets,
        np.full(scores.shape[0], alpha, dtype=np.float32),
        query_neighbor_k,
    )


def adaptive_rerank(
    initial_scores: np.ndarray,
    initial_indices: np.ndarray,
    reciprocal_sets: list[set[int]],
    alpha_min: float = 0.70,
    alpha_max: float = 0.95,
    confidence_temperature: float = 0.10,
    query_neighbor_k: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Adapt similarity weighting based on the top-two score margin.

    A larger margin receives a larger alpha, placing more weight on the
    original cosine scores. A smaller margin gives neighborhood evidence
    more influence. This is an experimental heuristic, not a calibrated
    probability of retrieval correctness.
    """
    scores, indices = _validate_inputs(
        initial_scores, initial_indices, reciprocal_sets
    )

    if not 0.0 <= alpha_min <= alpha_max <= 1.0:
        raise ValueError("Require 0 <= alpha_min <= alpha_max <= 1.")
    if confidence_temperature <= 0:
        raise ValueError("confidence_temperature must be positive.")
    if query_neighbor_k <= 0:
        raise ValueError("query_neighbor_k must be positive.")

    margins = np.maximum(scores[:, 0] - scores[:, 1], 0.0)
    confidence = 1.0 - np.exp(-margins / confidence_temperature)

    query_alphas = (
        alpha_min + (alpha_max - alpha_min) * confidence
    ).astype(np.float32)

    reranked_scores, reranked_indices = _rerank_with_weights(
        scores,
        indices,
        reciprocal_sets,
        query_alphas,
        query_neighbor_k,
    )

    return reranked_scores, reranked_indices, query_alphas


def selective_rerank(
    initial_scores: np.ndarray,
    initial_indices: np.ndarray,
    reciprocal_sets: list[set[int]],
    margin_threshold: float,
    alpha: float = 0.5,
    query_neighbor_k: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rerank only queries whose top-two similarity margin is small.

    The margin threshold must be selected using validation data only.
    Queries with a margin below the threshold are reranked; all others
    retain their original ranking and scores.

    Returns:
        scores: Output scores in output ranking order.
        indices: Output candidate indices in ranking order.
        reranked_mask: Boolean array indicating which queries were reranked.
    """
    scores, indices = _validate_inputs(
        initial_scores, initial_indices, reciprocal_sets
    )

    if not np.isfinite(margin_threshold) or margin_threshold < 0:
        raise ValueError("margin_threshold must be finite and non-negative.")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1.")
    if query_neighbor_k <= 0:
        raise ValueError("query_neighbor_k must be positive.")

    margins = np.maximum(scores[:, 0] - scores[:, 1], 0.0)
    reranked_mask = margins < margin_threshold

    output_scores = scores.copy()
    output_indices = indices.copy()

    if np.any(reranked_mask):
        reranked_scores, reranked_indices = _rerank_with_weights(
            scores[reranked_mask],
            indices[reranked_mask],
            reciprocal_sets,
            np.full(np.sum(reranked_mask), alpha, dtype=np.float32),
            query_neighbor_k,
        )
        output_scores[reranked_mask] = reranked_scores
        output_indices[reranked_mask] = reranked_indices

    return output_scores, output_indices, reranked_mask
