
"""Evaluation metrics for ranked image retrieval."""

from collections.abc import Sequence

import numpy as np


def _validate_inputs(
    query_labels: Sequence,
    retrieved_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Check label and retrieval-array shapes."""
    query_labels = np.asarray(query_labels)
    retrieved_labels = np.asarray(retrieved_labels)

    if query_labels.ndim != 1:
        raise ValueError("query_labels must be one-dimensional.")

    if retrieved_labels.ndim != 2:
        raise ValueError("retrieved_labels must be a two-dimensional array.")

    if retrieved_labels.shape[0] != len(query_labels):
        raise ValueError("Each query must have one row of retrieved labels.")

    if retrieved_labels.shape[1] == 0:
        raise ValueError("retrieved_labels must contain at least one result.")

    if len(query_labels) == 0:
        raise ValueError("At least one query is required.")

    return query_labels, retrieved_labels


def top1_accuracy(query_labels: Sequence, retrieved_labels: np.ndarray) -> float:
    """Fraction of queries whose first retrieved item has the correct label."""
    query_labels, retrieved_labels = _validate_inputs(
        query_labels, retrieved_labels
    )
    return float(np.mean(retrieved_labels[:, 0] == query_labels))


def hit_rate_at_k(
    query_labels: Sequence,
    retrieved_labels: np.ndarray,
    k: int,
) -> float:
    """Fraction of queries with at least one correct result in the top k."""
    query_labels, retrieved_labels = _validate_inputs(
        query_labels, retrieved_labels
    )

    if k <= 0:
        raise ValueError("k must be positive.")

    k = min(k, retrieved_labels.shape[1])
    hits = np.any(retrieved_labels[:, :k] == query_labels[:, None], axis=1)
    return float(np.mean(hits))


def mean_average_precision_at_k(
    query_labels: Sequence,
    retrieved_labels: np.ndarray,
    k: int,
    gallery_labels: Sequence | None = None,
) -> float:
    """Mean truncated AP@k for single-label image retrieval.

    If gallery_labels is provided, normalize each query's AP by
    min(number of relevant gallery items, k). Otherwise, use the
    number of relevant items retrieved within the top k.
    """
    query_labels, retrieved_labels = _validate_inputs(
        query_labels, retrieved_labels
    )

    if k <= 0:
        raise ValueError("k must be positive.")

    k = min(k, retrieved_labels.shape[1])
    relevance = retrieved_labels[:, :k] == query_labels[:, None]

    ranks = np.arange(1, k + 1, dtype=float)
    precision_at_rank = np.cumsum(relevance, axis=1) / ranks
    summed_precision = np.sum(precision_at_rank * relevance, axis=1)

    if gallery_labels is not None:
        gallery_labels = np.asarray(gallery_labels)
        if gallery_labels.ndim != 1:
            raise ValueError("gallery_labels must be one-dimensional.")
        relevant_total = np.sum(
            query_labels[:, None] == gallery_labels[None, :],
            axis=1,
        )
        denominator = np.minimum(relevant_total, k)
    else:
        denominator = np.sum(relevance, axis=1)

    denominator = np.maximum(denominator, 1)
    return float(np.mean(summed_precision / denominator))
