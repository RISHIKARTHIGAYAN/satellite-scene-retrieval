
import numpy as np
import pytest

from satellite_retrieval.retrieval import build_index, search_index
from scripts.run_reranking import reciprocal_rerank


def normalized_rows(values):
    values = np.asarray(values, dtype=np.float32)
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def test_alpha_one_preserves_original_ranking():
    gallery = normalized_rows([
        [1.0, 0.0],
        [0.8, 0.6],
        [0.0, 1.0],
        [-1.0, 0.0],
    ])
    query = normalized_rows([[1.0, 0.0]])

    index = build_index(gallery)
    scores, indices = search_index(index, query, top_k=4)

    reranked_scores, reranked_indices = reciprocal_rerank(
        gallery,
        query,
        scores,
        indices,
        alpha=1.0,
        reciprocal_k=2,
    )

    np.testing.assert_array_equal(reranked_indices, indices)
    np.testing.assert_allclose(reranked_scores, scores, atol=1e-6)


def test_reranking_preserves_candidate_set():
    gallery = normalized_rows([
        [1.0, 0.0],
        [0.8, 0.6],
        [0.0, 1.0],
        [-1.0, 0.0],
    ])
    query = normalized_rows([[0.9, 0.1], [0.0, 1.0]])

    index = build_index(gallery)
    scores, indices = search_index(index, query, top_k=3)

    _, reranked_indices = reciprocal_rerank(
        gallery, query, scores, indices, alpha=0.7, reciprocal_k=2
    )

    assert reranked_indices.shape == indices.shape
    for original, reranked in zip(indices, reranked_indices):
        assert set(original.tolist()) == set(reranked.tolist())


@pytest.mark.parametrize("alpha", [-0.1, 1.1])
def test_invalid_alpha_raises(alpha):
    gallery = normalized_rows([[1.0, 0.0], [0.0, 1.0]])
    query = normalized_rows([[1.0, 0.0]])
    scores = np.array([[1.0, 0.0]], dtype=np.float32)
    indices = np.array([[0, 1]], dtype=np.int64)

    with pytest.raises(ValueError, match="alpha"):
        reciprocal_rerank(
            gallery, query, scores, indices, alpha=alpha, reciprocal_k=1
        )


def test_invalid_reciprocal_k_raises():
    gallery = normalized_rows([[1.0, 0.0], [0.0, 1.0]])
    query = normalized_rows([[1.0, 0.0]])
    scores = np.array([[1.0, 0.0]], dtype=np.float32)
    indices = np.array([[0, 1]], dtype=np.int64)

    with pytest.raises(ValueError, match="reciprocal_k"):
        reciprocal_rerank(
            gallery, query, scores, indices, alpha=0.7, reciprocal_k=0
        )


def test_mismatched_scores_and_indices_raise():
    gallery = normalized_rows([[1.0, 0.0], [0.0, 1.0]])
    query = normalized_rows([[1.0, 0.0]])
    scores = np.array([[1.0, 0.0]], dtype=np.float32)
    indices = np.array([[0]], dtype=np.int64)

    with pytest.raises(ValueError, match="same shape"):
        reciprocal_rerank(
            gallery, query, scores, indices, alpha=0.7, reciprocal_k=1
        )
