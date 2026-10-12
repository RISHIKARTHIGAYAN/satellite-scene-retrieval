
import numpy as np
import pytest

from satellite_retrieval.reranking import adaptive_rerank


def test_adaptive_rerank_preserves_candidate_set():
    scores = np.array([[0.95, 0.90, 0.80]], dtype=np.float32)
    indices = np.array([[0, 1, 2]], dtype=np.int64)
    reciprocal_sets = [{0, 1}, {0, 1, 2}, {1, 2}]

    reranked_scores, reranked_indices, alphas = adaptive_rerank(
        scores, indices, reciprocal_sets
    )

    assert reranked_scores.shape == scores.shape
    assert set(reranked_indices[0]) == set(indices[0])
    assert alphas.shape == (1,)
    assert 0.70 <= alphas[0] <= 0.95


def test_ambiguous_query_uses_more_neighborhood_weight():
    scores = np.array([
        [0.95, 0.94, 0.70],
        [0.95, 0.70, 0.60],
    ], dtype=np.float32)
    indices = np.array([[0, 1, 2], [0, 1, 2]], dtype=np.int64)
    reciprocal_sets = [{0, 1}, {0, 1}, {2}]

    _, _, alphas = adaptive_rerank(scores, indices, reciprocal_sets)

    assert alphas[0] < alphas[1]


def test_adaptive_rerank_rejects_single_candidate():
    scores = np.array([[0.95]], dtype=np.float32)
    indices = np.array([[0]], dtype=np.int64)

    with pytest.raises(ValueError, match="two candidates"):
        adaptive_rerank(scores, indices, [{0}])


def test_adaptive_rerank_rejects_invalid_alpha_bounds():
    scores = np.array([[0.95, 0.90]], dtype=np.float32)
    indices = np.array([[0, 1]], dtype=np.int64)

    with pytest.raises(ValueError, match="alpha_min"):
        adaptive_rerank(
            scores, indices, [{0, 1}, {0, 1}],
            alpha_min=0.96, alpha_max=0.95
        )
