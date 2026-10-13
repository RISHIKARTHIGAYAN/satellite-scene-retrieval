
import numpy as np
import pytest

from satellite_retrieval.reranking import selective_rerank


def test_high_margin_query_keeps_original_ranking():
    scores = np.array([[0.95, 0.70, 0.60]], dtype=np.float32)
    indices = np.array([[0, 1, 2]], dtype=np.int64)
    reciprocal_sets = [{0, 1}, {0, 1}, {2}]

    out_scores, out_indices, mask = selective_rerank(
        scores, indices, reciprocal_sets, margin_threshold=0.10
    )

    assert mask.tolist() == [False]
    np.testing.assert_array_equal(out_scores, scores)
    np.testing.assert_array_equal(out_indices, indices)


def test_low_margin_query_is_selected():
    scores = np.array([[0.95, 0.94, 0.70]], dtype=np.float32)
    indices = np.array([[0, 1, 2]], dtype=np.int64)
    reciprocal_sets = [{0, 1}, {0, 1}, {2}]

    _, _, mask = selective_rerank(
        scores, indices, reciprocal_sets, margin_threshold=0.02
    )

    assert mask.tolist() == [True]


def test_selective_reranking_preserves_candidate_set():
    scores = np.array([
        [0.95, 0.94, 0.70],
        [0.95, 0.70, 0.60],
    ], dtype=np.float32)
    indices = np.array([[0, 1, 2], [0, 1, 2]], dtype=np.int64)
    reciprocal_sets = [{0, 1}, {0, 1}, {2}]

    _, out_indices, _ = selective_rerank(
        scores, indices, reciprocal_sets, margin_threshold=0.10
    )

    for original, reranked in zip(indices, out_indices):
        assert set(original.tolist()) == set(reranked.tolist())


@pytest.mark.parametrize("threshold", [-0.1, float("nan"), float("inf")])
def test_invalid_margin_threshold_raises(threshold):
    scores = np.array([[0.95, 0.90]], dtype=np.float32)
    indices = np.array([[0, 1]], dtype=np.int64)

    with pytest.raises(ValueError, match="margin_threshold"):
        selective_rerank(
            scores, indices, [{0, 1}, {0, 1}],
            margin_threshold=threshold
        )

def test_zero_threshold_preserves_all_rankings():
    scores = np.array(
        [[0.90, 0.70, 0.50],
         [0.80, 0.60, 0.40]],
        dtype=np.float32,
    )
    indices = np.array(
        [[0, 1, 2],
         [2, 1, 0]],
        dtype=np.int64,
    )
    reciprocal_sets = [
        {0, 1},
        {0, 1, 2},
        {1, 2},
    ]

    output_scores, output_indices, mask = selective_rerank(
        scores,
        indices,
        reciprocal_sets,
        margin_threshold=0.0,
        alpha=0.5,
        query_neighbor_k=2,
    )

    assert not mask.any()
    np.testing.assert_array_equal(output_indices, indices)
    np.testing.assert_array_equal(output_scores, scores)
