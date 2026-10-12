
import numpy as np
import pytest

from satellite_retrieval.metrics import (
    hit_rate_at_k,
    mean_average_precision_at_k,
    top1_accuracy,
)


def test_top1_accuracy():
    query_labels = np.array(["Forest", "River", "Highway"])
    retrieved = np.array([
        ["Forest", "River"],
        ["Highway", "River"],
        ["Highway", "Forest"],
    ])

    assert top1_accuracy(query_labels, retrieved) == pytest.approx(2 / 3)


def test_hit_rate_at_k():
    query_labels = np.array([0, 1, 2])
    retrieved = np.array([
        [0, 1, 2],
        [2, 1, 0],
        [1, 0, 2],
    ])

    assert hit_rate_at_k(query_labels, retrieved, 1) == pytest.approx(1 / 3)
    assert hit_rate_at_k(query_labels, retrieved, 2) == pytest.approx(2 / 3)
    assert hit_rate_at_k(query_labels, retrieved, 10) == pytest.approx(1.0)


def test_map_at_k():
    query_labels = np.array([0])
    retrieved = np.array([[1, 0, 0, 1]])

    # Relevant results at ranks 2 and 3:
    # AP@4 = ((1/2) + (2/3)) / 2
    expected = ((1 / 2) + (2 / 3)) / 2

    assert mean_average_precision_at_k(
        query_labels, retrieved, 4
    ) == pytest.approx(expected)


def test_invalid_k_raises():
    query_labels = np.array([0])
    retrieved = np.array([[0, 1]])

    with pytest.raises(ValueError):
        hit_rate_at_k(query_labels, retrieved, 0)


def test_mismatched_query_count_raises():
    with pytest.raises(ValueError):
        top1_accuracy(np.array([0, 1]), np.array([[0, 1]]))

def test_map_uses_total_relevant_gallery_items():
    query_labels = np.array([0])
    retrieved = np.array([[1, 0, 0, 1]])
    gallery_labels = np.array([0, 0, 0, 0, 1, 1])

    expected = ((1 / 2) + (2 / 3)) / 4

    result = mean_average_precision_at_k(
        query_labels,
        retrieved,
        k=4,
        gallery_labels=gallery_labels,
    )

    assert result == pytest.approx(expected)