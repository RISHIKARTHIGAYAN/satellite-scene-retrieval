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

    # Relevant results at ranks 2 and 3.
    expected = ((1 / 2) + (2 / 3)) / 2

    assert mean_average_precision_at_k(
        query_labels, retrieved, 4
    ) == pytest.approx(expected)


def test_map_uses_total_relevant_gallery_items():
    query_labels = np.array([0])
    retrieved = np.array([[1, 0, 0, 1]])
    gallery_labels = np.array([0, 0, 0, 0, 1, 1])

    # Four relevant gallery images; two are retrieved within the cutoff.
    expected = ((1 / 2) + (2 / 3)) / 4

    result = mean_average_precision_at_k(
        query_labels,
        retrieved,
        k=4,
        gallery_labels=gallery_labels,
    )

    assert result == pytest.approx(expected)


def test_map_denominator_is_capped_at_k():
    query_labels = np.array([0])
    retrieved = np.array([[0, 0, 1]])
    gallery_labels = np.array([0, 0, 0, 0, 0])

    # min(5 relevant gallery items, k=3) = 3.
    # Only the first two retrieved items are relevant.
    expected = (1 + 1) / 3

    assert mean_average_precision_at_k(
        query_labels, retrieved, k=3, gallery_labels=gallery_labels
    ) == pytest.approx(expected)


def test_invalid_k_raises():
    query_labels = np.array([0])
    retrieved = np.array([[0, 1]])

    with pytest.raises(ValueError, match="positive"):
        hit_rate_at_k(query_labels, retrieved, 0)

    with pytest.raises(ValueError, match="positive"):
        mean_average_precision_at_k(query_labels, retrieved, 0)


def test_mismatched_query_count_raises():
    with pytest.raises(ValueError, match="Each query"):
        top1_accuracy(np.array([0, 1]), np.array([[0, 1]]))


@pytest.mark.parametrize(
    ("query_labels", "retrieved"),
    [
        (np.array([]), np.empty((0, 2), dtype=int)),
        (np.array([0]), np.empty((1, 0), dtype=int)),
        (np.array([[0]]), np.array([[0, 1]])),
    ],
)
def test_invalid_metric_input_shapes_raise(query_labels, retrieved):
    with pytest.raises(ValueError):
        top1_accuracy(query_labels, retrieved)


def test_invalid_gallery_labels_shape_raises():
    with pytest.raises(ValueError, match="one-dimensional"):
        mean_average_precision_at_k(
            np.array([0]),
            np.array([[0, 1]]),
            k=2,
            gallery_labels=np.array([[0, 1]]),
        )


def test_empty_gallery_labels_produce_zero_map():
    # No relevant items in the gallery: denominator is guarded against zero.
    result = mean_average_precision_at_k(
        np.array([0]),
        np.array([[1, 1]]),
        k=2,
        gallery_labels=np.array([1, 1, 1]),
    )

    assert result == pytest.approx(0.0)
