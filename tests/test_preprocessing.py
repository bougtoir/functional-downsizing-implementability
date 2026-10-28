import numpy as np

from src.preprocess.build_inputs import weighted_kmeans


def test_weighted_kmeans_is_deterministic_and_complete():
    coordinates = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0]])
    weights = np.array([1.0, 2.0, 3.0, 4.0])
    labels_a, centres_a = weighted_kmeans(coordinates, weights, clusters=2, seed=17)
    labels_b, centres_b = weighted_kmeans(coordinates, weights, clusters=2, seed=17)
    assert np.array_equal(labels_a, labels_b)
    assert np.allclose(centres_a, centres_b)
    assert set(labels_a) == {0, 1}


def test_weighted_kmeans_preserves_singletons():
    coordinates = np.array([[3.0, 4.0], [8.0, 9.0]])
    labels, centres = weighted_kmeans(
        coordinates, np.array([5.0, 6.0]), clusters=2, seed=1
    )
    assert np.array_equal(labels, np.array([0, 1]))
    assert np.array_equal(centres, coordinates)
