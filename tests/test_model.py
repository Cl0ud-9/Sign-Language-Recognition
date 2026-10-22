import numpy as np
import pytest

from function import NUM_KEYPOINTS, actions, load_trained_model, sequence_length, split


def test_shipped_model_gives_one_probability_per_sign():
    model = load_trained_model()
    probs = model.predict(np.zeros((2, sequence_length, NUM_KEYPOINTS)), verbose=0)
    assert probs.shape == (2, len(actions))
    assert np.allclose(probs.sum(axis=1), 1, atol=1e-5)


def test_training_builds_the_shipped_architecture():
    pytest.importorskip('keras', reason='training needs TensorFlow: requirements-train.txt')
    from trainmodel import build_model

    fresh = build_model()
    shipped = load_trained_model().layers()
    assert [layer.count_params() for layer in fresh.layers] == [params for _, _, params in shipped]


def test_split_is_reproducible_and_keeps_every_sign_in_training():
    X = np.arange(240)
    y = np.repeat(np.arange(len(actions)), 30)
    first, second = split(X, y), split(X, y)
    assert all((a == b).all() for a, b in zip(first, second))
    assert set(first[2]) == set(range(len(actions)))
