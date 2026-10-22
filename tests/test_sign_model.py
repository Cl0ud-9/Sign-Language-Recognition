import os

import numpy as np
import pytest

from function import MODEL_NPZ, NUM_KEYPOINTS, actions, sequence_length
from sign_model import SignModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCE = os.path.join(ROOT, 'tests', 'fixtures', 'reference.npz')


def test_numpy_model_matches_keras():
    # tools/make_reference.py ran these inputs through Keras with the same weights
    with np.load(REFERENCE) as ref:
        x, expected = ref['x'], ref['probs']
    got = SignModel.load(MODEL_NPZ).predict(x)
    assert got.shape == expected.shape
    assert np.abs(got - expected).max() < 1e-5
    assert (got.argmax(axis=1) == expected.argmax(axis=1)).all()


def test_save_and_load_round_trip(tmp_path):
    model = SignModel.load(MODEL_NPZ)
    model.save(tmp_path / 'copy.npz')
    x = np.random.default_rng(0).uniform(0, 1, (3, sequence_length, NUM_KEYPOINTS))
    assert np.array_equal(SignModel.load(tmp_path / 'copy.npz').predict(x), model.predict(x))


def test_npz_matches_the_keras_weights_file():
    pytest.importorskip('h5py')
    from function import MODEL_JSON, MODEL_WEIGHTS

    from_h5 = SignModel.from_h5(MODEL_JSON, MODEL_WEIGHTS)
    shipped = SignModel.load(MODEL_NPZ)
    assert from_h5.spec == shipped.spec
    for a, b in zip(from_h5.weights, shipped.weights):
        assert a.keys() == b.keys()
        assert all(np.array_equal(a[k], b[k]) for k in a)


def test_describes_its_layers():
    rows = SignModel.load(MODEL_NPZ).layers()
    assert [shape for _, shape, _ in rows] == [(None, 30, 64), (None, 30, 128), (None, 64), (None, 64), (None, 32),
                                               (None, len(actions))]
    assert sum(params for _, _, params in rows) == 187_496


def test_loading_never_unpickles(tmp_path):
    path = tmp_path / 'bad.npz'
    np.savez(path, spec=np.array([{'type': 'lstm'}], dtype=object))
    with pytest.raises(ValueError):
        SignModel.load(path)
