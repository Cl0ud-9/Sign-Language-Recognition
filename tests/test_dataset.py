import os

import numpy as np
import pytest

from function import NUM_KEYPOINTS, actions, load_dataset, load_frame, no_sequences, sequence_length


def write_mp_data(root, frame=lambda a, s, f: np.full(NUM_KEYPOINTS, 0.5)):
    for action in actions:
        for sequence in range(no_sequences):
            folder = os.path.join(root, action, str(sequence))
            os.makedirs(folder)
            for frame_num in range(sequence_length):
                np.save(os.path.join(folder, str(frame_num)), frame(action, sequence, frame_num))


def test_loads_every_sequence_with_its_label(tmp_path):
    write_mp_data(tmp_path)
    X, y = load_dataset(str(tmp_path))
    assert X.shape == (len(actions) * no_sequences, sequence_length, NUM_KEYPOINTS)
    assert X.dtype == np.float32
    assert np.bincount(y).tolist() == [no_sequences] * len(actions)


def test_frame_saved_as_none_becomes_zeros(tmp_path):
    # MP_Data/hi/0/0.npy in the real dataset is a pickled None: training used to crash on it
    write_mp_data(tmp_path, lambda a, s, f: None if (a, s, f) == ('hi', 0, 0) else np.full(NUM_KEYPOINTS, 0.5))
    X, y = load_dataset(str(tmp_path))
    first_hi = X[list(y).index(list(actions).index('hi'))]
    assert not first_hi[0].any()
    assert first_hi[1:].all()


def test_pickled_frames_are_never_unpickled(tmp_path):
    path = tmp_path / 'frame.npy'
    np.save(path, np.array({'not': 'keypoints'}, dtype=object), allow_pickle=True)
    assert not load_frame(str(path)).any()


def test_incomplete_sequence_is_skipped(tmp_path):
    write_mp_data(tmp_path)
    os.remove(os.path.join(tmp_path, 'to', '3', '29.npy'))
    X, y = load_dataset(str(tmp_path))
    assert len(X) == len(actions) * no_sequences - 1


@pytest.mark.parametrize('bad', [np.zeros(10), np.zeros((2, NUM_KEYPOINTS))])
def test_wrong_shape_frame_becomes_zeros(tmp_path, bad):
    path = tmp_path / 'frame.npy'
    np.save(path, bad)
    assert load_frame(str(path)).shape == (NUM_KEYPOINTS,)
