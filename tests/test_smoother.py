import numpy as np

from function import SignSmoother, actions


def probs_for(sign, confidence):
    p = np.full(len(actions), (1 - confidence) / (len(actions) - 1))
    p[list(actions).index(sign)] = confidence
    return p


def test_needs_ten_agreeing_confident_predictions():
    smoother = SignSmoother()
    for _ in range(9):
        assert smoother.update(probs_for('hi', 0.95)) is None
    assert smoother.update(probs_for('hi', 0.95)) == 'hi'
    assert abs(smoother.confidence - 0.95) < 1e-9


def test_unsure_predictions_are_not_shown():
    smoother = SignSmoother()
    for _ in range(20):
        smoother.update(probs_for('my', 0.6))
    assert smoother.sign is None


def test_one_disagreeing_frame_delays_a_new_sign():
    smoother = SignSmoother()
    for _ in range(10):
        smoother.update(probs_for('hi', 0.9))
    for _ in range(5):
        smoother.update(probs_for('name', 0.9))
    smoother.update(probs_for('what', 0.9))
    for _ in range(9):
        smoother.update(probs_for('name', 0.9))
    assert smoother.sign == 'hi'
    smoother.update(probs_for('name', 0.9))
    assert smoother.sign == 'name'


def test_reset_forgets_the_sign():
    smoother = SignSmoother()
    for _ in range(10):
        smoother.update(probs_for('to', 0.99))
    smoother.reset()
    assert smoother.sign is None and smoother.confidence == 0
