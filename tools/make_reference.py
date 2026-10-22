"""Write tests/fixtures/reference.npz: inputs and the probabilities Keras gives for them.

usage: python tools/make_reference.py      (needs TensorFlow and h5py; any TensorFlow that reads Keras 2 .h5 weights)
tests/test_sign_model.py checks that the NumPy model gives the same answers, so the app can run without Keras.
This script imports only Keras and NumPy, so it runs in a TensorFlow environment without MediaPipe or OpenCV.
"""
import os

import keras
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAMES, KEYPOINTS, SIGNS = 30, 63, 8

# The architecture in trainmodel.build_model(); CI's train job checks the two agree
model = keras.Sequential([
    keras.Input((FRAMES, KEYPOINTS)),
    keras.layers.LSTM(64, return_sequences=True, activation='relu'),
    keras.layers.LSTM(128, return_sequences=True, activation='relu'),
    keras.layers.LSTM(64, return_sequences=False, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(SIGNS, activation='softmax'),
])
model.load_weights(os.path.join(ROOT, 'model.h5'))

rng = np.random.default_rng(7)
# Landmark coordinates lie roughly in [0, 1] (x, y) and [-0.2, 0.2] (z). Also the shapes the app really sends: a still
# hand repeated for 30 frames, and an all-zero window
random = rng.uniform(0, 1, (12, FRAMES, KEYPOINTS))
random[..., 2::3] = rng.uniform(-0.2, 0.2, random[..., 2::3].shape)
still = np.repeat(random[:4, :1], FRAMES, axis=1)
x = np.concatenate([random, still, np.zeros((1, FRAMES, KEYPOINTS))]).astype(np.float32)

probs = model.predict(x, verbose=0)
os.makedirs(os.path.join(ROOT, 'tests', 'fixtures'), exist_ok=True)
np.savez(os.path.join(ROOT, 'tests', 'fixtures', 'reference.npz'), x=x, probs=probs.astype(np.float32))
print(f'Keras {keras.__version__}: wrote tests/fixtures/reference.npz, {x.shape} -> {probs.shape}')
print('predicted signs:', probs.argmax(axis=1).tolist())
