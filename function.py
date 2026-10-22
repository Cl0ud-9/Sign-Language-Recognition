# Based on kumarvivek9088/SignLanguageDetectionUsingML (GPL-3.0). Modified by Cloud/9, 2024-2026: see NOTICE.
import argparse
import os
import sys
from collections import deque

# TensorFlow (for training) and MediaPipe's TensorFlow Lite runtime print start-up notices that would bury the
# scripts' own messages; errors still show
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import cv2  # noqa: E402
import mediapipe as mp  # noqa: E402
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Paths are relative to this file, so the scripts work from any working directory
ROOT = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(ROOT, 'Image')
DATA_PATH = os.path.join(ROOT, 'MP_Data')
LOG_PATH = os.path.join(ROOT, 'Logs')
MODEL_JSON = os.path.join(ROOT, 'model.json')
MODEL_WEIGHTS = os.path.join(ROOT, 'model.h5')
# The same weights for NumPy: what the app and evaluate.py load (see sign_model.py)
MODEL_NPZ = os.path.join(ROOT, 'model.npz')

# Signs the model recognises, in the order of its output layer
actions = np.array(['to', 'my', 'glad', 'hi', 'is', 'what', 'your', 'name'])

# 30 sequences per sign, 30 frames per sequence
no_sequences = 30
sequence_length = 30

# 21 hand landmarks, each (x, y, z)
NUM_KEYPOINTS = 21 * 3


def camera_arguments(description):
    """Parse the scripts' one option: which camera to use."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--camera', type=int, default=0, help='camera number, if you have more than one (default 0)')
    return parser.parse_args()


def open_camera(index=0):
    """Open a webcam, or stop with a message that says what to check."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        other = 1 if index == 0 else 0
        sys.exit(f"Could not open camera {index}. Check that it is connected, not in use by another app, and allowed "
                 f"in your privacy settings. To use a different camera, run with --camera {other}.")
    return cap


def mediapipe_detection(image, model):
    """Run MediaPipe on a BGR frame. Returns the frame and the detection results."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_styled_landmarks(image, results):
    """Draw every detected hand's landmarks onto the frame."""
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())


def extract_keypoints(results):
    """Flatten the first detected hand into 63 values, or return None when no hand is visible."""
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        return np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark]).flatten()
    return None


# The held-out split, shared by trainmodel.py and evaluate.py so both use the same sequences
TEST_SIZE = 0.05
SPLIT_SEED = 42


def split(X, y):
    from sklearn.model_selection import train_test_split

    return train_test_split(X, y, test_size=TEST_SIZE, random_state=SPLIT_SEED, stratify=y)


def load_trained_model():
    """Load the trained model from model.npz. It runs on NumPy, so recognising signs does not need TensorFlow."""
    from sign_model import SignModel

    return SignModel.load(MODEL_NPZ)


class SignSmoother:
    """Turn per-frame predictions into a steady answer.

    A sign is shown once the last `agree` predictions all name it and the newest is more than `threshold` sure. It
    stays shown, with its latest confidence, until another sign takes its place or reset() is called.
    """

    def __init__(self, agree=10, threshold=0.8):
        self.threshold = threshold
        self.recent = deque(maxlen=agree)
        self.sign = None
        self.confidence = 0.0

    def update(self, probs):
        best = int(np.argmax(probs))
        self.recent.append(best)
        steady = len(self.recent) == self.recent.maxlen and len(set(self.recent)) == 1
        if steady and probs[best] > self.threshold:
            self.sign, self.confidence = str(actions[best]), float(probs[best])
        elif self.sign == actions[best]:
            self.confidence = float(probs[best])
        return self.sign

    def reset(self):
        self.recent.clear()
        self.sign = None
        self.confidence = 0.0


def load_frame(path):
    """Load one keypoint frame, or zeros if MediaPipe found no hand in it.

    Older data saved a missing hand as a pickled None. Pickles are never loaded, since a crafted
    .npy file could run code: such a frame is treated as a missing hand instead.
    """
    try:
        frame = np.load(path, allow_pickle=False)
    except ValueError:
        return np.zeros(NUM_KEYPOINTS)
    if frame.shape != (NUM_KEYPOINTS,):
        return np.zeros(NUM_KEYPOINTS)
    return frame


def load_dataset(data_path=DATA_PATH):
    """Load MP_Data as X (sequences, 30, 63) and integer labels y.

    Frames without a hand become all zeros so every sequence keeps the same shape.
    Sequences with missing files are skipped.
    """
    sequences, labels, skipped = [], [], 0
    for label, action in enumerate(actions):
        for sequence in range(no_sequences):
            window = []
            for frame_num in range(sequence_length):
                path = os.path.join(data_path, action, str(sequence), '{}.npy'.format(frame_num))
                if not os.path.exists(path):
                    break
                window.append(load_frame(path))
            if len(window) != sequence_length:
                skipped += 1
                continue
            sequences.append(window)
            labels.append(label)
    if skipped:
        print('Skipped {} incomplete sequences'.format(skipped))
    return np.array(sequences, dtype='float32'), np.array(labels)
