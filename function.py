# Based on kumarvivek9088/SignLanguageDetectionUsingML (GPL-3.0). Modified by Cloud/9, 2024-2026: see NOTICE.
import os

import cv2
import mediapipe as mp
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

# Signs the model recognises, in the order of its output layer
actions = np.array(['to', 'my', 'glad', 'hi', 'is', 'what', 'your', 'name'])

# 30 sequences per sign, 30 frames per sequence
no_sequences = 30
sequence_length = 30

# 21 hand landmarks, each (x, y, z)
NUM_KEYPOINTS = 21 * 3


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


def load_trained_model():
    """Load the trained LSTM from model.json and model.h5."""
    from keras.models import model_from_json

    with open(MODEL_JSON) as f:
        model = model_from_json(f.read())
    model.load_weights(MODEL_WEIGHTS)
    return model


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
