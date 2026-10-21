# Based on kumarvivek9088/SignLanguageDetectionUsingML (GPL-3.0). Modified by Cloud/9, 2024-2026: see NOTICE.
# Turns the images in Image/ into MediaPipe keypoints in MP_Data/.
# Sequence n of a sign is built from Image/<sign>/<n>.png: the same picture is run through
# MediaPipe for all 30 frames, so the model learns hand shapes rather than motion.
import os

import cv2
import numpy as np

from function import (
    DATA_PATH,
    IMAGE_PATH,
    NUM_KEYPOINTS,
    actions,
    draw_styled_landmarks,
    extract_keypoints,
    mediapipe_detection,
    mp_hands,
    no_sequences,
    sequence_length,
)

for action in actions:
    for sequence in range(no_sequences):
        os.makedirs(os.path.join(DATA_PATH, action, str(sequence)), exist_ok=True)

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    for action in actions:
        for sequence in range(no_sequences):
            frame = cv2.imread(os.path.join(IMAGE_PATH, action, '{}.png'.format(sequence)))
            if frame is None:
                print('Missing Image/{}/{}.png, skipping this sequence'.format(action, sequence))
                continue

            for frame_num in range(sequence_length):
                image, results = mediapipe_detection(frame, hands)
                draw_styled_landmarks(image, results)

                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15, 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', image)
                cv2.waitKey(200 if frame_num == 0 else 1)

                # A frame with no hand is saved as zeros, the same as load_dataset() reads it
                keypoints = extract_keypoints(results)
                if keypoints is None:
                    keypoints = np.zeros(NUM_KEYPOINTS)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

    cv2.destroyAllWindows()
