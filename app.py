# Based on kumarvivek9088/SignLanguageDetectionUsingML (GPL-3.0). Modified by Cloud/9, 2024-2026: see NOTICE.
import time

import cv2
import numpy as np

from function import (
    SignSmoother,
    camera_arguments,
    draw_styled_landmarks,
    extract_keypoints,
    load_trained_model,
    mediapipe_detection,
    mp_hands,
    open_camera,
    sequence_length,
)

# The region of the frame that is read, as (left, top, right, bottom): the same box the training images were cut from
BOX = (0, 40, 300, 400)
WHITE, GREEN, DARK = (255, 255, 255), (80, 200, 120), (30, 24, 20)


def draw_hud(frame, sign, confidence, status, fps):
    """A panel along the top: the recognised sign, a confidence bar, what to do next, and the frame rate."""
    height, width = frame.shape[:2]
    panel = frame.copy()
    cv2.rectangle(panel, (0, 0), (width, 34), DARK, -1)
    cv2.rectangle(panel, (0, height - 30), (width, height), DARK, -1)
    cv2.addWeighted(panel, 0.75, frame, 0.25, 0, frame)
    text = f'{sign}  {confidence:.0%}' if sign else 'Waiting for a sign'
    cv2.putText(frame, text, (BOX[2] + 12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, WHITE, 2, cv2.LINE_AA)
    if sign:
        bar = int(160 * confidence)
        cv2.rectangle(frame, (width - 175, 12), (width - 15, 22), (90, 90, 90), -1)
        cv2.rectangle(frame, (width - 175, 12), (width - 175 + bar, 22), GREEN, -1)
    cv2.putText(frame, status, (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, f'{fps:.0f} fps   q quits', (width - 140, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1,
                cv2.LINE_AA)


def main():
    args = camera_arguments('Recognise signs live from a webcam. Press q to quit.')
    # Open the camera before loading the model, so a missing camera is reported straight away
    cap = open_camera(args.camera)
    model = load_trained_model()

    window = []  # keypoints of the latest frames, newest last
    smoother = SignSmoother()
    left, top, right, bottom = BOX
    last = time.perf_counter()
    fps = 0.0

    with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                print('The camera stopped sending frames.')
                break

            crop = frame[top:bottom, left:right]  # a view: drawing on it draws on the frame
            _, results = mediapipe_detection(crop, hands)
            keypoints = extract_keypoints(results)

            if keypoints is None:
                # The model reads 30 frames of one hand; without a hand, start the window again
                window.clear()
                smoother.reset()
                status = 'Show your hand inside the box'
            else:
                draw_styled_landmarks(crop, results)
                window = (window + [keypoints])[-sequence_length:]
                if len(window) == sequence_length:
                    probs = model.predict(np.expand_dims(window, axis=0))[0]
                    smoother.update(probs)
                    status = 'Hold the sign steady' if smoother.sign is None else 'Recognised'
                else:
                    status = f'Reading your hand... {len(window)}/{sequence_length}'

            cv2.rectangle(frame, (left, top), (right, bottom), GREEN if keypoints is not None else WHITE, 2)
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
            last = now
            draw_hud(frame, smoother.sign, smoother.confidence, status, fps)
            cv2.imshow('Sign Language Recognition', frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
