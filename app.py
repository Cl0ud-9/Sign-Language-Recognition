# Based on kumarvivek9088/SignLanguageDetectionUsingML (GPL-3.0). Modified by Cloud/9, 2024-2026: see NOTICE.
import cv2
import numpy as np

from function import actions, extract_keypoints, load_trained_model, mediapipe_detection, mp_hands, sequence_length

model = load_trained_model()

sequence = []
sentence = []
accuracy = []
predictions = []
threshold = 0.8

cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Only the boxed region on the left of the frame is read
        cropframe = frame[40:400, 0:300]
        frame = cv2.rectangle(frame, (0, 40), (300, 400), 255, 2)
        image, results = mediapipe_detection(cropframe, hands)

        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-sequence_length:]

        try:
            if len(sequence) == sequence_length:
                res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                print(actions[np.argmax(res)])
                predictions.append(np.argmax(res))

                # Accept a sign only when the last 10 predictions agree and the model is confident
                if len(predictions) >= 10 and len(np.unique(predictions[-10:])) == 1:
                    if res[np.argmax(res)] > threshold:
                        if len(sentence) > 0:
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                                accuracy.append(str(res[np.argmax(res)] * 100))
                        else:
                            sentence.append(actions[np.argmax(res)])
                            accuracy.append(str(res[np.argmax(res)] * 100))

                # Show only the latest sign
                if len(sentence) > 1:
                    sentence = sentence[-1:]
                    accuracy = accuracy[-1:]

        # A frame without a hand is None, so the window cannot be predicted until the hand is back
        except Exception:
            pass

        label = "Output: - " + ' '.join(sentence) + ' '.join([f" {float(acc):.1f}%" for acc in accuracy])
        cv2.putText(frame, label, (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('OpenCV Feed', frame)

        # Press q to quit
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
