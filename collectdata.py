# Based on kumarvivek9088/SignLanguageDetectionUsingML (GPL-3.0). Modified by Cloud/9, 2024-2026: see NOTICE.
# Saves webcam crops into Image/<label>/<n>.png.
#   a-z  save the crop as that letter (into Image/A ... Image/Z)
#   1-8  save the crop as a word: 1 name, 2 hi, 3 is, 4 my, 5 glad, 6 to, 7 what, 8 your
#   Esc  quit
import os
import string

import cv2

from function import IMAGE_PATH

WORD_KEYS = {'1': 'name', '2': 'hi', '3': 'is', '4': 'my', '5': 'glad', '6': 'to', '7': 'what', '8': 'your'}
LABELS = {c: c.upper() for c in string.ascii_lowercase}
LABELS.update(WORD_KEYS)

for folder in LABELS.values():
    os.makedirs(os.path.join(IMAGE_PATH, folder), exist_ok=True)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.rectangle(frame, (0, 40), (300, 400), (255, 255, 255), 2)
    cv2.imshow("data", frame)
    roi = frame[40:400, 0:300]
    cv2.imshow("ROI", roi)

    key = cv2.waitKey(10) & 0xFF
    if key == 27:
        break
    folder = LABELS.get(chr(key)) if key < 128 else None
    if folder:
        path = os.path.join(IMAGE_PATH, folder)
        cv2.imwrite(os.path.join(path, '{}.png'.format(len(os.listdir(path)))), roi)

cap.release()
cv2.destroyAllWindows()
