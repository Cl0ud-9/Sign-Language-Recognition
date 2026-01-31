import os
import cv2
import string

# Define the directory where images are stored
directory = 'Image/'

# Create directory if it doesn't exist
if not os.path.exists(directory):
    os.makedirs(directory)

# Define the list of actions/labels to collect
# Single letters a-z
labels = list(string.ascii_lowercase)
# Additional words
words = ['hi', 'glad', 'is', 'my', 'name', 'to', 'what', 'your']
labels.extend(words)

# Ensure subdirectories exist
for label in labels:
    path = os.path.join(directory, label)
    if not os.path.exists(path):
        os.makedirs(path)

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    
    # Get current count of images for each label
    count = {}
    for label in labels:
        # construct path, handling potential case sensitivity if folders are capitalized differently
        # Assuming folders are named exactly as labels for simplicity, or Capitalized for letters
        # based on original code 'A', 'B' etc.
        if len(label) == 1:
            path = os.path.join(directory, label.upper())
        else:
            path = os.path.join(directory, label)
            
        if not os.path.exists(path):
             os.makedirs(path)
             
        count[label] = len(os.listdir(path))

    # UI Layout
    row = frame.shape[1]
    col = frame.shape[0]
    
    # Draw ROI (Region of Interest)
    cv2.rectangle(frame, (0, 40), (300, 400), (255, 255, 255), 2)
    
    cv2.imshow("data", frame)
    
    # Extract ROI
    roi = frame[40:400, 0:300]
    cv2.imshow("ROI", roi)
    
    interrupt = cv2.waitKey(10)
    
    # Exit loop
    if interrupt & 0xFF == ord('q'): # Added exit condition
        break

    # Check for key presses to save images
    # 0-9 keys for words, a-z keys for letters
    
    # Logic for letters a-z
    if interrupt >= ord('a') and interrupt <= ord('z'):
        char = chr(interrupt)
        if char in labels:
            path = os.path.join(directory, char.upper())
            cv2.imwrite(os.path.join(path, str(count[char]) + '.png'), roi)
    
    # Logic for words (mapped to numbers 1-8 based on original code)
    # Original mapping:
    # 1: name, 2: hi, 3: is, 4: my, 5: glad, 6: to, 7: what, 8: your
    word_map = {
        '1': 'name',
        '2': 'hi',
        '3': 'is',
        '4': 'my',
        '5': 'glad',
        '6': 'to',
        '7': 'what',
        '8': 'your'
    }
    
    if interrupt & 0xFF in [ord(c) for c in word_map.keys()]:
        key = chr(interrupt & 0xFF)
        label = word_map[key]
        path = os.path.join(directory, label)
        cv2.imwrite(os.path.join(path, str(count[label]) + '.png'), roi)

cap.release()
cv2.destroyAllWindows()