from function import *
from time import sleep

# Create directories for each action and sequence
for action in actions: 
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

# Initialize MediaPipe Hands
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
    # NEW LOOP
    # Loop through actions
    for action in actions:
        # Loop through sequences aka videos
        for sequence in range(no_sequences):
            # Loop through video length aka sequence length
            for frame_num in range(sequence_length):

                # Read image from disk (simulating camera feed or processing collected images)
                # Note: Ensure the 'Image' directory uses the same 'action' naming convention (Case sensitive)
                # It seems 'Image' folder has subfolders like 'A', 'B' but actions are 'a', 'b'... 
                # or 'answer', 'bye' etc. 
                # Verify that actions array matches folder names in Image/
                
                # Attempting to read matching the current action
                # If your Image folder has Uppercase and actions are Lowercase, this might fail.
                # Assuming specific path structure based on previous code:
                try:
                    frame = cv2.imread('Image/{}/{}.png'.format(action, sequence))
                    if frame is None:
                        # Fallback for capitalization differences if needed, or just skip
                        # Try upper case for action folder if lower fails
                        frame = cv2.imread('Image/{}/{}.png'.format(action.capitalize(), sequence))
                    
                    if frame is None:
                        # Fallback for original code path structure 'Image/A/0.png' etc
                        # If actions are words like 'glad', 'hi' -> check those
                        # If logic is strictly reading from 'Image', skipping if file not found
                        continue 
                        
                except Exception as e:
                    continue

                # Make detections
                image, results = mediapipe_detection(frame, hands)

                # Draw landmarks
                draw_styled_landmarks(image, results)
                
                # Apply wait logic / UI feedback
                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(200)
                else: 
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                
                # Export keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                    
    cv2.destroyAllWindows()


