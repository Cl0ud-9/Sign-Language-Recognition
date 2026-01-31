from function import *
from keras.utils import to_categorical
from keras.models import model_from_json
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard

# Load the model architecture
json_file = open("model.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
# Load weights
model.load_weights("model.h5")

# Define visualization colors
colors = [(245,117,16)] * 20

# 1. New detection variables
sequence = []
sentence = []
accuracy=[]
predictions = []
threshold = 0.8 

cap = cv2.VideoCapture(0)
# Set mediapipe model 
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()
        if not ret:
            break

        # Crop frame for processing
        cropframe=frame[40:400,0:300]
        
        # Draw Region of Interest (ROI)
        frame=cv2.rectangle(frame,(0,40),(300,400),255,2)
        
        # Make detections
        image, results = mediapipe_detection(cropframe, hands)
        
        # 2. Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:] # Keep last 30 frames

        try: 
            if len(sequence) == 30:
                # Predict
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(actions[np.argmax(res)])
                predictions.append(np.argmax(res))
                
                # 3. Visualization logic
                # Ensure stability: prediction must be consistent for last 10 frames
                if np.unique(predictions[-10:])[0]==np.argmax(res): 
                    # Check confidence threshold
                    if res[np.argmax(res)] > threshold: 
                        if len(sentence) > 0: 
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                                accuracy.append(str(res[np.argmax(res)]*100))
                        else:
                            sentence.append(actions[np.argmax(res)])
                            accuracy.append(str(res[np.argmax(res)]*100)) 

                # Limit sentence length
                if len(sentence) > 1: 
                    sentence = sentence[-1:]
                    accuracy=accuracy[-1:]

        except Exception as e:
            pass
            
        # Display Output
        cv2.putText(frame, "Output: - " + ' '.join(sentence) + ' '.join([f" {float(acc):.1f}%" for acc in accuracy]), (3, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Show to screen
        cv2.imshow('OpenCV Feed', frame)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()