# Sign Language Detection

A real-time Sign Language Detection application using LSTM and MediaPipe. This project captures motion data from hand gestures and translates them into text/sign labels.

## Project Structure

- **app.py**: Main application entry point.
- **collectdata.py**: Script to collect data for training new signs.
- **trainmodel.py**: Script to train the LSTM model.
- **function.py**: Helper functions for MediaPipe detection.
- **MP_Data/**: Pre-processed numpy arrays used for training/inference.
- **Models/**: Contains trained model weights (`model.h5`, etc.).
- **docs/**: Project documentation (Reports, Research Paper, etc.).

## Dataset

The raw video dataset for this project is available on Kaggle:
[Link to Kaggle Dataset](Your-Kaggle-Dataset-Link-Here)

## How to Run

1.  **Install Dependencies**
    Ensure you have the required Python packages (TensorFlow, OpenCV, MediaPipe, etc.).
    ```bash
    pip install tensorflow opencv-python mediapipe numpy matplotlib
    ```

2.  **Run the Application**
    ```bash
    python app.py
    ```

## Training (Optional)

If you want to retrain the model on new data:
1.  Run `collectdata.py` to capture new sequences.
2.  Run `trainmodel.py` to train the LSTM model.
