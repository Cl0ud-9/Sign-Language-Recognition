from function import *
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import TensorBoard

# Create a mapping for labels
label_map = {label: num for num, label in enumerate(actions)}

sequences, labels = [], []

# OPTIONAL: Determine frame shape from first file if needed
# sample_frame = np.load(os.path.join(DATA_PATH, actions[0], "0", "0.npy"), allow_pickle=True)
# frame_shape = sample_frame.shape

# 1. Load Data
# Loop through all actions, sequences, and frames to load the extracted keypoints
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            # Load the keypoints for each frame
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)), allow_pickle=True)
            window.append(res)
        
        sequences.append(window)
        labels.append(label_map[action])

# 2. Preprocess Data
X = np.array(sequences)
y = to_categorical(labels).astype(int)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# Setup TensorBoard for monitoring
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# 3. Build LSTM Model
model = Sequential()
# LSTM layers with Relu activation
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 63))) # 30 frames, 63 keypoints (21 * 3)
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
# Dense layers for classification
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
# Output layer with Softmax for probability distribution
model.add(Dense(actions.shape[0], activation='softmax'))

# 4. Train Model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback])
model.summary()

# 5. Save Model
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save('model.h5')
