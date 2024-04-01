from function import *
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import TensorBoard

label_map = {label: num for num, label in enumerate(actions)}

sequences, labels = [], []

# Determine the shape of a single frame (assuming they all have the same shape)
sample_frame = np.load(os.path.join(DATA_PATH, actions[0], "0", "0.npy"), allow_pickle=True)
frame_shape = sample_frame.shape

print("Frame Shape:", frame_shape)

for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)), allow_pickle=True)
            
            # Ensure each frame has a consistent shape
            if res.shape != frame_shape:
                # Handle this according to your requirements (e.g., resizing, discarding, etc.)
                print(f"Warning: Frame shape is not consistent for {action}/{sequence}/{frame_num}")
                continue
            
            window.append(res)
        
        sequences.append(window)
        labels.append(label_map[action])

# Convert sequences to a 2D array
sequences_2d = [np.vstack(window) for window in sequences]

# Pad sequences to have the same length
X = pad_sequences(sequences_2d, padding='post', dtype='float32')
y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()

# Print the frame shape for debugging
print("Frame Shape for Input Layer:", frame_shape)

model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(sequence_length, frame_shape[0])))  # Adjust the index here
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

res = [.7, 0.2, 0.1]

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback])
model.summary()

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save('model.h5')
