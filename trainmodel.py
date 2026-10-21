# Based on kumarvivek9088/SignLanguageDetectionUsingML (GPL-3.0). Modified by Cloud/9, 2024-2026: see NOTICE.
# Trains the LSTM on MP_Data/ and writes model.json and model.h5.
import argparse

from keras.callbacks import TensorBoard
from keras.layers import LSTM, Dense
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from function import LOG_PATH, MODEL_JSON, MODEL_WEIGHTS, NUM_KEYPOINTS, actions, load_dataset, sequence_length

# Shared with evaluate.py so the held-out sequences are the same in both
TEST_SIZE = 0.05
SPLIT_SEED = 42


def build_model():
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(sequence_length, NUM_KEYPOINTS)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model


def split(X, y):
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=SPLIT_SEED, stratify=y)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the sign-language LSTM on MP_Data/.')
    parser.add_argument('--epochs', type=int, default=200)
    args = parser.parse_args()

    X, y = load_dataset()
    X_train, X_test, y_train, y_test = split(X, y)

    model = build_model()
    model.fit(X_train, to_categorical(y_train, len(actions)), epochs=args.epochs,
              callbacks=[TensorBoard(log_dir=LOG_PATH)])
    model.summary()

    _, test_acc = model.evaluate(X_test, to_categorical(y_test, len(actions)), verbose=0)
    print('Held-out accuracy: {:.1%} on {} sequences'.format(test_acc, len(X_test)))

    with open(MODEL_JSON, 'w') as json_file:
        json_file.write(model.to_json())
    # A full model file: app.py reads its weights, info.py loads it whole
    model.save(MODEL_WEIGHTS)
