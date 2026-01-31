from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import TensorBoard

# ... (rest of your code for data loading and preprocessing) ...

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# Define TensorBoard callback (optional, for visualization)
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Build the LSTM model
model = Sequential()
# ... (rest of your model architecture) ...

# Compile the model with categorical crossentropy loss and desired metrics (accuracy)
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback])

# Evaluate the model on the testing set and print accuracy
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_acc)

# Save the model (optional)
# ... (your model saving code) ...
