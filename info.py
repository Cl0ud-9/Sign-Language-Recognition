import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model('model.h5')

# Print the type of model
print("Model Type:")
print(type(model))

# Print layer information
print("\nLayer Information:")
for layer in model.layers:
    print(layer.name)
    print("Output Shape:", layer.output_shape)
    print("Number of Parameters:", layer.count_params())
    print()

# Print model summary
print("\nModel Summary:")
model.summary()
