# Prints the layers and parameter counts of the trained model.
from function import load_trained_model

model = load_trained_model()

print("Layer Information:")
for layer in model.layers:
    print(layer.name)
    print("Output Shape:", layer.output_shape)
    print("Number of Parameters:", layer.count_params())
    print()

print("Model Summary:")
model.summary()
