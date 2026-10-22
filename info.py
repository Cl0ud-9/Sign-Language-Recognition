# Prints the layers and parameter counts of the trained model (from model.npz, so TensorFlow is not needed).
from function import actions, load_trained_model

model = load_trained_model()

print('{:<10} {:<18} {:>10}'.format('Layer', 'Output shape', 'Parameters'))
total = 0
for name, shape, params in model.layers():
    print('{:<10} {:<18} {:>10,}'.format(name, str(shape), params))
    total += params
print()
print('Total parameters: {:,}'.format(total))
print('Signs, in output order:', ', '.join(actions))
