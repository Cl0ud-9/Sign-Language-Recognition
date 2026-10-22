"""The trained sign model, run with NumPy alone.

The model is three LSTM layers and three dense layers, trained with Keras (see trainmodel.py). Recognising signs only
needs its forward pass, which is a few matrix products per frame, so the app does not need TensorFlow. The weights
live in model.npz; tests/test_sign_model.py checks this implementation against outputs computed by Keras.

Keras LSTM, per time step (gates in the order input, forget, cell, output):
    z = x @ kernel + h @ recurrent_kernel + bias
    i, f, o = sigmoid(z_i), sigmoid(z_f), sigmoid(z_o)
    c = f * c + i * activation(z_c)
    h = o * activation(c)
"""
import json

import numpy as np

ACTIVATIONS = {
    'relu': lambda x: np.maximum(x, 0),
    'tanh': np.tanh,
    'sigmoid': lambda x: 0.5 * (1 + np.tanh(0.5 * x)),  # the same function, without overflow for large inputs
    'linear': lambda x: x,
}


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class SignModel:
    """A stack of LSTM and dense layers described by `spec` (one dict per layer) with its weights."""

    def __init__(self, spec, weights):
        self.spec = spec
        self.weights = weights

    # ---------- running

    def predict(self, x, verbose=0):
        """Probabilities for a batch of sequences shaped (batch, frames, 63). `verbose` matches Keras and is ignored."""
        out = np.asarray(x, dtype=np.float32)
        for n, layer in enumerate(self.spec):
            w = self.weights[n]
            if layer['type'] == 'lstm':
                out = self._lstm(out, w, layer)
            else:
                out = out @ w['kernel'] + w['bias']
                out = softmax(out) if layer['activation'] == 'softmax' else ACTIVATIONS[layer['activation']](out)
        return out

    @staticmethod
    def _lstm(x, w, layer):
        act = ACTIVATIONS[layer['activation']]
        gate = ACTIVATIONS[layer['recurrent_activation']]
        units = layer['units']
        batch, steps, _ = x.shape
        h = np.zeros((batch, units), dtype=np.float32)
        c = np.zeros((batch, units), dtype=np.float32)
        projected = x @ w['kernel'] + w['bias']  # the input part of every step at once
        outputs = []
        for t in range(steps):
            z = projected[:, t] + h @ w['recurrent_kernel']
            i, f, g, o = (z[:, k * units:(k + 1) * units] for k in range(4))
            c = gate(f) * c + gate(i) * act(g)
            h = gate(o) * act(c)
            outputs.append(h)
        return np.stack(outputs, axis=1) if layer['return_sequences'] else h

    # ---------- describing

    def layers(self):
        """(name, output shape, parameter count) for each layer, like Keras' summary."""
        rows, steps = [], None
        for n, layer in enumerate(self.spec):
            params = sum(int(a.size) for a in self.weights[n].values())
            if layer['type'] == 'lstm':
                steps = steps or layer.get('steps')
                shape = (None, steps, layer['units']) if layer['return_sequences'] else (None, layer['units'])
            else:
                shape = (None, layer['units'])
            rows.append((f"{layer['type']}_{n}", shape, params))
        return rows

    # ---------- loading and saving

    def save(self, path):
        arrays = {f'{n}_{name}': a.astype(np.float32) for n, w in enumerate(self.weights) for name, a in w.items()}
        np.savez(path, spec=np.array(json.dumps(self.spec)), **arrays)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            spec = json.loads(str(data['spec']))
            weights = [{} for _ in spec]
            for key in data.files:
                if key != 'spec':
                    n, name = key.split('_', 1)
                    weights[int(n)][name] = data[key]
        return cls(spec, weights)

    @classmethod
    def from_keras(cls, model):
        """Build from a Keras model in memory (used by trainmodel.py after training)."""
        spec, weights = [], []
        steps = model.inputs[0].shape[1]
        for layer in model.layers:
            config = layer.get_config()
            values = layer.get_weights()
            if type(layer).__name__ == 'LSTM':
                entry = _lstm_spec(config, None)
                entry['steps'] = steps
                spec.append(entry)
                weights.append(dict(zip(('kernel', 'recurrent_kernel', 'bias'), values)))
            elif type(layer).__name__ == 'Dense':
                spec.append({'type': 'dense', 'units': config['units'], 'activation': config['activation']})
                weights.append(dict(zip(('kernel', 'bias'), values)))
        return cls(spec, weights)

    @classmethod
    def from_h5(cls, json_path, h5_path):
        """Build from model.json and a Keras 2 model.h5, without TensorFlow (needs h5py)."""
        import h5py

        with open(json_path) as f:
            layers = json.load(f)['config']['layers']
        spec, weights = [], []
        steps = None
        with h5py.File(h5_path, 'r') as h5:
            group = h5['model_weights'] if 'model_weights' in h5 else h5
            for layer in layers:
                kind, config = layer['class_name'], layer['config']
                if kind == 'InputLayer':
                    steps = config['batch_input_shape'][1]
                    continue
                if config.get('batch_input_shape'):
                    steps = config['batch_input_shape'][1]
                names = [n.decode() if isinstance(n, bytes) else n for n in group[config['name']].attrs['weight_names']]
                # Names look like 'lstm/lstm_cell/recurrent_kernel:0'; match on the last part, not on the order
                values = {n.rsplit('/', 1)[-1].split(':')[0]: np.array(group[config['name']][n]) for n in names}
                if kind == 'LSTM':
                    entry = _lstm_spec(config, None)
                    entry['steps'] = steps
                    spec.append(entry)
                    weights.append({k: values[k] for k in ('kernel', 'recurrent_kernel', 'bias')})
                elif kind == 'Dense':
                    spec.append({'type': 'dense', 'units': config['units'], 'activation': config['activation']})
                    weights.append({k: values[k] for k in ('kernel', 'bias')})
                else:
                    raise ValueError(f'Unsupported layer {kind}')
        return cls(spec, weights)


def _lstm_spec(config, input_shape):
    unsupported = [k for k in ('go_backwards', 'stateful', 'unroll') if config.get(k)]
    if unsupported or config.get('dropout') or config.get('recurrent_dropout'):
        raise ValueError(f'Unsupported LSTM options: {unsupported}')
    spec = {'type': 'lstm', 'units': config['units'], 'activation': config['activation'],
            'recurrent_activation': config['recurrent_activation'],
            'return_sequences': bool(config['return_sequences'])}
    if input_shape and len(input_shape) == 3:
        spec['steps'] = input_shape[1]
    return spec
