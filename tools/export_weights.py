"""Write model.npz from model.json and model.h5, without TensorFlow.

usage: python tools/export_weights.py      (needs h5py: pip install -r requirements-dev.txt)
trainmodel.py writes model.npz itself; this converts a model.h5 that was trained before it did.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from function import MODEL_JSON, MODEL_NPZ, MODEL_WEIGHTS  # noqa: E402
from sign_model import SignModel  # noqa: E402

model = SignModel.from_h5(MODEL_JSON, MODEL_WEIGHTS)
model.save(MODEL_NPZ)
for name, shape, params in model.layers():
    print(f'{name:<10} {str(shape):<18} {params:>8,}')
print(f'Wrote {os.path.relpath(MODEL_NPZ, ROOT)}')
