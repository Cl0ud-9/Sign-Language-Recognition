# Measures the trained model on MP_Data/: accuracy, a per-sign report and a confusion matrix.
import argparse

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from function import actions, load_dataset, load_trained_model, split

parser = argparse.ArgumentParser(description='Evaluate model.json + model.h5 on MP_Data/.')
parser.add_argument('--all', action='store_true',
                    help='score every sequence instead of the held-out split used by trainmodel.py')
args = parser.parse_args()

X, y = load_dataset()
if not args.all:
    _, X, _, y = split(X, y)

model = load_trained_model()
pred = np.argmax(model.predict(X, verbose=0), axis=1)

print('{}: {:.1%} accuracy on {} sequences\n'.format(
    'All data' if args.all else 'Held-out split', accuracy_score(y, pred), len(y)))
print(classification_report(y, pred, labels=range(len(actions)), target_names=actions, zero_division=0))
print('Confusion matrix (rows: true sign, columns: predicted):')
print('       ' + ' '.join('{:>5}'.format(a) for a in actions))
for name, row in zip(actions, confusion_matrix(y, pred, labels=range(len(actions)))):
    print('{:>6} '.format(name) + ' '.join('{:>5}'.format(v) for v in row))
