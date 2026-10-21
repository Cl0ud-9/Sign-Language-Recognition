# Security policy

## Reporting a vulnerability

Please do not open a public issue for a security problem.

Report it privately through GitHub: open the [Security tab](https://github.com/Cl0ud-9/Sign-Language-Recognition/security)
and choose **Report a vulnerability**, or go straight to
[the advisory form](https://github.com/Cl0ud-9/Sign-Language-Recognition/security/advisories/new). Only the maintainer
sees what you send.

Please include what you found, how to reproduce it, and what you think an attacker could do with it. The maintainer
aims to answer within a week, and you will be credited in the fix unless you would rather not be.

## What counts

The project runs on your own computer and does not talk to a network. The risky part is the files it loads:

- A way for a crafted keypoint file (`.npy`) to run code or crash training. Keypoints are loaded with pickling off for
  this reason.
- A way for a crafted `model.json` or `model.h5` to run code when `app.py`, `evaluate.py` or `info.py` loads it.
- Anything in the scripts that writes outside the project folder without being asked to.

Not in scope: problems in TensorFlow, MediaPipe or OpenCV themselves (report those upstream), and files you chose to
run yourself.

## Keeping yourself safe

- Only use model files from this repository, or ones you trained yourself. A Keras model file is code as much as data.
- Download the dataset from its [Kaggle page](https://www.kaggle.com/datasets/paragtonpe/sign-language-recognition-dataset),
  not from a copy someone sent you.

## Supported versions

Only the current `main` gets fixes.
