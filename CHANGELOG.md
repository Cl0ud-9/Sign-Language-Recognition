# Changelog

What changed in each version of Sign Language Recognition.

The format follows [Keep a Changelog](https://keepachangelog.com), and the versions follow
[Semantic Versioning](https://semver.org).

## [1.2.0] - 2026-10-22

### Changed

- The app runs the model on NumPy instead of TensorFlow. `sign_model.py` implements the LSTM's forward pass and loads
  `model.npz`; a test checks it against Keras' own outputs for the same inputs. Installing the app no longer
  downloads TensorFlow, and it now works on Python 3.12 as well as 3.10 and 3.11.
- Training moves to `requirements-train.txt` (TensorFlow 2.15, Python 3.10 or 3.11), and `trainmodel.py` also writes
  `model.npz`. `tools/export_weights.py` converts an existing `model.h5` without TensorFlow.
- The on-screen display: a panel with the recognised sign and a confidence bar, the hand's landmarks drawn in the
  box, a box that turns green when a hand is found, a line saying what the app is waiting for, and the frame rate.
- `info.py` describes the model without TensorFlow.

### Fixed

- `app.py` hid every error behind a catch-all, so a real failure looked like "no sign yet". Frames without a hand now
  restart the 30-frame window on purpose, and other errors are no longer swallowed.
- The confidence shown stayed at the value from when the sign was first recognised; it now follows the model.

## [1.1.1] - 2026-10-22

### Fixed

- `app.py` and `collectdata.py` closed without a word when the webcam could not be opened, for example when another
  app was using it or the privacy settings blocked it. They now say so and list what to check, and `app.py` checks
  the camera before spending time loading the model.

### Added

- `--camera` chooses which camera `app.py` and `collectdata.py` use, for computers with more than one.

### Changed

- TensorFlow's start-up notices are hidden, so the scripts' own messages are easy to see. Errors still show.

## [1.1.0] - 2026-10-21

### Fixed

- Training crashed on the real dataset. One keypoint frame (`MP_Data/hi/0/0.npy`) has no hand in it and was saved as
  `None`; such frames are now read as zeros.
- The app accepted a sign when the smallest of the last ten predictions matched, not when all ten agreed, so a
  flickering prediction could still be shown.
- `collectdata.py` quit on `q`, so the letter Q could not be saved. `Esc` quits now.
- `collectdata.py` created lower-case letter folders next to the upper-case ones on Linux.
- The frame extractor kept single quotes around a path dragged into the terminal.

### Added

- `evaluate.py`: accuracy, a per-sign report and a confusion matrix, on a held-out split or on every sequence. It
  replaces `acccuracy.py`, which could not run.
- The dataset on [Kaggle](https://www.kaggle.com/datasets/paragtonpe/sign-language-recognition-dataset): raw videos,
  video frames, training images, keypoints and training logs.
- `dataset_tools/video_frame.py`, the frame extractor used to build the word dataset, with command-line arguments.
- `requirements.txt` with tested versions, tests, continuous integration, and the security, contributing and
  conduct guides.

### Changed

- Scripts find their files relative to the project, so they work from any folder.
- The train/test split is fixed and stratified, so `trainmodel.py` and `evaluate.py` use the same held-out sequences.
- Keypoint files are loaded with pickling off. `data.py` saves zeros instead of `None` for a frame without a hand.
- Explicit imports instead of `from function import *`.

## [1.0.0] - 2024-05-07

The version submitted as the project.

- Real-time recognition of eight Indian Sign Language words (hi, my, name, is, what, your, glad, to) from a webcam,
  with MediaPipe hand landmarks and an LSTM.
- A new dataset recorded for the project, and a model trained on it.
- Built on [SignLanguageDetectionUsingML](https://github.com/kumarvivek9088/SignLanguageDetectionUsingML), which
  recognised the letters A, B and C.
