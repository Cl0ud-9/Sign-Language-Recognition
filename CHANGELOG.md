# Changelog

What changed in each version of Sign Language Recognition.

The format follows [Keep a Changelog](https://keepachangelog.com), and the versions follow
[Semantic Versioning](https://semver.org).

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
