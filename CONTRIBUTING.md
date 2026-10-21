# Contributing to Sign Language Recognition

Thanks for wanting to help. Bug reports, ideas, new signs and pull requests are all welcome. Please keep to the
[code of conduct](CODE_OF_CONDUCT.md).

The project is licensed under the [GPL-3.0](LICENSE). By opening a pull request you confirm that you have the right to
submit the change and that it is licensed the same way.

## Before you start

- **Found a bug?** Search the [issues](https://github.com/Cl0ud-9/Sign-Language-Recognition/issues) first, then open
  one. Include the terminal output and, for a misread sign, a screenshot of the app.
- **Have an idea?** Open an issue before writing code, so we can agree on the approach. Large changes that were not
  discussed may be closed.
- **Found a security problem?** Do not open a public issue. Report it privately: see [SECURITY.md](SECURITY.md).

## Setup

Requirements: Python 3.10 or 3.11 (TensorFlow 2.15 does not support newer versions).

```bash
git clone https://github.com/<your-account>/Sign-Language-Recognition.git
cd Sign-Language-Recognition
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
ruff check .
pytest
```

The tests need no camera and no dataset. To retrain or evaluate on real data, download the
[dataset](https://www.kaggle.com/datasets/paragtonpe/sign-language-recognition-dataset) as the README describes.

## What a pull request needs

- `ruff check .` and `pytest` pass.
- A new behaviour comes with a test that fails without it.
- A retrained `model.h5` comes with the output of `python evaluate.py`, and the results in the README are updated.
- New signs: add them to `actions` in `function.py`, to the table in the README, and give the dataset change in the
  pull request (they can be added to Kaggle as a new version).
- No dataset files (`Image/`, `MP_Data/`, `Logs/`) or virtual environments are committed.

## Style

Match the surrounding code. Comments are short and explain why, not what. Plain wording in docs, with no marketing
language.
