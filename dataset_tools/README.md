# Video frame extractor

`video_frame.py` saves evenly spaced frames from a video as PNG images. It is how the word signs in this project's
dataset were turned from phone videos into images: the raw videos and the extracted frames are both in the
[Kaggle dataset](https://www.kaggle.com/datasets/paragtonpe/sign-language-recognition-dataset).

## Usage

It needs only OpenCV, which `requirements.txt` already installs.

```bash
python dataset_tools/video_frame.py path/to/video.mp4 -o output_frames -n 60
```

| Argument | Meaning | Default |
|---|---|---|
| `video` | The video file | asked for |
| `-o`, `--output` | Folder the frames are written to | `output_frames` |
| `-n`, `--frames` | How many frames to take, spread across the whole video | `60` |

Run it with no arguments to be asked for each value instead. You can then drag the video into the terminal: the
`& '...'` that PowerShell adds around a dropped path is removed for you, and so is a folder name that VS Code's
**Copy Relative Path** repeats when you run the script from inside that folder.

Frames are named `frame_000.png`, `frame_001.png` and so on, so they sort in order.

To use the frames for training, copy the ones that show the sign clearly into `Image/<sign>/`, numbered from `0.png`,
then run `data.py` (see the main [README](../README.md#retrain-the-model)).
