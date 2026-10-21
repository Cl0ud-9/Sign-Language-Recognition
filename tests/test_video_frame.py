import cv2
import numpy as np
import pytest

from video_frame import clean_path, extract_frames


@pytest.mark.parametrize('typed, expected', [
    ("& 'C:\\videos\\hi.mp4'", 'C:\\videos\\hi.mp4'),
    ('"C:\\videos\\hi.mp4"', 'C:\\videos\\hi.mp4'),
    ("'clip.mp4'", 'clip.mp4'),
    ('  clip.mp4 ', 'clip.mp4'),
])
def test_clean_path_removes_what_a_drag_and_drop_adds(typed, expected):
    assert clean_path(typed) == expected


def write_video(path, frames=40):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10, (64, 48))
    for i in range(frames):
        writer.write(np.full((48, 64, 3), i * 5, np.uint8))
    writer.release()


def test_extracts_the_requested_number_of_frames_in_order(tmp_path):
    video = tmp_path / 'clip.avi'
    write_video(video)
    extract_frames(video, tmp_path / 'out', num_frames=8)
    names = sorted(p.name for p in (tmp_path / 'out').iterdir())
    assert names == ['frame_{:03d}.png'.format(i) for i in range(8)]


def test_never_asks_for_more_frames_than_the_video_has(tmp_path):
    video = tmp_path / 'clip.avi'
    write_video(video, frames=5)
    extract_frames(video, tmp_path / 'out', num_frames=60)
    assert len(list((tmp_path / 'out').iterdir())) == 5


def test_missing_video_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_frames(tmp_path / 'missing.mp4', tmp_path / 'out')
