import argparse
import logging
import sys
from pathlib import Path

import cv2

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def extract_frames(
    video_path: Path,
    output_folder: Path,
    num_frames: int = 60
) -> None:
    """
    Extracts an evenly distributed sequence of frames from a video file.

    Args:
        video_path (Path): Path to the input video file.
        output_folder (Path): Directory where extracted frames will be saved.
        num_frames (int): Number of frames to extract. Defaults to 60.

    Raises:
        FileNotFoundError: If the input video file is not found.
        ValueError: If the video cannot be opened.
    """
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    logging.info(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Failed to open video file at: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        logging.warning("Video appears to have 0 frames. It may be corrupted.")
        return

    frame_interval = max(1, total_frames // num_frames)

    output_folder.mkdir(parents=True, exist_ok=True)
    logging.info(f"Saving {min(num_frames, total_frames)} frames to: {output_folder}")

    for i in range(num_frames):
        frame_position = i * frame_interval

        if frame_position >= total_frames:
             break

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
        ret, frame = cap.read()

        if not ret:
            logging.error(f"Error reading frame at position {frame_position}")
            continue

        # Zero-pad filenames so they sort mathematically correct (e.g., frame_001.png)
        frame_filename = output_folder / f"frame_{str(i).zfill(3)}.png"

        success = cv2.imwrite(str(frame_filename), frame)
        if not success:
             logging.error(f"Failed to save: {frame_filename.name}")

    cap.release()
    logging.info("Extraction complete.")


def clean_path(text: str) -> str:
    """Undo what a drag-and-drop into PowerShell adds: a leading '& ' and surrounding quotes."""
    text = text.strip()
    if text.startswith("& "):
        text = text[2:]
    return text.strip(" \"'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract evenly spaced frames from a video.")
    parser.add_argument("video", nargs="?", help="video file (asked for when omitted)")
    parser.add_argument("-o", "--output", help="output folder (default: output_frames)")
    parser.add_argument("-n", "--frames", type=int, help="number of frames to extract (default: 60)")
    args = parser.parse_args()

    if args.video:
        video_input = clean_path(args.video)
        output_input = clean_path(args.output or "output_frames")
        num_frames = args.frames or 60
    else:
        print("=== Video Frame Extractor ===")
        video_input = clean_path(input("Enter the path to the video file: "))
        output_input = clean_path(input("Enter the output folder (or press Enter for 'output_frames'): "))
        output_input = output_input or "output_frames"
        frames_input = input("Enter number of frames to extract (or press Enter for 60): ").strip()
        num_frames = int(frames_input) if frames_input.isdigit() else 60

    input_path = Path(video_input)

    # Auto-fix mismatch if user executes code while already inside the copied parent directory
    if not input_path.is_file():
        current_dir_name = Path.cwd().name
        parts = list(input_path.parts)
        if len(parts) > 1 and parts[0] == current_dir_name:
            corrected_path = Path(*parts[1:])
            if corrected_path.is_file():
                print(f"[*] Auto-corrected path mismatch: removed duplicate '{current_dir_name}' prefix.")
                input_path = corrected_path

    output_dir = Path(output_input)

    try:
        extract_frames(input_path, output_dir, num_frames=num_frames)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        sys.exit(1)
