"""
FaceTrace — Video loading & frame sampling utilities (Day 1-3)
-------------------------------------------------------------------
Core functions for reading a video file, getting basic info about it
(fps, duration, total frames), and sampling frames at a fixed time
interval instead of processing every single frame.

Used by every later stage (detection, recognition, reporting) — this
is the shared "video input" layer of the pipeline.
"""

import cv2
import os


def open_video(path: str) -> cv2.VideoCapture:
    """Open a video file and fail with a clear message if it can't be read."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Video file not found: {path}")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video file: {path}\n"
            f"Check that it's a valid video (.mp4, .avi, .mov) and not corrupted."
        )
    return cap


def get_video_info(cap: cv2.VideoCapture) -> dict:
    """Return basic metadata about an opened video: fps, frame count, duration, resolution."""
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    duration_sec = total_frames / fps if fps > 0 else 0

    return {
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "duration_sec": duration_sec,
    }


def format_timestamp(seconds: float) -> str:
    """Convert seconds -> HH:MM:SS string for human-readable reports."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def sample_frames(cap: cv2.VideoCapture, interval_sec: float = 1.0):
    """
    Generator that yields (frame, timestamp_seconds) pairs, sampled at a
    fixed time interval rather than every frame in the video.

    Example: interval_sec=1.0 on a 30fps video grabs 1 frame per second
    (every 30th frame) instead of all 30 — much faster to process, and
    plenty of resolution for tracking who appears when.
    """
    info = get_video_info(cap)
    fps = info["fps"]

    if fps <= 0:
        raise RuntimeError("Could not determine video FPS — file may be corrupted.")

    frame_interval = max(1, round(fps * interval_sec))
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # end of video

        if frame_idx % frame_interval == 0:
            timestamp_sec = frame_idx / fps
            yield frame, timestamp_sec

        frame_idx += 1
