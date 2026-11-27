"""
FaceTrace — Day 1-2: Video loading & frame sampling test
-------------------------------------------------------------------
Verifies video_utils.py works correctly: opens a video file, prints
its basic info, then samples a few frames and prints their timestamps.

This is the same kind of "does the pipeline stage work" test script
as day1_2_webcam_setup.py was for the attendance project.

Run:
    python day1_2_video_test.py path/to/your/video.mp4

If no path is given, it looks for the first video file inside
test_footage/ (the folder record_test_video.py saves to).
"""

import sys
import os
import glob
from video_utils import open_video, get_video_info, sample_frames, format_timestamp


def find_default_video():
    candidates = glob.glob(os.path.join("test_footage", "*.mp4"))
    if not candidates:
        return None
    return sorted(candidates)[-1]  # most recent


def main():
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = find_default_video()
        if video_path is None:
            print(
                "No video path given, and no test footage found in test_footage/.\n"
                "Either run: python day1_2_video_test.py path/to/video.mp4\n"
                "Or record a test clip first with record_test_video.py"
            )
            return
        print(f"No path given — using most recent test clip: {video_path}\n")

    cap = open_video(video_path)
    info = get_video_info(cap)

    print("Video Info")
    print("-" * 40)
    print(f"Resolution:      {info['width']}x{info['height']}")
    print(f"FPS:             {info['fps']:.2f}")
    print(f"Total frames:    {info['total_frames']}")
    print(f"Duration:        {format_timestamp(info['duration_sec'])}")
    print()

    print("Sampling frames every 1.0 second...")
    print("-" * 40)

    cap2 = open_video(video_path)  # fresh capture for sampling from the start
    count = 0
    for frame, timestamp in sample_frames(cap2, interval_sec=1.0):
        print(f"Frame at {format_timestamp(timestamp)}  (shape: {frame.shape})")
        count += 1
        if count >= 10:
            print("... (showing first 10 sampled frames only)")
            break

    cap.release()
    cap2.release()


if __name__ == "__main__":
    main()
