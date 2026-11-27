"""
FaceTrace — Day 3-4: Run detection + recognition on video footage
-------------------------------------------------------------------
Loads a video file, samples frames at a fixed interval (via
video_utils.py), runs face detection + recognition on each sampled
frame (via face_utils.py — the same YuNet + SFace pipeline from the
attendance project), and logs every recognized appearance with a
timestamp.

Requires embeddings.pkl to already exist — run enroll_faces.py first
(same enrollment script from the attendance project; copy it into
this folder unchanged).

Run:
    python analyze_video.py path\to\video.mp4

If no path is given, it uses the most recent clip in test_footage/.

Output:
    Prints each detection live, and saves the full raw log to
    detections_log.csv — this is the input Day 6's reporting script
    will aggregate into a per-person summary.
"""

import sys
import os
import glob
import pickle
import csv

from video_utils import open_video, get_video_info, sample_frames, format_timestamp
from face_utils import load_detector, load_recognizer, get_embedding, compare_embeddings

EMBEDDINGS_FILE = "embeddings.pkl"
LOG_FILE = "detections_log.csv"
SAMPLE_INTERVAL_SEC = 1.0  # analyze 1 frame per second


def find_default_video():
    candidates = glob.glob(os.path.join("test_footage", "*.mp4"))
    if not candidates:
        return None
    return sorted(candidates)[-1]


def load_known_faces():
    if not os.path.exists(EMBEDDINGS_FILE):
        raise RuntimeError(
            f"'{EMBEDDINGS_FILE}' not found. Run enroll_faces.py first "
            f"to build it from your dataset folder."
        )
    with open(EMBEDDINGS_FILE, "rb") as f:
        return pickle.load(f)


def identify_face(embedding, known_faces, recognizer):
    best_name, best_score = "Unknown", -1.0
    for name, known_embedding in known_faces.items():
        is_match, score = compare_embeddings(recognizer, embedding, known_embedding)
        if is_match and score > best_score:
            best_name, best_score = name, score
    return best_name, best_score


def main():
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = find_default_video()
        if video_path is None:
            print(
                "No video path given, and no test footage found in test_footage/.\n"
                "Either run: python analyze_video.py path\\to\\video.mp4\n"
                "Or record a test clip first with record_test_video.py"
            )
            return
        print(f"No path given — using most recent test clip: {video_path}\n")

    known_faces = load_known_faces()
    print(f"Loaded {len(known_faces)} known face(s): {', '.join(known_faces.keys())}\n")

    cap = open_video(video_path)
    info = get_video_info(cap)
    print(f"Video: {info['width']}x{info['height']}, {info['fps']:.1f} fps, "
          f"duration {format_timestamp(info['duration_sec'])}\n")

    print("Loading detection & recognition models...")
    detector = load_detector(info["width"], info["height"])
    recognizer = load_recognizer()

    print(f"Analyzing (1 frame every {SAMPLE_INTERVAL_SEC}s)...\n")
    print("-" * 50)

    detections = []  # list of dicts: name, timestamp_sec, confidence

    frame_count = 0
    for frame, timestamp in sample_frames(cap, interval_sec=SAMPLE_INTERVAL_SEC):
        frame_count += 1
        _, faces = detector.detect(frame)

        if faces is None:
            continue

        for face in faces:
            embedding = get_embedding(frame, face, recognizer)
            name, score = identify_face(embedding, known_faces, recognizer)

            if name != "Unknown":
                ts_str = format_timestamp(timestamp)
                print(f"  {ts_str}  ->  {name}  (confidence {score:.2f})")
                detections.append({
                    "name": name,
                    "timestamp_sec": round(timestamp, 2),
                    "timestamp": ts_str,
                    "confidence": round(float(score), 3),
                })

    cap.release()

    print("-" * 50)
    print(f"\nProcessed {frame_count} sampled frames.")
    print(f"Total recognized appearances logged: {len(detections)}")

    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "timestamp_sec", "timestamp", "confidence"])
        writer.writeheader()
        writer.writerows(detections)

    print(f"Raw detection log saved -> {LOG_FILE}")


if __name__ == "__main__":
    main()
