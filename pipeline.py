"""
FaceTrace — Shared pipeline core (used by both the terminal scripts
and the Streamlit UI, so the analysis logic lives in exactly one place)
-------------------------------------------------------------------
"""

import os
import pickle
import numpy as np
import pandas as pd

from video_utils import open_video, get_video_info, sample_frames, format_timestamp
from face_utils import load_detector, load_recognizer, get_embedding, compare_embeddings

EMBEDDINGS_FILE = "embeddings.pkl"
SESSION_GAP_SEC = 4.0  # gap larger than this = a new, separate appearance


# ---------- enrollment / known-faces ----------

def load_known_faces() -> dict:
    if not os.path.exists(EMBEDDINGS_FILE):
        return {}
    with open(EMBEDDINGS_FILE, "rb") as f:
        return pickle.load(f)


def save_known_faces(known_faces: dict):
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(known_faces, f)


def enroll_person_from_images(name: str, images: list, detector, recognizer) -> tuple:
    """
    images: list of OpenCV BGR images (already loaded, e.g. from uploaded files).
    Returns (success: bool, message: str, num_encoded: int).
    Adds/overwrites `name` in embeddings.pkl using the averaged embedding.
    """
    embeddings = []
    for img in images:
        h, w = img.shape[:2]
        detector.setInputSize((w, h))
        _, faces = detector.detect(img)
        if faces is None or len(faces) == 0:
            continue
        if len(faces) > 1:
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        embedding = get_embedding(img, faces[0], recognizer)
        embeddings.append(embedding)

    if not embeddings:
        return False, "No face detected in any of the uploaded photos.", 0

    known_faces = load_known_faces()
    known_faces[name] = np.mean(embeddings, axis=0)
    save_known_faces(known_faces)
    return True, f"Enrolled '{name}' using {len(embeddings)} photo(s).", len(embeddings)


def remove_person(name: str) -> bool:
    known_faces = load_known_faces()
    if name in known_faces:
        del known_faces[name]
        save_known_faces(known_faces)
        return True
    return False


# ---------- video analysis ----------

def identify_face(embedding, known_faces: dict, recognizer):
    best_name, best_score = "Unknown", -1.0
    for name, known_embedding in known_faces.items():
        is_match, score = compare_embeddings(recognizer, embedding, known_embedding)
        if is_match and score > best_score:
            best_name, best_score = name, score
    return best_name, best_score


def analyze_video(video_path: str, known_faces: dict, recognizer,
                   sample_interval_sec: float = 1.0, progress_callback=None) -> pd.DataFrame:
    """
    Runs detection + recognition on sampled frames of a video.
    progress_callback(fraction: float) is called periodically if provided.
    Returns a DataFrame with columns: name, timestamp_sec, timestamp, confidence
    (one row per recognized detection — the RAW log, not yet summarized).
    """
    cap = open_video(video_path)
    info = get_video_info(cap)
    duration = info["duration_sec"] or 1.0

    detector = load_detector(info["width"], info["height"])

    detections = []
    for frame, timestamp in sample_frames(cap, interval_sec=sample_interval_sec):
        _, faces = detector.detect(frame)

        if faces is not None:
            for face in faces:
                embedding = get_embedding(frame, face, recognizer)
                name, score = identify_face(embedding, known_faces, recognizer)
                if name != "Unknown":
                    detections.append({
                        "name": name,
                        "timestamp_sec": round(timestamp, 2),
                        "timestamp": format_timestamp(timestamp),
                        "confidence": round(float(score), 3),
                    })

        if progress_callback:
            progress_callback(min(timestamp / duration, 1.0))

    cap.release()

    if progress_callback:
        progress_callback(1.0)

    return pd.DataFrame(detections, columns=["name", "timestamp_sec", "timestamp", "confidence"])


# ---------- report generation ----------

def _group_into_appearances(person_detections: pd.DataFrame) -> list:
    rows = person_detections.sort_values("timestamp_sec").to_dict("records")
    appearances = []
    current = None

    for row in rows:
        if current is None:
            current = {"start_sec": row["timestamp_sec"], "end_sec": row["timestamp_sec"],
                       "confidences": [row["confidence"]]}
        elif row["timestamp_sec"] - current["end_sec"] <= SESSION_GAP_SEC:
            current["end_sec"] = row["timestamp_sec"]
            current["confidences"].append(row["confidence"])
        else:
            appearances.append(current)
            current = {"start_sec": row["timestamp_sec"], "end_sec": row["timestamp_sec"],
                       "confidences": [row["confidence"]]}

    if current is not None:
        appearances.append(current)

    return appearances


def build_summary(detections_df: pd.DataFrame) -> pd.DataFrame:
    """Takes the raw detections DataFrame and returns the per-person summary report."""
    if detections_df.empty:
        return pd.DataFrame(columns=[
            "Name", "First Seen", "Last Seen", "Number of Appearances",
            "Total Detections", "Est. Time on Camera (sec)", "Avg Confidence",
        ])

    summary_rows = []
    for name, group in detections_df.groupby("name"):
        appearances = _group_into_appearances(group)
        all_confidences = [c for a in appearances for c in a["confidences"]]
        total_duration = sum(a["end_sec"] - a["start_sec"] for a in appearances)

        summary_rows.append({
            "Name": name,
            "First Seen": format_timestamp(appearances[0]["start_sec"]),
            "Last Seen": format_timestamp(appearances[-1]["end_sec"]),
            "Number of Appearances": len(appearances),
            "Total Detections": len(group),
            "Est. Time on Camera (sec)": round(total_duration, 1),
            "Avg Confidence": round(sum(all_confidences) / len(all_confidences), 3),
        })

    return pd.DataFrame(summary_rows).sort_values("First Seen").reset_index(drop=True)
