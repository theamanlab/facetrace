"""
FaceTrace — Day 7-8: Streamlit UI
-------------------------------------------------------------------
Ties the whole pipeline into one browser app:
  - Sidebar: view enrolled people, add a new person (upload photos),
    remove a person
  - Main area: upload CCTV footage, run analysis with a live progress
    bar, view the appearance report, download it as CSV/Excel

Run:
    pip install streamlit opencv-python numpy pandas openpyxl
    streamlit run app.py

Requires face_utils.py, video_utils.py, and pipeline.py in the same folder.
"""

import os
import io
import tempfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from pipeline import (
    load_known_faces, enroll_person_from_images, remove_person,
    analyze_video, build_summary,
)
from face_utils import load_detector, load_recognizer

st.set_page_config(page_title="FaceTrace", page_icon="🎥", layout="wide")


# ---------- cached model loading (avoids re-downloading/reloading every rerun) ----------

@st.cache_resource(show_spinner="Loading recognition model...")
def get_recognizer():
    return load_recognizer()


@st.cache_resource(show_spinner="Loading detection model...")
def get_enroll_detector():
    # used only for enrollment (single images) — a fixed input size is fine here,
    # since setInputSize() is called per-image anyway inside enroll_person_from_images
    return load_detector(320, 320)


def read_uploaded_image(uploaded_file) -> np.ndarray:
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


# ---------- sidebar: enrollment management ----------

with st.sidebar:
    st.header("👥 Enrolled People")

    known_faces = load_known_faces()

    if known_faces:
        for name in sorted(known_faces.keys()):
            col1, col2 = st.columns([3, 1])
            col1.write(f"• {name}")
            if col2.button("✕", key=f"remove_{name}", help=f"Remove {name}"):
                remove_person(name)
                st.rerun()
    else:
        st.caption("No one enrolled yet. Add someone below.")

    st.divider()
    st.subheader("Add a Person")

    with st.form("enroll_form", clear_on_submit=True):
        new_name = st.text_input("Name")
        new_photos = st.file_uploader(
            "Upload 2-4 clear photos", type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
        )
        submitted = st.form_submit_button("Enroll")

        if submitted:
            if not new_name.strip():
                st.error("Please enter a name.")
            elif not new_photos:
                st.error("Please upload at least one photo.")
            else:
                images = [read_uploaded_image(f) for f in new_photos]
                detector = get_enroll_detector()
                recognizer = get_recognizer()
                success, message, _ = enroll_person_from_images(
                    new_name.strip(), images, detector, recognizer
                )
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)


# ---------- main area ----------

st.title("🎥 FaceTrace")
st.caption("CCTV Footage–Based Person Identification Using Facial Recognition")

known_faces = load_known_faces()

if not known_faces:
    st.warning("No one is enrolled yet. Add at least one person from the sidebar before analyzing footage.")

st.subheader("1. Upload Footage")
video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

st.subheader("2. Run Analysis")
analyze_clicked = st.button("▶ Analyze Footage", type="primary", disabled=(video_file is None or not known_faces))

if "summary_df" not in st.session_state:
    st.session_state.summary_df = None
if "detections_df" not in st.session_state:
    st.session_state.detections_df = None

if analyze_clicked and video_file is not None:
    # save the uploaded video to a temp file so OpenCV can read it by path
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_file.name)[1]) as tmp:
        tmp.write(video_file.read())
        tmp_path = tmp.name

    recognizer = get_recognizer()

    progress_bar = st.progress(0.0, text="Analyzing footage...")

    def update_progress(frac):
        progress_bar.progress(frac, text=f"Analyzing footage... {int(frac * 100)}%")

    detections_df = analyze_video(
        tmp_path, known_faces, recognizer,
        sample_interval_sec=1.0, progress_callback=update_progress,
    )

    progress_bar.empty()
    os.unlink(tmp_path)

    st.session_state.detections_df = detections_df
    st.session_state.summary_df = build_summary(detections_df)

    if detections_df.empty:
        st.info("No enrolled individuals were recognized in this footage.")
    else:
        st.success(f"Analysis complete — {len(detections_df)} recognized detections found.")

st.subheader("3. Results")

if st.session_state.summary_df is not None and not st.session_state.summary_df.empty:
    st.dataframe(st.session_state.summary_df, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)

    csv_bytes = st.session_state.summary_df.to_csv(index=False).encode("utf-8")
    col1.download_button(
        "⬇ Download CSV", data=csv_bytes,
        file_name="appearance_report.csv", mime="text/csv",
    )

    excel_buffer = io.BytesIO()
    st.session_state.summary_df.to_excel(excel_buffer, index=False)
    col2.download_button(
        "⬇ Download Excel", data=excel_buffer.getvalue(),
        file_name="appearance_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    with st.expander("View raw detection log"):
        st.dataframe(st.session_state.detections_df, use_container_width=True, hide_index=True)
else:
    st.caption("Run an analysis above to see results here.")
