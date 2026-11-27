# FaceTrace

FaceTrace is a Streamlit app for finding enrolled people in a video and exporting an appearance report. It uses OpenCV's YuNet detector and SFace recognizer.

## Run locally

From the project folder, run the following commands in a Python environment.

```sh
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Enroll a person using their reference photos, then upload a video to analyze. The app downloads both ONNX models on first use. If you already have face_detection_yunet.onnx and face_recognition_sface.onnx, place them beside app.py to use them offline.

Enrollment data, footage, and generated reports stay out of Git.
