"""
FaceTrace — Record a staged test video (helper, not part of the pipeline)
-------------------------------------------------------------------
Since using real CCTV footage without institutional clearance isn't
appropriate, use this to record your own short staged test clip with
your webcam — you and a few teammates walking in/out of frame, sitting,
talking, etc. This gives you realistic footage to test the pipeline on.

Run:
    python record_test_video.py

Controls:
    r  -> start/stop recording
    q  -> quit
"""

import cv2
import os
from datetime import datetime

OUTPUT_DIR = "test_footage"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = 20.0

    recording = False
    writer = None

    print("Press 'r' to start/stop recording, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        display = frame.copy()

        if recording:
            cv2.circle(display, (30, 30), 10, (0, 0, 255), -1)
            cv2.putText(display, "REC", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            writer.write(frame)

        cv2.imshow("FaceTrace - Record Test Footage", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            if not recording:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = os.path.join(OUTPUT_DIR, f"test_clip_{timestamp}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
                recording = True
                print(f"Recording started -> {path}")
            else:
                recording = False
                writer.release()
                writer = None
                print("Recording stopped.")

        elif key == ord("q"):
            break

    if writer is not None:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
