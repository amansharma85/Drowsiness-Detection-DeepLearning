import cv2
import numpy as np
import time
import threading
from tensorflow.keras.models import load_model
from playsound import playsound

# ---------------- LOAD MODEL ----------------
model = load_model("model/drowsiness_model.h5")

# ---------------- CASCADES ----------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# ---------------- STATES ----------------
eye_closed_start = None
eye_open_start = None
closed_duration = 0
open_duration = 0

alarm_active = False
alarm_thread_running = False

CLOSED_THRESHOLD = 10   # seconds
OPEN_RESET_TIME = 3     # seconds


# ---------------- ALARM FUNCTION ----------------
def alarm_loop():
    global alarm_active, alarm_thread_running
    alarm_thread_running = True
    while alarm_active:
        playsound("siren.mp3")
    alarm_thread_running = False


# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # -------- FACE DETECTION --------
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # ❌ NO PERSON → RESET EVERYTHING
    if len(faces) == 0:
        eye_closed_start = None
        eye_open_start = None
        closed_duration = 0
        open_duration = 0
        alarm_active = False

        cv2.putText(frame, "NO PERSON DETECTED",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 0),
                    2)

        cv2.imshow("Drowsiness Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # -------- FACE PRESENT --------
    (fx, fy, fw, fh) = faces[0]
    face_roi_gray = gray[fy:fy+fh, fx:fx+fw]

    eyes = eye_cascade.detectMultiScale(face_roi_gray, 1.3, 5)

    eye_closed = False

    # -------- EYE LOGIC --------
    if len(eyes) == 0:
        eye_closed = True
    else:
        for (ex, ey, ew, eh) in eyes:
            eye = face_roi_gray[ey:ey+eh, ex:ex+ew]
            eye = cv2.resize(eye, (24, 24))
            eye = eye / 255.0
            eye = eye.reshape(1, 24, 24, 1)

            pred = model.predict(eye, verbose=0)
            if pred < 0.5:
                eye_closed = True
            else:
                eye_closed = False
            break  # only one eye is enough

    now = time.time()

    # -------- CLOSED TIMER --------
    if eye_closed:
        eye_open_start = None
        if eye_closed_start is None:
            eye_closed_start = now
        closed_duration = int(now - eye_closed_start)

        if closed_duration >= CLOSED_THRESHOLD:
            alarm_active = True
            if not alarm_thread_running:
                threading.Thread(target=alarm_loop, daemon=True).start()
    else:
        eye_closed_start = None
        closed_duration = 0

        if alarm_active:
            if eye_open_start is None:
                eye_open_start = now
            open_duration = int(now - eye_open_start)

            if open_duration >= OPEN_RESET_TIME:
                alarm_active = False
                eye_open_start = None
                open_duration = 0

    # ---------------- UI (ALWAYS VISIBLE) ----------------
    cv2.putText(frame,
                f"EYES CLOSED TIME: {closed_duration}s",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255) if eye_closed else (0, 255, 0),
                2)

    if alarm_active:
        cv2.putText(frame,
                    "DROWSINESS ALERT!",
                    (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    3)

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
