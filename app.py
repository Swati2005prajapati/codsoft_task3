"""
app.py - Flask app with detection / capture / train / recognize / analyze (DeepFace)
Place at: face_app/app.py
"""

import os
import json
import time
import cv2
import numpy as np
from flask import Flask, render_template, Response, request, redirect, url_for, send_from_directory, flash

# Try to import DeepFace (we will use it for analyze)
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except Exception:
    DEEPFACE_AVAILABLE = False

app = Flask(__name__)
app.secret_key = "very-secret-key-swati"

# ---------- PATHS ----------
KNOWN_FACES_DIR = "data/known_faces"
CASCADE_PATH = "models/haarcascade_frontalface_default.xml"
TRAINER_DIR = "trainer"
TRAINER_MODEL = os.path.join(TRAINER_DIR, "trainer.yml")
LABELS_JSON = os.path.join(TRAINER_DIR, "labels.json")
# (we won't require age/gender caffe models because we're using DeepFace here)
EMOTION_MODEL_FILE = None  # not used when using DeepFace

os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
os.makedirs(TRAINER_DIR, exist_ok=True)

# ---------- LOAD DETECTOR & RECOGNIZER ----------
if not os.path.exists(CASCADE_PATH):
    raise FileNotFoundError(f"Haarcascade not found at {CASCADE_PATH}. Please download and place it there.")

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# LBPH recognizer (opencv-contrib)
try:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
except Exception as e:
    raise RuntimeError("cv2.face.LBPHFaceRecognizer_create() not available. Install opencv-contrib-python.") from e

if os.path.exists(TRAINER_MODEL):
    try:
        recognizer.read(TRAINER_MODEL)
    except Exception:
        pass

# ---------- Utility ----------
def load_labels_dict():
    if os.path.exists(LABELS_JSON):
        try:
            with open(LABELS_JSON, "r") as f:
                labels = json.load(f)
            return {int(v): k for k, v in labels.items()}
        except Exception:
            return {}
    return {}

def safe_deepface_analyze(img, actions=['age','emotion'], enforce_detection=False, detector_backend='opencv'):
    """
    Run DeepFace.analyze on a cropped face (BGR image). Returns dict or None.
    Wrap in try/except to avoid crashes.
    """
    try:
        # DeepFace accepts numpy arrays (BGR) in latest versions
        # Some DeepFace versions return a dict, some return list; handle both.
        res = DeepFace.analyze(img, actions=actions, enforce_detection=enforce_detection, detector_backend=detector_backend)
        # If result is list, take first
        if isinstance(res, list) and len(res) > 0:
            return res[0]
        return res
    except Exception as e:
        # don't spam terminal, but you can print once if needed
        # print("DeepFace analyze error:", e)
        return None

# ---------- FLASK ROUTES (UI) ----------
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/detection")
def detection():
    return render_template("detection.html")

@app.route("/recognition")
def recognition():
    people = sorted([d for d in os.listdir(KNOWN_FACES_DIR) if os.path.isdir(os.path.join(KNOWN_FACES_DIR, d))])
    return render_template("recognition.html", people=people)

@app.route("/analyze")
def analyze():
    # show page which itself uses /video_feed_analyze to display stream
    return render_template("analyze.html", deepface_available=DEEPFACE_AVAILABLE)



@app.route("/capture_result/<person>")
def capture_result(person):
    candidates = [d for d in os.listdir(KNOWN_FACES_DIR) if d.startswith(person)]
    if not candidates:
        flash("No images found for that person.", "error")
        return redirect(url_for("recognition"))
    folder = sorted(candidates)[-1]
    folder_path = os.path.join(KNOWN_FACES_DIR, folder)
    images = [f"/data/known_faces/{folder}/{f}" for f in sorted(os.listdir(folder_path)) if f.lower().endswith((".jpg", ".png"))]
    return render_template("capture_result.html", name=folder, images=images)

@app.route("/data/known_faces/<person>/<filename>")
def serve_image(person, filename):
    return send_from_directory(os.path.join(KNOWN_FACES_DIR, person), filename)

# ---------- CAPTURE / TRAIN ----------
@app.route("/capture", methods=["POST"])
def capture():
    name = (request.form.get("name") or "").strip()
    if not name:
        flash("Enter a name before capturing.", "error")
        return redirect(url_for("recognition"))
    safe_name = name.replace(" ", "_")
    base_dir = os.path.join(KNOWN_FACES_DIR, safe_name)
    if os.path.exists(base_dir) and len(os.listdir(base_dir)) >= 10:
        base_dir = f"{base_dir}_{int(time.time())}"
    os.makedirs(base_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0
    flash("Capturing started - close small window or press 'q' to stop early.", "info")
    while count < 50:
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))
        for (x,y,w,h) in faces:
            face = gray[y:y+h, x:x+w]
            cv2.imwrite(os.path.join(base_dir, f"{count}.jpg"), face)
            count += 1
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.imshow("Capturing faces - press q to stop", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    flash(f"Captured {count} images for {safe_name}", "success")
    return redirect(url_for("capture_result", person=safe_name))

@app.route("/train")
def train():
    faces = []
    ids = []
    label_ids = {}
    current_id = 0
    for root, dirs, files in os.walk(KNOWN_FACES_DIR):
        for file in files:
            if not file.lower().endswith((".jpg", ".png")):
                continue
            path = os.path.join(root, file)
            label = os.path.basename(root)
            if label not in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            faces.append(img)
            ids.append(id_)
    if len(faces) == 0:
        flash("No images found to train.", "error")
        return redirect(url_for("recognition"))

    recognizer.train(faces, np.array(ids))
    recognizer.save(TRAINER_MODEL)
    with open(LABELS_JSON, "w") as f:
        json.dump(label_ids, f)

    flash(f"Training complete: {len(ids)} images, {len(label_ids)} labels.", "success")
    return redirect(url_for("recognition"))

# ---------- STREAM GENERATORS ----------
def gen_detection_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
    cap.release()

@app.route("/video_feed_detection")
def video_feed_detection():
    return Response(gen_detection_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_recognize_frames():
    cap = cv2.VideoCapture(0)
    labels = load_labels_dict()
    threshold = 70.0
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))
        for (x,y,w,h) in faces:
            roi = gray[y:y+h, x:x+w]
            name = "Unknown"
            color = (0,0,255)
            try:
                label_id, conf = recognizer.predict(roi)
                if conf < threshold:
                    name = labels.get(label_id, "Unknown")
                    color = (0,255,0)
                    # blur the face area
                    face_area = frame[y:y+h, x:x+w]
                    blurred = cv2.GaussianBlur(face_area, (99,99), 30)
                    frame[y:y+h, x:x+w] = blurred
                else:
                    name = "Unknown"
            except Exception:
                name = "Unknown"
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
    cap.release()

@app.route("/video_feed_recognize")
def video_feed_recognize():
    return Response(gen_recognize_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_analyze_frames():
    """
    Analyze stream using DeepFace (age + emotion).
    Runs heavy inference (DeepFace.analyze) every N frames to keep performance acceptable.
    """
    cap = cv2.VideoCapture(0)
    frame_count = 0
    last_results = {}  # per-face index cache: {i: (age_text, emotion_text, ts)}
    N = 10  # run DeepFace every N frames (adjust if needed)

    if not DEEPFACE_AVAILABLE:
        print("DeepFace not available. Install deepface to enable analyze features.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))

        do_heavy = DEEPFACE_AVAILABLE and (frame_count % N == 0)

        for i, (x,y,w,h) in enumerate(faces):
            roi_color = frame[y:y+h, x:x+w].copy()
            age_text = ""
            emotion_text = ""

            if do_heavy:
                # Run DeepFace analyze on the cropped face
                res = safe_deepface_analyze(roi_color, actions=['age','emotion'], enforce_detection=False, detector_backend='opencv')
                if res is not None:
                    # DeepFace behavior: result may include 'age' and 'dominant_emotion'
                    # Some DeepFace returns {'age':..., 'dominant_emotion':..., ...}
                    if isinstance(res, dict):
                        # if res contains 'age' (int) and 'dominant_emotion' (str)
                        age_val = res.get('age', None)
                        emotion_val = res.get('dominant_emotion', None) or res.get('dominant_emotion', '')
                        if age_val is not None:
                            age_text = f"{int(age_val)} yrs"
                        if emotion_val:
                            emotion_text = str(emotion_val)
                    elif isinstance(res, list) and len(res) > 0:
                        # some versions return a list
                        r0 = res[0]
                        age_val = r0.get('age', None)
                        emotion_val = r0.get('dominant_emotion', None) or r0.get('dominant_emotion', '')
                        if age_val is not None:
                            age_text = f"{int(age_val)} yrs"
                        if emotion_val:
                            emotion_text = str(emotion_val)

                # save to cache
                last_results[i] = (age_text, emotion_text, time.time())

            # if not heavy frame, use cache if present
            if (not do_heavy) and (i in last_results):
                age_text, emotion_text, ts = last_results[i]

            # build label_text
            parts = []
            if emotion_text:
                parts.append(emotion_text)
            if age_text:
                parts.append(age_text)
            if not parts:
                if not DEEPFACE_AVAILABLE:
                    label_text = "DeepFace not installed"
                else:
                    label_text = ""  # inference pending
            else:
                label_text = ", ".join(parts)

            # Draw red rectangle and label (if any)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)
            if label_text:
                cv2.putText(frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

    cap.release()

@app.route("/video_feed_analyze")
def video_feed_analyze():
    return Response(gen_analyze_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ---------- Run ----------
if __name__ == "__main__":
    app.run(debug=True)
