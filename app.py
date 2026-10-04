import cv2
import os
import numpy as np
import threading
from flask import Flask, jsonify, render_template, request, redirect, url_for, send_file
import csv
from datetime import datetime
from openpyxl import Workbook, load_workbook
import smtplib
from email.mime.text import MIMEText
import time
from werkzeug.utils import secure_filename

# ================= CONFIG =================
DATASET_DIR = "dataset"
STUDENTS_FILE = "students.csv"
TRAINER_FILE = "trainer.yml"
ATTENDANCE_FILE = "attendance.xlsx"

EMAIL_SENDER = "balajigtbook@gmail.com"
EMAIL_PASSWORD = "jkmo yjbh nity dqpi"

app = Flask(__name__)
recognizer_lock = threading.Lock()
cached_recognizer = None
cached_trainer_mtime = None
capture_last_frame_at = {}
capture_lock = threading.Lock()

# ============ Haar Cascade ================
cascade_filename = "haarcascade_frontalface_default.xml"
cascade_paths = (
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", cascade_filename),
    os.path.join(cv2.data.haarcascades, cascade_filename),
)
face_cascade = None

for cascade_path in cascade_paths:
    if os.path.isfile(cascade_path):
        candidate = cv2.CascadeClassifier(cascade_path)
        if not candidate.empty():
            face_cascade = candidate
            break

if face_cascade is None:
    raise RuntimeError(
        "Unable to load the Haar cascade. Checked: " + ", ".join(cascade_paths)
    )

# ================= EMAIL ==================
def send_email(to_email, student_name, date_str, time_str):
    try:
        msg = MIMEText(
            f"Dear Parent,\n\nYour child {student_name} attended on {date_str} at {time_str}.\n\nRegards,\nAttendance System"
        )
        msg["Subject"] = "Attendance Notification"
        msg["From"] = EMAIL_SENDER
        msg["To"] = to_email

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, to_email, msg.as_string())

        print("✅ Email sent to", to_email)

    except Exception as e:
        print("❌ Email Error:", e)


# ================= STUDENTS =================
def load_students():
    students = {}

    if os.path.exists(STUDENTS_FILE):
        with open(STUDENTS_FILE, newline="") as file:
            reader = csv.DictReader(file)

            for row in reader:
                if not row.get("id") or not row.get("name"):
                    continue

                parent_email = row.get("parent_email") or row.get("parent_email1")

                students[int(row["id"])] = {
                    "name": row["name"],
                    "reg_no": row.get("reg_no", ""),
                    "parent_email": parent_email or ""
                }

    return students


def save_student(name, reg_no, parent_email):
    students = load_students()
    student_id = len(students) + 1
    os.makedirs(os.path.join(DATASET_DIR, str(student_id)), exist_ok=True)

    file_exists = os.path.exists(STUDENTS_FILE)
    write_header = True

    if file_exists:
        write_header = os.path.getsize(STUDENTS_FILE) == 0

    with open(STUDENTS_FILE, "a", newline="") as file:
        fieldnames = ["id", "reg_no", "name", "parent_email"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()

        writer.writerow({
            "id": student_id,
            "reg_no": reg_no,
            "name": name,
            "parent_email": parent_email
        })

    return student_id


def decode_browser_frame():
    uploaded_frame = request.files.get("frame")
    if uploaded_frame is None:
        return None

    encoded_frame = np.frombuffer(uploaded_frame.read(), dtype=np.uint8)
    if encoded_frame.size == 0:
        return None

    return cv2.imdecode(encoded_frame, cv2.IMREAD_COLOR)


def student_dataset_dir(student_id, student_name):
    student_dir = os.path.join(DATASET_DIR, str(student_id))
    if os.path.isdir(student_dir) and os.listdir(student_dir):
        return student_dir
    if os.path.basename(student_name) != student_name or student_name in (".", ".."):
        return os.path.join(os.path.abspath(DATASET_DIR), secure_filename(student_name) or str(student_id))
    legacy_dir = os.path.abspath(os.path.join(DATASET_DIR, student_name))
    dataset_root = os.path.abspath(DATASET_DIR)
    if os.path.commonpath((dataset_root, legacy_dir)) == dataset_root:
        return legacy_dir
    return os.path.join(dataset_root, secure_filename(student_name) or str(student_id))


def get_trained_recognizer():
    global cached_recognizer, cached_trainer_mtime

    trainer_mtime = os.path.getmtime(TRAINER_FILE)
    with recognizer_lock:
        if cached_recognizer is None or cached_trainer_mtime != trainer_mtime:
            if not hasattr(cv2, "face"):
                raise RuntimeError(
                    "OpenCV face recognition is unavailable. Install opencv-contrib-python-headless."
                )
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read(TRAINER_FILE)
            cached_recognizer = recognizer
            cached_trainer_mtime = trainer_mtime
        return cached_recognizer


# ================= TRAIN MODEL =================
def train_lbph():
    global cached_recognizer, cached_trainer_mtime

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    faces = []
    labels = []

    students = load_students()

    for student_id, info in students.items():
        person_dir = student_dataset_dir(student_id, info["name"])

        if not os.path.exists(person_dir):
            print("❌ Folder Missing:", person_dir)
            continue

        for img_file in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_file)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            faces.append(img)
            labels.append(student_id)

    if len(faces) == 0:
        print("❌ No training images found!")
        return False

    recognizer.train(faces, np.array(labels))
    recognizer.save(TRAINER_FILE)
    with recognizer_lock:
        cached_recognizer = None
        cached_trainer_mtime = None

    print("✅ Training Completed Successfully!")
    return True


# ================= ATTENDANCE =================
def mark_attendance(student_id):
    students = load_students()

    if student_id not in students:
        return

    name = students[student_id]["name"]
    email = students[student_id]["parent_email"]

    date_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H:%M:%S")

    if not os.path.exists(ATTENDANCE_FILE):
        wb = Workbook()
        ws = wb.active
        ws.append(["ID", "Name", "Date", "Time"])
    else:
        wb = load_workbook(ATTENDANCE_FILE)
        ws = wb.active

        for row in ws.iter_rows(min_row=2, values_only=True):
            row_date = row[2]
            if isinstance(row_date, datetime):
                row_date = row_date.strftime("%Y-%m-%d")
            try:
                same_student = int(row[0]) == int(student_id)
            except (TypeError, ValueError):
                same_student = False
            if same_student and str(row_date) == date_str:
                wb.close()
                return False

    ws.append([student_id, name, date_str, time_str])
    wb.save(ATTENDANCE_FILE)
    wb.close()

    send_email(email, name, date_str, time_str)
    return True


# ================= ROUTES =================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/notify_today", methods=["GET", "POST"])
def notify_today():
    if request.method == "POST":
        if not os.path.exists(ATTENDANCE_FILE):
            message = "❌ No attendance file found."
        else:
            wb = load_workbook(ATTENDANCE_FILE)
            ws = wb.active

            today_str = datetime.now().strftime("%Y-%m-%d")
            sent = 0
            students = load_students()

            for row in ws.iter_rows(min_row=2, values_only=True):  # Skip header automatically
                student_id, name, date, time = row

                if str(date) == today_str:
                    student_id = int(student_id)

                    if student_id in students:
                        email = students[student_id]["parent_email"]
                        send_email(email, name, date, time)
                        sent += 1

            message = f"✅ Notifications sent to {sent} students present today."

        return render_template("notify_today.html", message=message)
    return render_template("notify_today.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        student_data = request.get_json() if request.is_json else request.form
        name = str(student_data.get("name", "")).strip()
        reg_no = str(student_data.get("reg_no", "")).strip()
        parent_email = str(student_data.get("parent_email", "")).strip()
        if not name or not reg_no or not parent_email:
            return jsonify(error="Name, register number, and parent email are required."), 400

        student_id = save_student(name, reg_no, parent_email)
        return jsonify(student_id=student_id, name=name, target_images=30), 201

    return render_template("register.html")


@app.route("/capture_frame/<int:student_id>", methods=["POST"])
def capture_frame(student_id):
    students = load_students()
    if student_id not in students:
        return jsonify(error="Student registration was not found."), 404

    frame = decode_browser_frame()
    if frame is None:
        return jsonify(error="A valid camera frame is required."), 400

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    student_dir = os.path.join(DATASET_DIR, str(student_id))
    os.makedirs(student_dir, exist_ok=True)
    captured_files = [
        filename for filename in os.listdir(student_dir)
        if filename.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    captured_count = len(captured_files)

    if captured_count >= 30:
        return jsonify(captured=captured_count, complete=True)
    if len(faces) == 0:
        return jsonify(captured=captured_count, complete=False, message="No face detected. Look at the camera.")
    if len(faces) > 1:
        return jsonify(captured=captured_count, complete=False, message="Only one person should be in the frame.")

    now = time.monotonic()
    with capture_lock:
        last_capture = capture_last_frame_at.get(student_id, 0)
        if now - last_capture < 0.85:
            return jsonify(captured=captured_count, complete=False, message="Hold still for a moment.")

        x, y, width, height = faces[0]
        face_image = cv2.resize(gray[y:y + height, x:x + width], (200, 200))
        filename = os.path.join(student_dir, f"{captured_count + 1}.jpg")
        if not cv2.imwrite(filename, face_image):
            return jsonify(error="Could not save the captured face image."), 500
        capture_last_frame_at[student_id] = now
        captured_count += 1

    return jsonify(
        captured=captured_count,
        complete=captured_count >= 30,
        message="Face captured." if captured_count < 30 else "Registration photos captured.",
    )


@app.route("/train", methods=["GET", "POST"])
def train():
    if request.method == "POST":
        if train_lbph():
            message = "✅ Training Completed Successfully!"
        else:
            message = "❌ No Dataset Found. Please Register Students First."
        return render_template("train.html", message=message)
    return render_template("train.html")


@app.route("/recognize")
def recognize():
    return render_template("recognize.html")


@app.route("/start_recognition", methods=["POST"])
def start_recognition():
    if not os.path.exists(TRAINER_FILE):
        return jsonify(error="No trained model found. Train the model first."), 400
    try:
        get_trained_recognizer()
    except RuntimeError as error:
        return jsonify(error=str(error)), 503
    return jsonify(ready=True)


@app.route("/recognition_frame", methods=["POST"])
def recognition_frame():
    if not os.path.exists(TRAINER_FILE):
        return jsonify(error="No trained model found. Train the model first."), 400

    frame = decode_browser_frame()
    if frame is None:
        return jsonify(error="A valid camera frame is required."), 400

    try:
        recognizer = get_trained_recognizer()
    except RuntimeError as error:
        return jsonify(error=str(error)), 503

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    students = load_students()
    results = []

    for x, y, width, height in faces:
        face_image = cv2.resize(gray[y:y + height, x:x + width], (200, 200))
        with recognizer_lock:
            student_id, confidence = recognizer.predict(face_image)

        student = students.get(student_id)
        if confidence < 60 and student:
            attendance_added = mark_attendance(student_id)
            results.append({
                "name": student["name"],
                "box": [int(x), int(y), int(width), int(height)],
                "status": "Attendance marked" if attendance_added else "Already marked today",
            })
        else:
            results.append({
                "name": "Unknown",
                "box": [int(x), int(y), int(width), int(height)],
                "status": "Not recognized",
            })

    return jsonify(faces=results)

@app.route("/view_attendance")
def view_attendance():
    if not os.path.exists(ATTENDANCE_FILE):
        return "No attendance file found."

    wb = load_workbook(ATTENDANCE_FILE)
    ws = wb.active

    records = []
    students = load_students()

    first_row = True   # Header row flag

    for row in ws.iter_rows(values_only=True):
        if first_row:   # Skip header
            first_row = False
            continue

        student_id = int(row[0])
        name = row[1]
        date = row[2]
        time = row[3]

        reg_no = students.get(student_id, {}).get("reg_no", "N/A")
        records.append({
            "reg_no": reg_no,
            "name": name,
            "date": date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date),
            "time": time
        })

    wb.close()
    today = datetime.now().strftime("%Y-%m-%d")
    return render_template("view_attendance.html", attendance=records, today=today)

@app.route("/download_excel_attendance")
def download_excel_attendance():
    if not os.path.exists(ATTENDANCE_FILE):
        return "No attendance file found."
    return send_file(ATTENDANCE_FILE, as_attachment=True, download_name="attendance.xlsx", mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

@app.route("/students")
def students():
    if not os.path.exists(STUDENTS_FILE):
        return render_template("view_students.html", students=[])

    students_list = []

    with open(STUDENTS_FILE, newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            students_list.append(row)

    return render_template("view_students.html", students=students_list)


@app.route("/report")
def report():
    return render_template("report advaisor.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


# ================= RUN =================
if __name__ == "__main__":
    os.makedirs(DATASET_DIR, exist_ok=True)

    app.run(debug=True, port=5050, use_reloader=False)
