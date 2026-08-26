import cv2
import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_file
import csv
from datetime import datetime
from openpyxl import Workbook, load_workbook
import smtplib
from email.mime.text import MIMEText
import time

# ================= CONFIG =================
DATASET_DIR = "dataset"
STUDENTS_FILE = "students.csv"
TRAINER_FILE = "trainer.yml"
ATTENDANCE_FILE = "attendance.xlsx"

EMAIL_SENDER = "balajigtbook@gmail.com"
EMAIL_PASSWORD = "jkmo yjbh nity dqpi"

app = Flask(__name__)

# ============ Haar Cascade ================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
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
    os.makedirs(os.path.join(DATASET_DIR, name), exist_ok=True)

    students = load_students()
    student_id = len(students) + 1

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


# ================= CAPTURE =================

def capture_images(student_name):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera not accessible. Trying index 1...")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("❌ Camera still not accessible.")
            return False

    count = 0
    last_capture_time = 0  # ⏱️ track last capture

    cv2.namedWindow("Register - Press Q to Stop", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Register - Press Q to Stop", cv2.WND_PROP_TOPMOST, 1)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        current_time = time.time()

        for (x, y, w, h) in faces:
            # ⏱️ Capture only if 1 second passed
            if current_time - last_capture_time >= 1 and count < 30:
                count += 1
                last_capture_time = current_time

                face_img = gray[y:y+h, x:x+w]
                img_path = os.path.join(DATASET_DIR, student_name, f"{count}.jpg")
                cv2.imwrite(img_path, face_img)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Capturing {count}/30", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Register - Press Q to Stop", frame)

        if cv2.waitKey(1) & 0xFF == ord("q") or count >= 30:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Captured {count} images for {student_name}")
    return count > 0


# ================= TRAIN MODEL =================
def train_lbph():
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    faces = []
    labels = []

    students = load_students()

    for student_id, info in students.items():
        person_dir = os.path.join(DATASET_DIR, info["name"])

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

    ws.append([student_id, name, date_str, time_str])
    wb.save(ATTENDANCE_FILE)

    send_email(email, name, date_str, time_str)


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
        name = request.form["name"]
        reg_no = request.form["reg_no"]
        parent_email = request.form["parent_email"]

        # ✅ Correct Order
        save_student(name, reg_no, parent_email)

        if capture_images(name):
            message = "✅ Registration successful!"
        else:
            message = "❌ Registration failed - camera issue."

        return render_template("register.html", message=message)

    return render_template("register.html")


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
    students = load_students()

    if not os.path.exists(TRAINER_FILE):
        return "❌ No trained model found. Train first."

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(TRAINER_FILE)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera not accessible. Trying index 1...")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            return "❌ Camera not accessible."

    cv2.namedWindow("Recognition", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Recognition", cv2.WND_PROP_TOPMOST, 1)

    recognized_ids = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            id_, conf = recognizer.predict(face_img)

            if conf < 60:
                name = students.get(id_, {}).get("name", "Unknown")
                cv2.putText(frame, name, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if id_ not in recognized_ids:
                    mark_attendance(id_)
                    recognized_ids.add(id_)
            else:
                cv2.putText(frame, "Unknown", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

        cv2.imshow("Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Recognition completed. Recognized {len(recognized_ids)} students.")

    return redirect(url_for("index"))

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
            "date": date,
            "time": time
        })

    return render_template("view_attendance.html", attendance=records)

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
