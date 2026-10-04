import os
import csv
import cv2
import time
import threading
import smtplib

import numpy as np
import psycopg2

from io import BytesIO
from datetime import datetime

from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    send_file
)

from openpyxl import Workbook, load_workbook
from email.mime.text import MIMEText
from werkzeug.utils import secure_filename

from dotenv import load_dotenv

load_dotenv()


# ============================================================
# STORAGE CONFIGURATION
# ============================================================

IS_RENDER = os.getenv("RENDER") == "true"

if IS_RENDER:
    # Render Persistent Disk
    STORAGE_DIR = "/opt/render/project/src/storage"

    DATASET_DIR = os.path.join(
        STORAGE_DIR,
        "dataset"
    )

    TRAINER_FILE = os.path.join(
        STORAGE_DIR,
        "trainer.yml"
    )

else:
    # Local storage - unchanged
    STORAGE_DIR = "."

    DATASET_DIR = "dataset"

    TRAINER_FILE = "trainer.yml"


# Local-only files
STUDENTS_FILE = "students.csv"
ATTENDANCE_FILE = "attendance.xlsx"

# Make sure dataset folder exists
os.makedirs(
    DATASET_DIR,
    exist_ok=True
)

# ------------------------------------------------------------
# Environment variables
# ------------------------------------------------------------

DATABASE_URL = os.getenv("DATABASE_URL")

EMAIL_SENDER = os.getenv("EMAIL_SENDER")

EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")



# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


recognizer_lock = threading.Lock()

cached_recognizer = None
cached_trainer_mtime = None

capture_last_frame_at = {}
capture_lock = threading.Lock()


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_db_connection():
    """
    Connect to Supabase PostgreSQL.

    This is used only when the application is running
    on Render.
    """

    if not DATABASE_URL:
        return None

    database_url = DATABASE_URL

    # Some providers return postgres://
    # psycopg2 expects postgresql://
    if database_url.startswith("postgres://"):
        database_url = database_url.replace(
            "postgres://",
            "postgresql://",
            1
        )

    return psycopg2.connect(database_url)


# ============================================================
# CREATE SUPABASE TABLES
# ============================================================

def initialize_database():

    if not IS_RENDER:
        return

    if not DATABASE_URL:
        print("DATABASE_URL not configured.")
        return

    conn = None

    try:

        conn = get_db_connection()

        with conn.cursor() as cur:

            cur.execute("""
                CREATE TABLE IF NOT EXISTS students (
                    id BIGSERIAL PRIMARY KEY,
                    reg_no VARCHAR(50) UNIQUE NOT NULL,
                    name VARCHAR(150) NOT NULL,
                    parent_email VARCHAR(255) NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS attendance (
                    id BIGSERIAL PRIMARY KEY,
                    student_id BIGINT NOT NULL
                        REFERENCES students(id)
                        ON DELETE CASCADE,

                    reg_no VARCHAR(50) NOT NULL,
                    name VARCHAR(150) NOT NULL,

                    attendance_date DATE NOT NULL,
                    attendance_time TIME NOT NULL,

                    created_at TIMESTAMPTZ DEFAULT NOW(),

                    UNIQUE(student_id, attendance_date)
                );
            """)

        conn.commit()

        print("Supabase database connected.")
        print("Database tables ready.")

    except Exception as e:

        print("Database initialization error:", e)

    finally:

        if conn:
            conn.close()



# ============================================================
# HAAR CASCADE
# ============================================================

cascade_filename = "haarcascade_frontalface_default.xml"

cascade_paths = (
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data",
        cascade_filename
    ),

    os.path.join(
        cv2.data.haarcascades,
        cascade_filename
    ),
)

face_cascade = None

for cascade_path in cascade_paths:

    if os.path.isfile(cascade_path):

        candidate = cv2.CascadeClassifier(
            cascade_path
        )

        if not candidate.empty():

            face_cascade = candidate
            break


if face_cascade is None:

    raise RuntimeError(
        "Unable to load the Haar cascade. "
        "Checked: " + ", ".join(cascade_paths)
    )


# ============================================================
# EMAIL
# ============================================================

def send_email(
    to_email,
    student_name,
    date_str,
    time_str
):

    if not to_email:
        print("⚠️ Parent email not available.")
        return False

    if not EMAIL_PASSWORD:

        print(
            "⚠️ EMAIL_PASSWORD is not configured."
        )

        return False

    try:

        msg = MIMEText(
            f"""Dear Parent,

Your child {student_name} attended on {date_str} at {time_str}.

Regards,
Attendance System
"""
        )

        msg["Subject"] = "Attendance Notification"
        msg["From"] = EMAIL_SENDER
        msg["To"] = to_email

        with smtplib.SMTP_SSL(
            "smtp.gmail.com",
            465
        ) as server:

            server.login(
                EMAIL_SENDER,
                EMAIL_PASSWORD
            )

            server.sendmail(
                EMAIL_SENDER,
                to_email,
                msg.as_string()
            )

        print("Email sent to", to_email)

        return True

    except Exception as e:

        print("Email Error:", e)

        return False


# ============================================================
# STUDENTS
# ============================================================

def load_students():

    students = {}

    # ========================================================
    # LOCAL
    # ========================================================

    if not IS_RENDER:

        if os.path.exists(STUDENTS_FILE):

            with open(
                STUDENTS_FILE,
                newline="",
                encoding="utf-8"
            ) as file:

                reader = csv.DictReader(file)

                for row in reader:

                    if (
                        not row.get("id")
                        or not row.get("name")
                    ):
                        continue

                    parent_email = (
                        row.get("parent_email")
                        or row.get("parent_email1")
                        or ""
                    )

                    try:

                        student_id = int(
                            row["id"]
                        )

                    except (
                        TypeError,
                        ValueError
                    ):

                        continue

                    students[student_id] = {

                        "name": row.get(
                            "name",
                            ""
                        ),

                        "reg_no": row.get(
                            "reg_no",
                            ""
                        ),

                        "parent_email":
                            parent_email
                    }

        return students

    # ========================================================
    # RENDER / SUPABASE
    # ========================================================

    conn = get_db_connection()

    if conn is None:

        print(
            "⚠️ DATABASE_URL is not configured."
        )

        return students

    try:

        with conn.cursor() as cur:

            cur.execute("""
                SELECT
                    id,
                    reg_no,
                    name,
                    parent_email
                FROM students
                ORDER BY id
            """)

            rows = cur.fetchall()

            for row in rows:

                student_id = row[0]

                students[int(student_id)] = {

                    "name": row[2],

                    "reg_no": row[1],

                    "parent_email":
                        row[3] or ""
                }

    except Exception as e:

        print(
            "Error loading students:",
            e
        )

    finally:

        conn.close()

    return students


# ============================================================
# SAVE STUDENT
# ============================================================

def save_student(
    name,
    reg_no,
    parent_email
):

    # ========================================================
    # LOCAL
    # ========================================================

    if not IS_RENDER:

        students = load_students()

        student_id = len(students) + 1

        os.makedirs(
            os.path.join(
                DATASET_DIR,
                str(student_id)
            ),
            exist_ok=True
        )

        file_exists = os.path.exists(
            STUDENTS_FILE
        )

        write_header = True

        if file_exists:

            write_header = (
                os.path.getsize(
                    STUDENTS_FILE
                ) == 0
            )

        with open(
            STUDENTS_FILE,
            "a",
            newline="",
            encoding="utf-8"
        ) as file:

            fieldnames = [
                "id",
                "reg_no",
                "name",
                "parent_email"
            ]

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames
            )

            if write_header:
                writer.writeheader()

            writer.writerow({

                "id": student_id,

                "reg_no": reg_no,

                "name": name,

                "parent_email":
                    parent_email
            })

        return student_id

    # ========================================================
    # RENDER / SUPABASE
    # ========================================================

    conn = get_db_connection()

    if conn is None:

        raise RuntimeError(
            "DATABASE_URL is not configured."
        )

    try:

        with conn.cursor() as cur:

            cur.execute("""
                INSERT INTO students
                    (
                        reg_no,
                        name,
                        parent_email
                    )
                VALUES
                    (%s, %s, %s)
                RETURNING id
            """, (
                reg_no,
                name,
                parent_email
            ))

            student_id = cur.fetchone()[0]

        conn.commit()

        # Create temporary dataset folder
        # for the current Render instance.
        os.makedirs(
            os.path.join(
                DATASET_DIR,
                str(student_id)
            ),
            exist_ok=True
        )

        return student_id

    except psycopg2.errors.UniqueViolation:

        conn.rollback()

        raise ValueError(
            "Register number already exists."
        )

    except Exception:

        conn.rollback()

        raise

    finally:

        conn.close()


# ============================================================
# BROWSER FRAME
# ============================================================

def decode_browser_frame():

    uploaded_frame = request.files.get(
        "frame"
    )

    if uploaded_frame is None:
        return None

    encoded_frame = np.frombuffer(
        uploaded_frame.read(),
        dtype=np.uint8
    )

    if encoded_frame.size == 0:
        return None

    return cv2.imdecode(
        encoded_frame,
        cv2.IMREAD_COLOR
    )


# ============================================================
# STUDENT DATASET DIRECTORY
# ============================================================

def student_dataset_dir(
    student_id,
    student_name
):

    student_dir = os.path.join(
        DATASET_DIR,
        str(student_id)
    )

    if (
        os.path.isdir(student_dir)
        and os.listdir(student_dir)
    ):

        return student_dir

    if (
        os.path.basename(student_name)
        != student_name
        or student_name in (".", "..")
    ):

        return os.path.join(
            os.path.abspath(DATASET_DIR),
            secure_filename(student_name)
            or str(student_id)
        )

    legacy_dir = os.path.abspath(
        os.path.join(
            DATASET_DIR,
            student_name
        )
    )

    dataset_root = os.path.abspath(
        DATASET_DIR
    )

    if (
        os.path.commonpath(
            (
                dataset_root,
                legacy_dir
            )
        )
        == dataset_root
    ):

        return legacy_dir

    return os.path.join(
        dataset_root,
        secure_filename(student_name)
        or str(student_id)
    )


# ============================================================
# TRAINED RECOGNIZER
# ============================================================

def get_trained_recognizer():

    global cached_recognizer
    global cached_trainer_mtime

    if not os.path.exists(
        TRAINER_FILE
    ):

        raise RuntimeError(
            "No trained model found. "
            "Train the model first."
        )

    trainer_mtime = os.path.getmtime(
        TRAINER_FILE
    )

    with recognizer_lock:

        if (
            cached_recognizer is None
            or cached_trainer_mtime
            != trainer_mtime
        ):

            if not hasattr(cv2, "face"):

                raise RuntimeError(
                    "OpenCV face recognition "
                    "is unavailable. "
                    "Install "
                    "opencv-contrib-python-headless."
                )

            recognizer = (
                cv2.face
                .LBPHFaceRecognizer_create()
            )

            recognizer.read(
                TRAINER_FILE
            )

            cached_recognizer = recognizer

            cached_trainer_mtime = (
                trainer_mtime
            )

        return cached_recognizer


# ============================================================
# TRAIN MODEL
# ============================================================

def train_lbph():

    global cached_recognizer
    global cached_trainer_mtime

    if not hasattr(cv2, "face"):

        print(
            "cv2.face is unavailable."
        )

        return False

    recognizer = (
        cv2.face
        .LBPHFaceRecognizer_create()
    )

    faces = []
    labels = []

    students = load_students()

    for student_id, info in students.items():

        person_dir = student_dataset_dir(
            student_id,
            info["name"]
        )

        if not os.path.exists(
            person_dir
        ):

            print(
                "Folder Missing:",
                person_dir
            )

            continue

        for img_file in os.listdir(
            person_dir
        ):

            img_path = os.path.join(
                person_dir,
                img_file
            )

            img = cv2.imread(
                img_path,
                cv2.IMREAD_GRAYSCALE
            )

            if img is None:
                continue

            faces.append(img)
            labels.append(student_id)

    if len(faces) == 0:

        print(
            "No training images found!"
        )

        return False

    recognizer.train(
        faces,
        np.array(labels)
    )

    recognizer.save(
        TRAINER_FILE
    )

    with recognizer_lock:

        cached_recognizer = None
        cached_trainer_mtime = None

    print(
        "Training Completed Successfully!"
    )

    return True


# ============================================================
# MARK ATTENDANCE
# ============================================================

def mark_attendance(student_id):

    students = load_students()

    if student_id not in students:
        return False

    student = students[student_id]

    name = student["name"]
    reg_no = student["reg_no"]
    email = student["parent_email"]

    now = datetime.now()

    date_str = now.strftime(
        "%Y-%m-%d"
    )

    time_str = now.strftime(
        "%H:%M:%S"
    )

    # ========================================================
    # LOCAL
    # ========================================================

    if not IS_RENDER:

        if not os.path.exists(
            ATTENDANCE_FILE
        ):

            wb = Workbook()

            ws = wb.active

            ws.append([
                "ID",
                "Name",
                "Date",
                "Time"
            ])

        else:

            wb = load_workbook(
                ATTENDANCE_FILE
            )

            ws = wb.active

            for row in ws.iter_rows(
                min_row=2,
                values_only=True
            ):

                row_date = row[2]

                if isinstance(
                    row_date,
                    datetime
                ):

                    row_date = (
                        row_date.strftime(
                            "%Y-%m-%d"
                        )
                    )

                try:

                    same_student = (
                        int(row[0])
                        == int(student_id)
                    )

                except (
                    TypeError,
                    ValueError
                ):

                    same_student = False

                if (
                    same_student
                    and str(row_date)
                    == date_str
                ):

                    wb.close()

                    return False

        ws.append([
            student_id,
            name,
            date_str,
            time_str
        ])

        wb.save(
            ATTENDANCE_FILE
        )

        wb.close()

        send_email(
            email,
            name,
            date_str,
            time_str
        )

        return True

    # ========================================================
    # RENDER / SUPABASE
    # ========================================================

    conn = get_db_connection()

    if conn is None:

        raise RuntimeError(
            "DATABASE_URL is not configured."
        )

    try:

        with conn.cursor() as cur:

            cur.execute("""
                SELECT id
                FROM attendance
                WHERE student_id = %s
                AND attendance_date = %s
            """, (
                student_id,
                date_str
            ))

            existing = cur.fetchone()

            if existing:

                return False

            cur.execute("""
                INSERT INTO attendance
                (
                    student_id,
                    reg_no,
                    name,
                    attendance_date,
                    attendance_time
                )
                VALUES
                (%s, %s, %s, %s, %s)
            """, (
                student_id,
                reg_no,
                name,
                date_str,
                time_str
            ))

        conn.commit()

        send_email(
            email,
            name,
            date_str,
            time_str
        )

        return True

    except Exception:

        conn.rollback()

        raise

    finally:

        conn.close()


# ============================================================
# HOME
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# ============================================================
# NOTIFY TODAY
# ============================================================

@app.route(
    "/notify_today",
    methods=["GET", "POST"]
)
def notify_today():

    if request.method == "GET":

        return render_template(
            "notify_today.html"
        )

    today_str = datetime.now().strftime(
        "%Y-%m-%d"
    )

    sent = 0

    students = load_students()

    # ========================================================
    # LOCAL
    # ========================================================

    if not IS_RENDER:

        if not os.path.exists(
            ATTENDANCE_FILE
        ):

            message = (
                "❌ No attendance file found."
            )

            return render_template(
                "notify_today.html",
                message=message
            )

        wb = load_workbook(
            ATTENDANCE_FILE
        )

        ws = wb.active

        for row in ws.iter_rows(
            min_row=2,
            values_only=True
        ):

            if len(row) < 4:
                continue

            student_id = row[0]
            name = row[1]
            date = row[2]
            time_value = row[3]

            if hasattr(
                date,
                "strftime"
            ):

                date_str = date.strftime(
                    "%Y-%m-%d"
                )

            else:

                date_str = str(date)

            if date_str != today_str:
                continue

            try:

                student_id = int(
                    student_id
                )

            except (
                TypeError,
                ValueError
            ):

                continue

            if student_id in students:

                email = students[
                    student_id
                ]["parent_email"]

                if hasattr(
                    time_value,
                    "strftime"
                ):

                    time_str = (
                        time_value.strftime(
                            "%H:%M:%S"
                        )
                    )

                else:

                    time_str = str(
                        time_value
                    )

                if send_email(
                    email,
                    name,
                    date_str,
                    time_str
                ):

                    sent += 1

        wb.close()

    # ========================================================
    # RENDER / SUPABASE
    # ========================================================

    else:

        conn = get_db_connection()

        if conn is None:

            message = (
                "Database connection "
                "not configured."
            )

            return render_template(
                "notify_today.html",
                message=message
            )

        try:

            with conn.cursor() as cur:

                cur.execute("""
                    SELECT
                        student_id,
                        name,
                        attendance_date,
                        attendance_time
                    FROM attendance
                    WHERE attendance_date = %s
                    ORDER BY attendance_time
                """, (
                    today_str,
                ))

                rows = cur.fetchall()

                for row in rows:

                    student_id = row[0]
                    name = row[1]
                    date_value = row[2]
                    time_value = row[3]

                    if student_id not in students:
                        continue

                    email = students[
                        student_id
                    ]["parent_email"]

                    date_str = (
                        date_value.strftime(
                            "%Y-%m-%d"
                        )
                    )

                    time_str = (
                        time_value.strftime(
                            "%H:%M:%S"
                        )
                    )

                    if send_email(
                        email,
                        name,
                        date_str,
                        time_str
                    ):

                        sent += 1

        finally:

            conn.close()

    message = (
        f"Notifications sent to "
        f"{sent} students present today."
    )

    return render_template(
        "notify_today.html",
        message=message
    )


# ============================================================
# REGISTER
# ============================================================

@app.route(
    "/register",
    methods=["GET", "POST"]
)
def register():

    if request.method == "POST":

        student_data = (
            request.get_json()
            if request.is_json
            else request.form
        )

        name = str(
            student_data.get(
                "name",
                ""
            )
        ).strip()

        reg_no = str(
            student_data.get(
                "reg_no",
                ""
            )
        ).strip()

        parent_email = str(
            student_data.get(
                "parent_email",
                ""
            )
        ).strip()

        if (
            not name
            or not reg_no
            or not parent_email
        ):

            return jsonify(
                error=(
                    "Name, register number, "
                    "and parent email are required."
                )
            ), 400

        try:

            student_id = save_student(
                name,
                reg_no,
                parent_email
            )

        except ValueError as error:

            return jsonify(
                error=str(error)
            ), 400

        except Exception as error:

            print(
                "Registration error:",
                error
            )

            return jsonify(
                error=(
                    "Unable to register "
                    "student."
                )
            ), 500

        return jsonify(
            student_id=student_id,
            name=name,
            target_images=30
        ), 201

    return render_template(
        "register.html"
    )


# ============================================================
# CAPTURE FRAME
# ============================================================

@app.route(
    "/capture_frame/<int:student_id>",
    methods=["POST"]
)
def capture_frame(student_id):

    students = load_students()

    if student_id not in students:

        return jsonify(
            error=(
                "Student registration "
                "was not found."
            )
        ), 404

    frame = decode_browser_frame()

    if frame is None:

        return jsonify(
            error=(
                "A valid camera frame "
                "is required."
            )
        ), 400

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        1.3,
        5
    )

    student_dir = os.path.join(
        DATASET_DIR,
        str(student_id)
    )

    os.makedirs(
        student_dir,
        exist_ok=True
    )

    captured_files = [

        filename

        for filename in os.listdir(
            student_dir
        )

        if filename.lower().endswith(
            (
                ".jpg",
                ".jpeg",
                ".png"
            )
        )
    ]

    captured_count = len(
        captured_files
    )

    if captured_count >= 30:

        return jsonify(
            captured=captured_count,
            complete=True
        )

    if len(faces) == 0:

        return jsonify(
            captured=captured_count,
            complete=False,
            message=(
                "No face detected. "
                "Look at the camera."
            )
        )

    if len(faces) > 1:

        return jsonify(
            captured=captured_count,
            complete=False,
            message=(
                "Only one person "
                "should be in the frame."
            )
        )

    now = time.monotonic()

    with capture_lock:

        last_capture = (
            capture_last_frame_at.get(
                student_id,
                0
            )
        )

        if (
            now - last_capture
            < 0.85
        ):

            return jsonify(
                captured=captured_count,
                complete=False,
                message=(
                    "Hold still for "
                    "a moment."
                )
            )

        x, y, width, height = faces[0]

        face_image = cv2.resize(
            gray[
                y:y + height,
                x:x + width
            ],
            (200, 200)
        )

        filename = os.path.join(
            student_dir,
            f"{captured_count + 1}.jpg"
        )

        if not cv2.imwrite(
            filename,
            face_image
        ):

            return jsonify(
                error=(
                    "Could not save "
                    "the captured face image."
                )
            ), 500

        capture_last_frame_at[
            student_id
        ] = now

        captured_count += 1

    return jsonify(
        captured=captured_count,
        complete=(
            captured_count >= 30
        ),
        message=(
            "Face captured."
            if captured_count < 30
            else
            "Registration photos captured."
        )
    )


# ============================================================
# TRAIN
# ============================================================

@app.route(
    "/train",
    methods=["GET", "POST"]
)
def train():

    if request.method == "POST":

        if train_lbph():

            message = (
                "Training Completed Successfully!"
            )

        else:

            message = (
                "No Dataset Found. "
                "Please Register Students First."
            )

        return render_template(
            "train.html",
            message=message
        )

    return render_template(
        "train.html"
    )


# ============================================================
# RECOGNIZE
# ============================================================

@app.route("/recognize")
def recognize():

    return render_template(
        "recognize.html"
    )


# ============================================================
# START RECOGNITION
# ============================================================

@app.route(
    "/start_recognition",
    methods=["POST"]
)
def start_recognition():

    if not os.path.exists(
        TRAINER_FILE
    ):

        return jsonify(
            error=(
                "No trained model found. "
                "Train the model first."
            )
        ), 400

    try:

        get_trained_recognizer()

    except RuntimeError as error:

        return jsonify(
            error=str(error)
        ), 503

    return jsonify(
        ready=True
    )


# ============================================================
# RECOGNITION FRAME
# ============================================================

@app.route(
    "/recognition_frame",
    methods=["POST"]
)
def recognition_frame():

    if not os.path.exists(
        TRAINER_FILE
    ):

        return jsonify(
            error=(
                "No trained model found. "
                "Train the model first."
            )
        ), 400

    frame = decode_browser_frame()

    if frame is None:

        return jsonify(
            error=(
                "A valid camera frame "
                "is required."
            )
        ), 400

    try:

        recognizer = (
            get_trained_recognizer()
        )

    except RuntimeError as error:

        return jsonify(
            error=str(error)
        ), 503

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        1.3,
        5
    )

    students = load_students()

    results = []

    for x, y, width, height in faces:

        face_image = cv2.resize(
            gray[
                y:y + height,
                x:x + width
            ],
            (200, 200)
        )

        with recognizer_lock:

            student_id, confidence = (
                recognizer.predict(
                    face_image
                )
            )

        student = students.get(
            student_id
        )

        if (
            confidence < 60
            and student
        ):

            attendance_added = (
                mark_attendance(
                    student_id
                )
            )

            results.append({

                "name":
                    student["name"],

                "box": [
                    int(x),
                    int(y),
                    int(width),
                    int(height)
                ],

                "status": (
                    "Attendance marked"
                    if attendance_added
                    else
                    "Already marked today"
                )
            })

        else:

            results.append({

                "name": "Unknown",

                "box": [
                    int(x),
                    int(y),
                    int(width),
                    int(height)
                ],

                "status":
                    "Not recognized"
            })

    return jsonify(
        faces=results
    )


# ============================================================
# VIEW ATTENDANCE
# ============================================================

@app.route("/view_attendance")
def view_attendance():

    records = []

    students = load_students()

    # ========================================================
    # LOCAL
    # ========================================================

    if not IS_RENDER:

        if not os.path.exists(
            ATTENDANCE_FILE
        ):

            return (
                "No attendance file found."
            )

        wb = load_workbook(
            ATTENDANCE_FILE
        )

        ws = wb.active

        first_row = True

        for row in ws.iter_rows(
            values_only=True
        ):

            if first_row:

                first_row = False

                continue

            if len(row) < 4:
                continue

            try:

                student_id = int(
                    row[0]
                )

            except (
                TypeError,
                ValueError
            ):

                continue

            name = row[1]
            date = row[2]
            time_value = row[3]

            reg_no = students.get(
                student_id,
                {}
            ).get(
                "reg_no",
                "N/A"
            )

            date_str = (
                date.strftime(
                    "%Y-%m-%d"
                )
                if hasattr(
                    date,
                    "strftime"
                )
                else str(date)
            )

            time_str = (
                time_value.strftime(
                    "%H:%M:%S"
                )
                if hasattr(
                    time_value,
                    "strftime"
                )
                else str(time_value)
            )

            records.append({

                "reg_no": reg_no,

                "name": name,

                "date": date_str,

                "time": time_str
            })

        wb.close()

    # ========================================================
    # RENDER / SUPABASE
    # ========================================================

    else:

        conn = get_db_connection()

        if conn is None:

            return (
                "Database connection "
                "not configured."
            )

        try:

            with conn.cursor() as cur:

                cur.execute("""
                    SELECT
                        reg_no,
                        name,
                        attendance_date,
                        attendance_time
                    FROM attendance
                    ORDER BY
                        attendance_date DESC,
                        attendance_time DESC
                """)

                rows = cur.fetchall()

                for row in rows:

                    records.append({

                        "reg_no": row[0],

                        "name": row[1],

                        "date":
                            row[2].strftime(
                                "%Y-%m-%d"
                            ),

                        "time":
                            row[3].strftime(
                                "%H:%M:%S"
                            )
                    })

        finally:

            conn.close()

    today = datetime.now().strftime(
        "%Y-%m-%d"
    )

    return render_template(
        "view_attendance.html",
        attendance=records,
        today=today
    )


# ============================================================
# DOWNLOAD ATTENDANCE EXCEL
# ============================================================

@app.route(
    "/download_excel_attendance"
)
def download_excel_attendance():

    # ========================================================
    # LOCAL
    # ========================================================

    if not IS_RENDER:

        if not os.path.exists(
            ATTENDANCE_FILE
        ):

            return (
                "No attendance file found."
            )

        return send_file(
            ATTENDANCE_FILE,
            as_attachment=True,
            download_name="attendance.xlsx",
            mimetype=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            )
        )

    # ========================================================
    # RENDER / SUPABASE
    # ========================================================

    conn = get_db_connection()

    if conn is None:

        return (
            "Database connection "
            "not configured."
        )

    try:

        with conn.cursor() as cur:

            cur.execute("""
                SELECT
                    reg_no,
                    name,
                    attendance_date,
                    attendance_time
                FROM attendance
                ORDER BY
                    attendance_date,
                    attendance_time
            """)

            rows = cur.fetchall()

    finally:

        conn.close()

    wb = Workbook()

    ws = wb.active

    ws.title = "Attendance"

    ws.append([
        "Register Number",
        "Name",
        "Date",
        "Time"
    ])

    for row in rows:

        ws.append([
            row[0],
            row[1],
            row[2].strftime(
                "%Y-%m-%d"
            ),
            row[3].strftime(
                "%H:%M:%S"
            )
        ])

    output = BytesIO()

    wb.save(output)

    output.seek(0)

    return send_file(
        output,
        as_attachment=True,
        download_name="attendance.xlsx",
        mimetype=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        )
    )


# ============================================================
# STUDENTS PAGE
# ============================================================

@app.route("/students")
def students():

    students_data = load_students()

    students_list = []

    for student_id, info in students_data.items():

        students_list.append({

            "id": student_id,

            "reg_no":
                info.get(
                    "reg_no",
                    ""
                ),

            "name":
                info.get(
                    "name",
                    ""
                ),

            "parent_email":
                info.get(
                    "parent_email",
                    ""
                )
        })

    return render_template(
        "view_students.html",
        students=students_list
    )


# ============================================================
# REPORT
# ============================================================

@app.route("/report")
def report():

    return render_template(
        "report advaisor.html"
    )


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/dashboard")
def dashboard():

    return render_template(
        "dashboard.html"
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    if not IS_RENDER:

        return jsonify({
            "status": "ok",
            "mode": "local",
            "database": "local files"
        })

    if not DATABASE_URL:

        return jsonify({
            "status": "error",
            "mode": "render",
            "database":
                "DATABASE_URL missing"
        }), 500

    conn = None

    try:

        conn = get_db_connection()

        with conn.cursor() as cur:

            cur.execute(
                "SELECT 1"
            )

            cur.fetchone()

        return jsonify({

            "status": "ok",

            "mode": "render",

            "database":
                "supabase connected"

        })

    except Exception as e:

        return jsonify({

            "status": "error",

            "mode": "render",

            "database":
                str(e)

        }), 500

    finally:

        if conn:
            conn.close()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    os.makedirs(
        DATASET_DIR,
        exist_ok=True
    )

    initialize_database()

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5050,
        use_reloader=False
    )