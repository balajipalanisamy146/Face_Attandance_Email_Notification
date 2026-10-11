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
from supabase import create_client

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



SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv(
    "SUPABASE_SERVICE_ROLE_KEY"
)

supabase_storage = None

if IS_RENDER:
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise RuntimeError(
            "Supabase Storage credentials are missing."
        )

    supabase_storage = create_client(
        SUPABASE_URL,
        SUPABASE_SERVICE_ROLE_KEY
    )

FACE_BUCKET = "face-datasets"



# ============================================================
# SUPABASE STORAGE HELPERS — RENDER ONLY
# ============================================================

FACE_IMAGES_PREFIX = "students"
MODEL_STORAGE_PATH = "models/trainer.yml"



def list_cloud_face_images(student_id):
    """List face images stored for a student in Supabase Storage."""

    if not IS_RENDER:
        return []

    if supabase_storage is None:
        raise RuntimeError("Supabase Storage is not configured.")

    folder = f"{FACE_IMAGES_PREFIX}/{student_id}"

    try:
        result = (
            supabase_storage
            .storage
            .from_(FACE_BUCKET)
            .list(
                folder,
                {
                    "limit": 100,
                    "offset": 0
                }
            )
        )

        image_paths = []

        for item in result or []:
            filename = item.get("name", "")

            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(f"{folder}/{filename}")

        def sort_key(path):
            filename = os.path.splitext(path.rsplit("/", 1)[-1])[0]
            if filename.isdigit():
                return (0, int(filename))
            return (1, filename)

        return sorted(image_paths, key=sort_key)

    except Exception as error:
        app.logger.exception(
            "Unable to list face images in Supabase Storage: %s",
            folder
        )
        raise RuntimeError(
            "Could not list face images in Supabase Storage."
        ) from error



def upload_face_image(student_id, image_number, image_bytes):
    """Upload one captured face image to Supabase Storage."""

    if not IS_RENDER or supabase_storage is None:
        raise RuntimeError("Supabase Storage is not configured.")

    path = f"{FACE_IMAGES_PREFIX}/{student_id}/{image_number}.jpg"

    try:
        result = (
            supabase_storage.storage
            .from_(FACE_BUCKET)
            .upload(
                path,
                image_bytes,
                file_options={
                    "content-type": "image/jpeg",
                    "upsert": "true"
                }
            )
        )

        app.logger.info(
            "Face image uploaded successfully: %s",
            path
        )
        return result

    except Exception:
        app.logger.exception(
            "Supabase upload failed. Bucket=%s, path=%s",
            FACE_BUCKET,
            path
        )
        raise



def download_trainer_from_cloud():
    """Download the trained model into the local runtime cache."""

    if not IS_RENDER or supabase_storage is None:
        return False

    try:
        model_bytes = supabase_storage.storage.from_(
            FACE_BUCKET
        ).download(MODEL_STORAGE_PATH)

        if not model_bytes:
            return False

        os.makedirs(
            os.path.dirname(TRAINER_FILE),
            exist_ok=True
        )

        with open(TRAINER_FILE, "wb") as model_file:
            model_file.write(model_bytes)

        return True

    except Exception as error:
        print("Cloud model download failed:", error)
        return False


def upload_trainer_to_cloud():
    """Persist trainer.yml in Supabase Storage."""

    if not IS_RENDER or supabase_storage is None:
        return True

    with open(TRAINER_FILE, "rb") as model_file:
        model_bytes = model_file.read()

    supabase_storage.storage.from_(
        FACE_BUCKET
    ).upload(
        MODEL_STORAGE_PATH,
        model_bytes,
        file_options={
            "content-type": "application/octet-stream",
            "upsert": "true"
        }
    )

    print("Trained model uploaded to Supabase Storage.")
    return True



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
        print(" Parent email not available.")
        return False

    if not EMAIL_PASSWORD:

        print(
            " EMAIL_PASSWORD is not configured."
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
            " DATABASE_URL is not configured."
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

    if not hasattr(cv2, "face"):
        raise RuntimeError(
            "OpenCV face recognition is unavailable. "
            "Install opencv-contrib-python-headless."
        )

    with recognizer_lock:
        # Fetch the persistent model when this process first needs it.
        if IS_RENDER and cached_recognizer is None:
            if not download_trainer_from_cloud():
                raise RuntimeError(
                    "No trained model found in Supabase Storage. "
                    "Capture face images and train the model first."
                )

        if not os.path.isfile(TRAINER_FILE):
            raise RuntimeError(
                "No trained model found. Train the model first."
            )

        trainer_mtime = os.path.getmtime(TRAINER_FILE)

        if (
            cached_recognizer is None
            or cached_trainer_mtime != trainer_mtime
        ):
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read(TRAINER_FILE)

            cached_recognizer = recognizer
            cached_trainer_mtime = trainer_mtime

        return cached_recognizer



# ============================================================
# TRAIN MODEL
# ============================================================


def train_lbph():
    global cached_recognizer
    global cached_trainer_mtime

    if not hasattr(cv2, "face"):
        print("cv2.face is unavailable.")
        return False

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    faces = []
    labels = []

    students = load_students()

    for student_id, info in students.items():

        if IS_RENDER:
            # Read the student's face images from Supabase Storage.
            try:
                image_paths = list_cloud_face_images(student_id)

                for image_path in image_paths:
                    image_bytes = supabase_storage.storage.from_(
                        FACE_BUCKET
                    ).download(image_path)

                    img = cv2.imdecode(
                        np.frombuffer(image_bytes, dtype=np.uint8),
                        cv2.IMREAD_GRAYSCALE
                    )

                    if img is None:
                        print("Invalid cloud image:", image_path)
                        continue

                    faces.append(img)
                    labels.append(int(student_id))

            except Exception as error:
                print(
                    f"Could not retrieve images for student {student_id}:",
                    error
                )
                return False

        else:
            # Preserve the existing local dataset behaviour.
            person_dir = student_dataset_dir(
                student_id, info["name"]
            )

            if not os.path.isdir(person_dir):
                print("Folder Missing:", person_dir)
                continue

            for img_file in os.listdir(person_dir):
                if not img_file.lower().endswith(
                    (".jpg", ".jpeg", ".png")
                ):
                    continue

                img_path = os.path.join(person_dir, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                if img is None:
                    continue

                faces.append(img)
                labels.append(int(student_id))

    if not faces:
        print("No training images found!")
        return False

    try:
        recognizer.train(faces, np.array(labels, dtype=np.int32))

        os.makedirs(
            os.path.dirname(os.path.abspath(TRAINER_FILE)),
            exist_ok=True
        )

        recognizer.save(TRAINER_FILE)

        # On Render, save a persistent copy in Supabase Storage.
        if IS_RENDER:
            upload_trainer_to_cloud()

    except Exception as error:
        print("Model training or upload failed:", error)
        return False

    with recognizer_lock:
        cached_recognizer = None
        cached_trainer_mtime = None

    print(
        f"Training completed successfully using {len(faces)} images."
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
                " No attendance file found."
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


@app.route("/capture_frame/<int:student_id>", methods=["POST"])
def capture_frame(student_id):
    students = load_students()

    if student_id not in students:
        return jsonify(
            error="Student registration was not found."
        ), 404

    frame = decode_browser_frame()

    if frame is None:
        return jsonify(
            error="A valid camera frame is required."
        ), 400

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, 1.3, 5
    )

    # Count existing images in the correct storage.
    try:
        if IS_RENDER:
            existing_images = list_cloud_face_images(student_id)
            captured_count = len(existing_images)
        else:
            student_dir = os.path.join(
                DATASET_DIR, str(student_id)
            )
            os.makedirs(student_dir, exist_ok=True)

            existing_images = [
                f for f in os.listdir(student_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            captured_count = len(existing_images)

    except Exception as error:
        print("Unable to list face images:", error)
        return jsonify(
            error="Unable to access face-image storage. Check Supabase Storage configuration."
        ), 503

    if captured_count >= 30:
        return jsonify(captured=captured_count, complete=True)

    if len(faces) == 0:
        return jsonify(
            captured=captured_count,
            complete=False,
            message="No face detected. Look at the camera."
        )

    if len(faces) > 1:
        return jsonify(
            captured=captured_count,
            complete=False,
            message="Only one person should be in the frame."
        )

    now = time.monotonic()

    with capture_lock:
        last_capture = capture_last_frame_at.get(student_id, 0)

        if now - last_capture < 0.85:
            return jsonify(
                captured=captured_count,
                complete=False,
                message="Hold still for a moment."
            )

        x, y, width, height = faces[0]

        face_image = cv2.resize(
            gray[y:y + height, x:x + width],
            (200, 200)
        )

        next_image_number = captured_count + 1

        if IS_RENDER:
            success, encoded = cv2.imencode(".jpg", face_image)

            if not success:
                return jsonify(
                    error="Could not encode captured face image."
                ), 500

            try:
                upload_face_image(
                    student_id,
                    next_image_number,
                    encoded.tobytes()
                )
            except Exception as error:
                print("Supabase face upload failed:", error)
                return jsonify(
                    error="Could not upload face image to Supabase Storage."
                ), 503

        else:
            filename = os.path.join(
                DATASET_DIR,
                str(student_id),
                f"{next_image_number}.jpg"
            )

            if not cv2.imwrite(filename, face_image):
                return jsonify(
                    error="Could not save the captured face image."
                ), 500

        capture_last_frame_at[student_id] = now
        captured_count = next_image_number

    return jsonify(
        captured=captured_count,
        complete=captured_count >= 30,
        message=(
            "Face captured."
            if captured_count < 30
            else "Registration photos captured."
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
    try:
        get_trained_recognizer()

    except RuntimeError as error:
        return jsonify(
            error=str(error)
        ), 503

    except Exception as error:
        app.logger.exception("Failed to load trained model")
        return jsonify(
            error="Unable to load trained model. Please try training again."
        ), 500

    return jsonify(ready=True)


# ============================================================
# RECOGNITION FRAME
# ============================================================

@app.route(
    "/recognition_frame",
    methods=["POST"]
)
def recognition_frame():

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


@app.route("/debug_model")
def debug_model():
    students = load_students()

    student_folders = []
    total_images = 0

    for student_id, info in students.items():
        folder = student_dataset_dir(
            student_id,
            info["name"]
        )

        images = []

        if os.path.isdir(folder):
            images = [
                f for f in os.listdir(folder)
                if f.lower().endswith(
                    (".jpg", ".jpeg", ".png")
                )
            ]

        total_images += len(images)

        student_folders.append({
            "student_id": student_id,
            "folder": folder,
            "folder_exists": os.path.isdir(folder),
            "image_count": len(images)
        })

    return jsonify({
        "render_mode": IS_RENDER,
        "students_in_database": len(students),
        "dataset_dir": os.path.abspath(DATASET_DIR),
        "dataset_dir_exists": os.path.isdir(DATASET_DIR),
        "trainer_file": os.path.abspath(TRAINER_FILE),
        "trainer_exists": os.path.isfile(TRAINER_FILE),
        "total_images": total_images,
        "student_folders": student_folders
    })


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