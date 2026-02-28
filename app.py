from flask import Flask, render_template, request
import qrcode
import uuid
import datetime
import sqlite3
import math
import os
import re
import shutil
import cv2
from deepface import DeepFace

app = Flask(__name__)

# CONFIG
CLASS_LAT = 12.96905169
CLASS_LON = 77.7110380
ALLOWED_RADIUS = 0.5  # 500 meters (distance is in KM)
QR_EXPIRY_MINUTES = 2
STUDENT_FACES_DIR = os.path.join("faces", "students")
ALLOWED_UPLOAD_EXTENSIONS = (".jpg", ".jpeg", ".png")


def init_db():
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()

    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        token TEXT,
        expiry TEXT
    )
    """
    )

    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_name TEXT,
        timestamp TEXT
    )
    """
    )

    # Backfill schema for existing DBs so each attendance row can be tied to a QR session token.
    cursor.execute("PRAGMA table_info(attendance)")
    attendance_columns = {row[1] for row in cursor.fetchall()}
    if "session_token" not in attendance_columns:
        cursor.execute("ALTER TABLE attendance ADD COLUMN session_token TEXT")

    conn.commit()
    conn.close()


init_db()


def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def _extract_face(gray_img):
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    faces = cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        return gray_img

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return gray_img[y : y + h, x : x + w]


def _preprocess_for_compare(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    face = _extract_face(img)
    face = cv2.resize(face, (256, 256), interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(face)


def verify_face_fallback(captured_path, stored_path):
    img1 = _preprocess_for_compare(captured_path)
    img2 = _preprocess_for_compare(stored_path)
    if img1 is None or img2 is None:
        return False, "Could not read one or both face images"

    orb = cv2.ORB_create(1200)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return False, "No usable face features detected"

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if not matches:
        return False, "No feature matches found"

    good = [m for m in matches if m.distance < 50]
    orb_ratio = len(good) / len(matches)

    hist1 = cv2.calcHist([img1], [0], None, [64], [0, 256])
    hist2 = cv2.calcHist([img2], [0], None, [64], [0, 256])
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)
    hist_corr = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    hist_score = max(0.0, min(1.0, (hist_corr + 1) / 2))

    score = 0.7 * orb_ratio + 0.3 * hist_score
    return score >= 0.18, f"score={score:.2f}, orb={orb_ratio:.2f}, hist={hist_score:.2f}"


def student_key(student_name):
    key = re.sub(r"[^a-zA-Z0-9]+", "_", student_name.strip().lower()).strip("_")
    return key or "unknown_student"


def get_student_reference_images(student_name):
    key = student_key(student_name)
    folder = os.path.join(STUDENT_FACES_DIR, key)
    if not os.path.isdir(folder):
        return [], folder

    images = []
    for name in sorted(os.listdir(folder)):
        if name.lower().endswith((".jpg", ".jpeg", ".png")):
            images.append(os.path.join(folder, name))
    return images, folder


def verify_with_single_reference(captured_path, reference_path):
    try:
        verify = DeepFace.verify(
            img1_path=captured_path,
            img2_path=reference_path,
            enforce_detection=False,
        )
        if verify.get("verified", False):
            return True, "deepface_match"
    except Exception:
        pass

    matched, info = verify_face_fallback(captured_path, reference_path)
    return matched, f"fallback_{info}"


def verify_with_student_gallery(captured_path, student_name):
    refs, folder = get_student_reference_images(student_name)
    if not refs:
        return False, (
            f"No reference photos found for '{student_name}'. "
            f"Add images in: {folder}"
        )

    matches = 0
    for ref in refs:
        ok, _ = verify_with_single_reference(captured_path, ref)
        if ok:
            matches += 1

    required = 1 if len(refs) <= 2 else 2
    if matches >= required:
        return True, f"matched {matches}/{len(refs)}"
    return False, f"matched {matches}/{len(refs)} (required {required})"


def save_uploaded_face(file_storage):
    if not file_storage or not file_storage.filename:
        return None, "Capture or upload a face photo"

    ext = os.path.splitext(file_storage.filename)[1].lower()
    if ext not in ALLOWED_UPLOAD_EXTENSIONS:
        return None, "Only .jpg, .jpeg, and .png are allowed"

    os.makedirs("faces", exist_ok=True)
    captured_path = os.path.join("faces", f"captured_{uuid.uuid4().hex}{ext}")
    file_storage.save(captured_path)

    if not os.path.exists(captured_path) or os.path.getsize(captured_path) == 0:
        return None, "Could not save captured face image"

    return captured_path, None


def safe_remove(path):
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register_student_photos():
    if request.method == "POST":
        student_name = request.form.get("student_name", "").strip()
        files = request.files.getlist("photos")

        if not student_name:
            return render_template("result.html", message="Student name is required", status="error")

        if not files or all(not f or not f.filename for f in files):
            return render_template("result.html", message="Upload at least one photo", status="error")

        key = student_key(student_name)
        target_dir = os.path.join(STUDENT_FACES_DIR, key)
        os.makedirs(target_dir, exist_ok=True)

        saved_count = 0
        next_index = len([n for n in os.listdir(target_dir) if n.lower().endswith((".jpg", ".jpeg", ".png"))]) + 1

        for f in files:
            if not f or not f.filename:
                continue

            ext = os.path.splitext(f.filename)[1].lower()
            if ext not in (".jpg", ".jpeg", ".png"):
                continue

            filename = f"{next_index}{ext}"
            save_path = os.path.join(target_dir, filename)
            f.save(save_path)
            saved_count += 1
            next_index += 1

        if saved_count == 0:
            return render_template("result.html", message="No valid images uploaded (.jpg/.jpeg/.png)", status="error")

        return render_template(
            "result.html",
            message=f"Saved {saved_count} photo(s) for {student_name} in {target_dir}",
            status="success",
        )

    return render_template("register.html")


@app.route("/students")
def list_students():
    os.makedirs(STUDENT_FACES_DIR, exist_ok=True)
    students = []

    for folder_name in sorted(os.listdir(STUDENT_FACES_DIR)):
        folder_path = os.path.join(STUDENT_FACES_DIR, folder_name)
        if not os.path.isdir(folder_path):
            continue

        photo_count = len([n for n in os.listdir(folder_path) if n.lower().endswith((".jpg", ".jpeg", ".png"))])
        students.append({"key": folder_name, "photo_count": photo_count, "path": folder_path})

    return render_template("students.html", students=students)


@app.route("/students/delete", methods=["POST"])
def delete_student():
    student_folder = request.form.get("student_folder", "").strip()
    safe_folder = student_key(student_folder)
    target_dir = os.path.join(STUDENT_FACES_DIR, safe_folder)

    if not os.path.isdir(target_dir):
        return render_template("result.html", message="Student folder not found", status="error")

    shutil.rmtree(target_dir)
    return render_template("result.html", message=f"Deleted student data: {safe_folder}", status="success")


@app.route("/teacher")
def teacher_portal():
    return render_template("teacher.html")


@app.route("/teacher/qr")
def teacher_qr():
    token = str(uuid.uuid4())
    expiry = datetime.datetime.now() + datetime.timedelta(minutes=QR_EXPIRY_MINUTES)

    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO sessions (token, expiry) VALUES (?, ?)", (token, str(expiry)))
    conn.commit()
    conn.close()

    if not os.path.exists("static"):
        os.makedirs("static")

    img = qrcode.make(token)
    img.save("static/qr.png")

    return render_template("teacher_qr.html", expiry=QR_EXPIRY_MINUTES * 60)


@app.route("/student", methods=["GET", "POST"])
def student():
    if request.method == "POST":
        token = request.form["token"]
        name = request.form["student_name"].strip()
        lat = float(request.form["lat"])
        lon = float(request.form["lon"])
        uploaded_face = request.files.get("face_photo")

        conn = sqlite3.connect("attendance.db")
        cursor = conn.cursor()

        cursor.execute("SELECT expiry FROM sessions WHERE token=?", (token,))
        result = cursor.fetchone()

        if not result:
            conn.close()
            return render_template("result.html", message="Invalid QR Token", status="error")

        expiry_time = datetime.datetime.strptime(result[0], "%Y-%m-%d %H:%M:%S.%f")
        if datetime.datetime.now() > expiry_time:
            conn.close()
            return render_template("result.html", message="QR Expired", status="error")

        distance = calculate_distance(CLASS_LAT, CLASS_LON, lat, lon)
        if distance > ALLOWED_RADIUS:
            conn.close()
            return render_template("result.html", message="Outside Classroom", status="error")

        captured_path, upload_error = save_uploaded_face(uploaded_face)
        if upload_error:
            conn.close()
            return render_template("result.html", message=upload_error, status="error")

        matched, info = verify_with_student_gallery(captured_path, name)
        if not matched:
            conn.close()
            safe_remove(captured_path)
            return render_template("result.html", message=f"Face Not Matched ({info})", status="error")

        # Prevent duplicate marking for the same student in the same QR session.
        cursor.execute(
            "SELECT 1 FROM attendance WHERE student_name=? AND session_token=? LIMIT 1",
            (name, token),
        )
        if cursor.fetchone():
            conn.close()
            safe_remove(captured_path)
            return render_template(
                "result.html",
                message="Attendance already marked for this session",
                status="error",
            )

        timestamp = str(datetime.datetime.now())
        cursor.execute(
            "INSERT INTO attendance (student_name, timestamp, session_token) VALUES (?, ?, ?)",
            (name, timestamp, token),
        )
        conn.commit()
        conn.close()
        safe_remove(captured_path)

        return render_template("result.html", message="Attendance Marked Successfully", status="success")

    return render_template("student.html")


@app.route("/dashboard")
def dashboard():
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    cursor.execute("SELECT student_name, COUNT(*) FROM attendance GROUP BY student_name")
    data = cursor.fetchall()
    conn.close()
    return render_template("dashboard.html", data=data)


if __name__ == "__main__":
    app.run(debug=True)

