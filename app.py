from flask import Flask, render_template, request
import qrcode
import uuid
import datetime
import sqlite3
import math
import os
import re
import shutil
import base64
from PIL import Image
import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

try:
    from deepface import DeepFace
except Exception:
    DeepFace = None

app = Flask(__name__)

# CONFIG
DEFAULT_CLASS_LAT = 12.9690516
DEFAULT_CLASS_LON = 77.7110380
DEFAULT_ALLOWED_RADIUS = 0.5  # 500 meters (distance is in KM)
DEFAULT_CAMPUS_NAME = "Main Campus"
DEFAULT_SUBJECT_NAME = "General"
GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY", "")
QR_EXPIRY_MINUTES = 2
STUDENT_FACES_DIR = os.path.join("faces", "students")
ALLOWED_UPLOAD_EXTENSIONS = (".jpg", ".jpeg", ".png")
DEEPFACE_MODEL_NAME = "Facenet512"
# OpenCV detector is much faster than RetinaFace for realtime attendance checks.
DEEPFACE_DETECTOR = "opencv"
DEEPFACE_DISTANCE_METRIC = "cosine"
STRICT_MAX_DISTANCE = 0.28
RELAXED_BEST_DISTANCE = 0.38
MIN_REFERENCE_PHOTOS = 2
EMBEDDING_MATCH_THRESHOLD = 0.38
EMBEDDING_MARGIN = 0.04
EMBEDDING_CACHE_LIMIT = 4096
_embedding_cache = {}


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
    if "subject_name" not in attendance_columns:
        cursor.execute("ALTER TABLE attendance ADD COLUMN subject_name TEXT")

    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS location_settings (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        campus_name TEXT NOT NULL,
        subject_name TEXT NOT NULL,
        class_lat REAL NOT NULL,
        class_lon REAL NOT NULL,
        allowed_radius REAL NOT NULL,
        updated_at TEXT NOT NULL
    )
    """
    )

    cursor.execute("SELECT id FROM location_settings WHERE id=1")
    if not cursor.fetchone():
        cursor.execute(
            """
            INSERT INTO location_settings (id, campus_name, subject_name, class_lat, class_lon, allowed_radius, updated_at)
            VALUES (1, ?, ?, ?, ?, ?, ?)
            """,
            (
                DEFAULT_CAMPUS_NAME,
                DEFAULT_SUBJECT_NAME,
                DEFAULT_CLASS_LAT,
                DEFAULT_CLASS_LON,
                DEFAULT_ALLOWED_RADIUS,
                str(datetime.datetime.now()),
            ),
        )
    else:
        cursor.execute("PRAGMA table_info(location_settings)")
        location_columns = {row[1] for row in cursor.fetchall()}
        if "subject_name" not in location_columns:
            cursor.execute(
                f"ALTER TABLE location_settings ADD COLUMN subject_name TEXT NOT NULL DEFAULT '{DEFAULT_SUBJECT_NAME}'"
            )

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


def get_location_settings():
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    cursor.execute(
        "SELECT campus_name, subject_name, class_lat, class_lon, allowed_radius FROM location_settings WHERE id=1"
    )
    row = cursor.fetchone()
    conn.close()

    if not row:
        return {
            "campus_name": DEFAULT_CAMPUS_NAME,
            "subject_name": DEFAULT_SUBJECT_NAME,
            "class_lat": DEFAULT_CLASS_LAT,
            "class_lon": DEFAULT_CLASS_LON,
            "allowed_radius": DEFAULT_ALLOWED_RADIUS,
        }

    return {
        "campus_name": row[0],
        "subject_name": row[1],
        "class_lat": float(row[2]),
        "class_lon": float(row[3]),
        "allowed_radius": float(row[4]),
    }


def save_location_settings(campus_name, subject_name, class_lat, class_lon, allowed_radius):
    safe_name = campus_name.strip() or DEFAULT_CAMPUS_NAME
    safe_subject = subject_name.strip() or DEFAULT_SUBJECT_NAME
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE location_settings
        SET campus_name=?, subject_name=?, class_lat=?, class_lon=?, allowed_radius=?, updated_at=?
        WHERE id=1
        """,
        (
            safe_name,
            safe_subject,
            class_lat,
            class_lon,
            allowed_radius,
            str(datetime.datetime.now()),
        ),
    )
    conn.commit()
    conn.close()


def _extract_face(gray_img):
    if cv2 is None:
        return gray_img
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    faces = cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
    if len(faces) == 0:
        # Fallback to center crop when detector misses due distance/lighting.
        h, w = gray_img.shape[:2]
        side = int(min(h, w) * 0.7)
        y1 = max(0, (h - side) // 2)
        x1 = max(0, (w - side) // 2)
        return gray_img[y1 : y1 + side, x1 : x1 + side]

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return gray_img[y : y + h, x : x + w]


def count_faces_in_image(img_path):
    if cv2 is None and DeepFace is None:
        # Lightweight/serverless fallback: skip hard face-count gating.
        return 1

    deepface_count = 0
    try:
        if DeepFace is None:
            raise RuntimeError("DeepFace unavailable")
        faces = DeepFace.extract_faces(
            img_path=img_path,
            detector_backend=DEEPFACE_DETECTOR,
            enforce_detection=False,
            align=True,
        )
        if isinstance(faces, list) and faces:
            deepface_count = len(faces)
    except Exception:
        deepface_count = 0

    if cv2 is None:
        return deepface_count if deepface_count > 0 else 1

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return deepface_count

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    faces = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=6, minSize=(50, 50))
    opencv_count = len(faces)

    # Reduce false "multiple faces" rejections:
    # block only when both detectors strongly indicate multiple people.
    if opencv_count > 1 and deepface_count > 1:
        return max(opencv_count, deepface_count)
    if opencv_count == 1 or deepface_count == 1:
        return 1
    if opencv_count > 1 or deepface_count > 1:
        return 1
    return 0


def _preprocess_for_compare(img_path):
    if cv2 is not None:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None

        face = _extract_face(img)
        face = cv2.resize(face, (256, 256), interpolation=cv2.INTER_AREA)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(face)

    try:
        with Image.open(img_path) as im:
            gray = im.convert("L").resize((256, 256))
            arr = np.asarray(gray, dtype=np.float32)
        return arr
    except Exception:
        return None


def verify_face_fallback(captured_path, stored_path):
    img1 = _preprocess_for_compare(captured_path)
    img2 = _preprocess_for_compare(stored_path)
    if img1 is None or img2 is None:
        return False, "Could not read one or both face images"

    if cv2 is None:
        h1, _ = np.histogram(img1.flatten(), bins=64, range=(0, 255), density=True)
        h2, _ = np.histogram(img2.flatten(), bins=64, range=(0, 255), density=True)
        denom = np.linalg.norm(h1) * np.linalg.norm(h2)
        if denom == 0:
            return False, "No usable fallback features"
        sim = float(np.dot(h1, h2) / denom)
        matched = sim >= 0.85
        return matched, f"simple_hist_similarity={sim:.3f}"

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
    matched = score >= 0.32 and orb_ratio >= 0.20 and hist_score >= 0.55
    return matched, f"score={score:.2f}, orb={orb_ratio:.2f}, hist={hist_score:.2f}"


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


def _cosine_distance(vec1, vec2):
    dot = sum(a * b for a, b in zip(vec1, vec2))
    n1 = math.sqrt(sum(a * a for a in vec1))
    n2 = math.sqrt(sum(b * b for b in vec2))
    if n1 == 0 or n2 == 0:
        return 1.0
    cosine_sim = dot / (n1 * n2)
    cosine_sim = max(-1.0, min(1.0, cosine_sim))
    return 1.0 - cosine_sim


def _get_embedding_cache_key(img_path):
    try:
        stat = os.stat(img_path)
    except OSError:
        return None
    return (
        os.path.abspath(img_path),
        stat.st_mtime_ns,
        stat.st_size,
        DEEPFACE_MODEL_NAME,
        DEEPFACE_DETECTOR,
    )


def get_face_embedding(img_path):
    if DeepFace is None:
        return None
    cache_key = _get_embedding_cache_key(img_path)
    if cache_key is not None and cache_key in _embedding_cache:
        return _embedding_cache[cache_key]

    try:
        reps = DeepFace.represent(
            img_path=img_path,
            model_name=DEEPFACE_MODEL_NAME,
            detector_backend=DEEPFACE_DETECTOR,
            enforce_detection=False,
            align=True,
        )
    except Exception:
        return None

    if isinstance(reps, list) and reps:
        emb = reps[0].get("embedding")
        if isinstance(emb, list) and emb:
            if cache_key is not None:
                if len(_embedding_cache) >= EMBEDDING_CACHE_LIMIT:
                    _embedding_cache.clear()
                _embedding_cache[cache_key] = emb
            return emb
    return None


def get_other_students_reference_images(student_name):
    claimed_key = student_key(student_name)
    if not os.path.isdir(STUDENT_FACES_DIR):
        return []

    images = []
    for folder_name in sorted(os.listdir(STUDENT_FACES_DIR)):
        folder_path = os.path.join(STUDENT_FACES_DIR, folder_name)
        if not os.path.isdir(folder_path) or folder_name == claimed_key:
            continue
        for file_name in sorted(os.listdir(folder_path)):
            if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                images.append(os.path.join(folder_path, file_name))
    return images


def verify_identity_separation(captured_path, student_name, claimed_refs):
    captured_emb = get_face_embedding(captured_path)
    if captured_emb is None:
        return True, "separation_skipped_no_capture_embedding"

    claimed_distances = []
    for ref in claimed_refs:
        emb = get_face_embedding(ref)
        if emb is not None:
            claimed_distances.append(_cosine_distance(captured_emb, emb))

    if not claimed_distances:
        return True, "separation_skipped_no_claimed_embeddings"

    claimed_min = min(claimed_distances)
    if claimed_min > EMBEDDING_MATCH_THRESHOLD:
        return False, f"claimed_distance={claimed_min:.3f}_limit={EMBEDDING_MATCH_THRESHOLD:.3f}"

    other_refs = get_other_students_reference_images(student_name)
    if not other_refs:
        return True, f"claimed_distance={claimed_min:.3f}_no_other_gallery"

    other_min = 1.0
    any_other = False
    for ref in other_refs:
        emb = get_face_embedding(ref)
        if emb is None:
            continue
        any_other = True
        d = _cosine_distance(captured_emb, emb)
        if d < other_min:
            other_min = d

    if not any_other:
        return True, f"claimed_distance={claimed_min:.3f}_no_other_embeddings"

    if claimed_min + EMBEDDING_MARGIN > other_min:
        return False, (
            f"identity_conflict(claimed={claimed_min:.3f},other={other_min:.3f},"
            f"margin={EMBEDDING_MARGIN:.3f})"
        )

    return True, f"identity_ok(claimed={claimed_min:.3f},other={other_min:.3f})"


def verify_with_single_reference(captured_path, reference_path, captured_emb=None):
    if captured_emb is None:
        captured_emb = get_face_embedding(captured_path)
    ref_emb = get_face_embedding(reference_path)

    if captured_emb is not None and ref_emb is not None:
        distance = _cosine_distance(captured_emb, ref_emb)
        if distance <= STRICT_MAX_DISTANCE:
            return True, f"deepface_distance={distance:.3f}"
        return False, f"deepface_distance={distance:.3f}_limit={STRICT_MAX_DISTANCE:.3f}"

    # If DeepFace embedding is unavailable, use stricter local fallback.
    matched, info = verify_face_fallback(captured_path, reference_path)
    return matched, f"fallback_{info}"


def verify_with_student_gallery(captured_path, student_name):
    refs, folder = get_student_reference_images(student_name)
    if not refs:
        return False, (
            f"No reference photos found for '{student_name}'. "
            f"Add images in: {folder}"
        )

    if len(refs) < MIN_REFERENCE_PHOTOS:
        return False, (
            f"Add at least {MIN_REFERENCE_PHOTOS} reference photos for '{student_name}'. "
            "Two clear photos are required for secure verification."
        )

    captured_emb = get_face_embedding(captured_path)
    matches = 0
    best_distance = None
    for ref in refs:
        ref_emb = get_face_embedding(ref)
        if captured_emb is not None and ref_emb is not None:
            distance = _cosine_distance(captured_emb, ref_emb)
            if best_distance is None or distance < best_distance:
                best_distance = distance
            if distance <= STRICT_MAX_DISTANCE:
                matches += 1
        else:
            ok, _ = verify_with_single_reference(captured_path, ref, captured_emb=captured_emb)
            if ok:
                matches += 1

    required = 1 if len(refs) <= 3 else 2
    qualifies = matches >= required
    if not qualifies and best_distance is not None and best_distance <= RELAXED_BEST_DISTANCE:
        qualifies = True

    if qualifies:
        sep_ok, sep_info = verify_identity_separation(captured_path, student_name, refs)
        if not sep_ok:
            return False, sep_info
        if best_distance is None:
            return True, f"matched {matches}/{len(refs)} | {sep_info}"
        return True, f"matched {matches}/{len(refs)} best={best_distance:.3f} | {sep_info}"

    if best_distance is not None:
        return False, (
            f"mismatch_best_distance={best_distance:.3f}_"
            f"strict={STRICT_MAX_DISTANCE:.3f}_relaxed={RELAXED_BEST_DISTANCE:.3f}"
        )
    return False, "mismatch"


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


def save_captured_face_data(face_data):
    if not face_data:
        return None, "Capture a face photo"

    if "," in face_data:
        _, face_data = face_data.split(",", 1)

    try:
        raw = base64.b64decode(face_data, validate=True)
    except Exception:
        return None, "Invalid captured face image"

    if not raw:
        return None, "Captured face image is empty"

    os.makedirs("faces", exist_ok=True)
    captured_path = os.path.join("faces", f"captured_{uuid.uuid4().hex}.jpg")
    with open(captured_path, "wb") as f:
        f.write(raw)

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


@app.route("/teacher/location", methods=["GET", "POST"])
def teacher_location():
    settings = get_location_settings()

    if request.method == "POST":
        campus_name = request.form.get("campus_name", "").strip()
        subject_name = request.form.get("subject_name", "").strip()
        lat_raw = request.form.get("class_lat", "").strip()
        lon_raw = request.form.get("class_lon", "").strip()
        radius_raw = request.form.get("allowed_radius", "").strip()

        try:
            class_lat = float(lat_raw)
            class_lon = float(lon_raw)
            allowed_radius = float(radius_raw)
        except ValueError:
            return render_template(
                "result.html",
                message="Invalid location values. Pick a valid point on map and radius.",
                status="error",
            )

        if not (-90 <= class_lat <= 90 and -180 <= class_lon <= 180):
            return render_template(
                "result.html",
                message="Invalid coordinates selected on map.",
                status="error",
            )

        if allowed_radius <= 0 or allowed_radius > 10:
            return render_template(
                "result.html",
                message="Allowed radius must be between 0 and 10 km.",
                status="error",
            )

        save_location_settings(campus_name, subject_name, class_lat, class_lon, allowed_radius)
        return render_template(
            "result.html",
            message="Campus location updated successfully",
            status="success",
        )

    return render_template(
        "teacher_location.html",
        settings=settings,
        google_maps_api_key=GOOGLE_MAPS_API_KEY,
    )


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

    return render_template(
        "teacher_qr.html",
        expiry=QR_EXPIRY_MINUTES * 60,
        settings=get_location_settings(),
    )


@app.route("/student", methods=["GET", "POST"])
def student():
    settings = get_location_settings()

    if request.method == "POST":
        token = request.form["token"]
        name = request.form["student_name"].strip()
        lat_raw = request.form.get("lat", "").strip()
        lon_raw = request.form.get("lon", "").strip()
        try:
            lat = float(lat_raw)
            lon = float(lon_raw)
        except ValueError:
            return render_template(
                "result.html",
                message="Location not available. Allow location permission and try again.",
                status="error",
            )
        uploaded_face = request.files.get("face_photo")
        captured_face_data = request.form.get("captured_face_data", "").strip()

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

        distance = calculate_distance(settings["class_lat"], settings["class_lon"], lat, lon)
        if distance > settings["allowed_radius"]:
            conn.close()
            return render_template(
                "result.html",
                message=f"Outside allowed campus area ({settings['campus_name']})",
                status="error",
            )

        if uploaded_face and uploaded_face.filename:
            captured_path, upload_error = save_uploaded_face(uploaded_face)
        else:
            captured_path, upload_error = save_captured_face_data(captured_face_data)
        if upload_error:
            conn.close()
            return render_template("result.html", message=upload_error, status="error")

        face_count = count_faces_in_image(captured_path)
        if face_count == 0:
            conn.close()
            safe_remove(captured_path)
            return render_template(
                "result.html",
                message="Face not detected clearly. Keep one face in frame and recapture.",
                status="error",
            )

        if face_count > 1:
            conn.close()
            safe_remove(captured_path)
            return render_template(
                "result.html",
                message="Multiple faces detected. Ensure only your face is visible.",
                status="error",
            )

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
            "INSERT INTO attendance (student_name, timestamp, session_token, subject_name) VALUES (?, ?, ?, ?)",
            (name, timestamp, token, settings["subject_name"]),
        )
        conn.commit()
        conn.close()
        safe_remove(captured_path)

        return render_template(
            "result.html",
            message=f"Attendance Marked Successfully for {settings['subject_name']}",
            status="success",
        )

    return render_template("student.html", settings=settings)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)

