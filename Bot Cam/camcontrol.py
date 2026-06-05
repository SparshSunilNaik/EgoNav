import cv2
import threading
import time
import numpy as np
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

# ---- CONFIG ----

# Physical mapping (IMPORTANT: you control this)
CAMERA_MAP = {
    "left": 1,     # cam0 physically → LEFT
    "center": 2,   # cam1 physically → CENTER
    "right": 0     # cam2 physically → RIGHT
}

WIDTH = 160
HEIGHT = 120
FPS = 10
JPEG_QUALITY = 40

frames = {key: None for key in CAMERA_MAP.keys()}
frames_lock = threading.Lock()


# ---- CAMERA THREAD ----
class CameraThread(threading.Thread):
    def __init__(self, name, cam_id):
        super().__init__()
        self.name = name
        self.cam_id = cam_id
        self.cap = None
        self.running = True

    def open_camera(self):
        cap = cv2.VideoCapture(self.cam_id)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        return cap

    def run(self):
        print(f"[INFO] {self.name.upper()} camera started (ID={self.cam_id})")
        self.cap = self.open_camera()
        consecutive_fails = 0

        while self.running:
            if self.cap is None or not self.cap.isOpened():
                print(f"[WARN] Reopening {self.name}")
                with frames_lock:
                    frames[self.name] = None  # mark as dead
                time.sleep(1)
                self.cap = self.open_camera()
                consecutive_fails = 0
                continue

            ret, frame = self.cap.read()

            if ret and frame is not None:
                with frames_lock:
                    frames[self.name] = frame
                consecutive_fails = 0
            else:
                consecutive_fails += 1
                # Immediately mark as failed so stale frame isn't served
                with frames_lock:
                    frames[self.name] = None
                if consecutive_fails == 1:
                    print(f"[WARN] {self.name} read failed")
                if consecutive_fails >= 10:
                    print(f"[WARN] {self.name} failed {consecutive_fails}x → reopening")
                    self.cap.release()
                    time.sleep(1)
                    self.cap = self.open_camera()
                    consecutive_fails = 0

            time.sleep(1.0 / FPS)

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()


camera_threads = []


# ---- STARTUP ----
@app.on_event("startup")
def start_cameras():
    print("[INFO] Starting cameras...")

    for name, cam_id in CAMERA_MAP.items():
        t = CameraThread(name, cam_id)
        t.start()
        camera_threads.append(t)

    print("[INFO] Cameras running")


@app.on_event("shutdown")
def stop_cameras():
    print("[INFO] Stopping cameras...")
    for t in camera_threads:
        t.stop()


# ---- STREAM GENERATOR ----
# Black frame for dead cameras
_black_frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
_, _black_jpg = cv2.imencode('.jpg', _black_frame, [cv2.IMWRITE_JPEG_QUALITY, 20])
BLACK_BYTES = _black_jpg.tobytes()


def generate_stream(name):
    while True:
        with frames_lock:
            frame = frames.get(name)

        if frame is None:
            # Camera is dead → serve a black frame
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + BLACK_BYTES + b'\r\n'
            )
            time.sleep(1.0 / FPS)
            continue

        ret, buffer = cv2.imencode(
            '.jpg',
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        )

        if not ret:
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + BLACK_BYTES + b'\r\n'
            )
            time.sleep(1.0 / FPS)
            continue

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
        )

        time.sleep(1.0 / FPS)


# ---- API ENDPOINTS ----
@app.get("/left")
def stream_left():
    return StreamingResponse(generate_stream("left"),
        media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/center")
def stream_center():
    return StreamingResponse(generate_stream("center"),
        media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/right")
def stream_right():
    return StreamingResponse(generate_stream("right"),
        media_type="multipart/x-mixed-replace; boundary=frame")