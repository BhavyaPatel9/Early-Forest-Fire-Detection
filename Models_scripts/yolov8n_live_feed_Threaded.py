#!/usr/bin/env python3
"""
yolov8n_picam_live_threaded.py
- Threaded version of your YOLOv8 Picamera script.
- Camera capture + TFLite inference run inside YOLORunner thread.
- Main thread reads latest frame + detections for display & keyboard control.
- Snapshot requests are queued: put (basefilename, count) into snapshot_q.
"""
import tflite_runtime.interpreter as tflite
import numpy as np
import math
import cv2
import time
import threading
import queue
import os
from picamera2 import Picamera2

# ---- USER CONFIG ----
MODEL = "/home/pi/Desktop/fire_detection/best_yolov8n_float16.tflite"
IMG_SIZE = 320
CONF_THRESH = 0.32
IOU_THRESH = 0.45
MAX_DET = 100
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480

SNAPSHOT_DIR = "/home/pi/Desktop/fire_detection/snapshots"
SNAPSHOTS_PER_PRESS = 1
SNAPSHOT_DELAY = 0.6  # sec
SHOW_DISPLAY = True
# ---------------------

os.makedirs(SNAPSHOT_DIR, exist_ok=True)

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def letterbox(img, new_shape=(IMG_SIZE, IMG_SIZE), color=(114, 114, 114)):
    h0, w0 = img.shape[:2]
    nh, nw = new_shape
    r = min(nw / w0, nh / h0)
    new_unpad_w, new_unpad_h = int(round(w0 * r)), int(round(h0 * r))
    dw, dh = nw - new_unpad_w, nh - new_unpad_h
    dw /= 2
    dh /= 2
    resized = cv2.resize(img, (new_unpad_w, new_unpad_h))
    top, bottom = int(round(dh - 0.5)), int(round(dh + 0.5))
    left, right = int(round(dw - 0.5)), int(round(dw + 0.5))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=color)
    return padded, r, (dw, dh), (w0, h0)

def nms(boxes, scores, iou_thresh=0.45):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    scores = np.array(scores)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-8)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return keep

def process_frame(interp, in_d, out_ds, frame):
    """Process a single frame and return detections (boxes in orig frame coords)."""
    img, ratio, (dw, dh), (orig_w, orig_h) = letterbox(frame, new_shape=(IMG_SIZE, IMG_SIZE))
    inp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    inp = np.expand_dims(inp, 0)

    # Handle dtype for interpreter
    if in_d['dtype'] == np.float16:
        inp = inp.astype(np.float16)
    elif in_d['dtype'] in (np.uint8, np.int8):
        quant = in_d.get('quantization', (1.0, 0))
        if len(quant) >= 2:
            scale, zp = quant[0], int(quant[1])
        else:
            scale, zp = 1.0, 0
        if scale == 0:
            scale = 1.0
        inp = (inp / scale + zp).astype(in_d['dtype'])
    else:
        inp = inp.astype(in_d['dtype'])

    interp.set_tensor(in_d['index'], inp)
    interp.invoke()

    # Find the (1,6,N) output if present
    out_arr = None
    for od in out_ds:
        arr = interp.get_tensor(od['index'])
        if arr.ndim == 3 and arr.shape[1] == 6:
            out_arr = arr
            break
    if out_arr is None:
        out_arr = interp.get_tensor(out_ds[0]['index'])

    s = np.squeeze(out_arr)  # (6,N)
    props = s.T  # (N,6)

    boxes_norm = []
    scores = []
    for p in props:
        cx, cy, w, h = float(p[0]), float(p[1]), float(p[2]), float(p[3])
        v5, v6 = float(p[4]), float(p[5])
        sc = sigmoid(v5) * sigmoid(v6)
        if sc < CONF_THRESH:
            continue
        x1n, y1n, x2n, y2n = cx - w/2, cy - h/2, cx + w/2, cy + h/2
        x1 = (x1n * IMG_SIZE - dw) / ratio
        x2 = (x2n * IMG_SIZE - dw) / ratio
        y1 = (y1n * IMG_SIZE - dh) / ratio
        y2 = (y2n * IMG_SIZE - dh) / ratio
        x1 = max(0, min(orig_w - 1, x1))
        x2 = max(0, min(orig_w - 1, x2))
        y1 = max(0, min(orig_h - 1, y1))
        y2 = max(0, min(orig_h - 1, y2))
        boxes_norm.append([x1, y1, x2, y2])
        scores.append(sc)

    if len(scores) == 0:
        return []

    keep = nms(boxes_norm, scores, IOU_THRESH)[:MAX_DET]
    detections = [(boxes_norm[i], scores[i]) for i in keep]
    return detections

# -----------------------------
# YOLO runner thread
# -----------------------------
class YOLORunner(threading.Thread):
    def __init__(self, model_path, snapshot_queue, stop_event):
        super().__init__(daemon=True)
        self.model_path = model_path
        self.snapshot_queue = snapshot_queue  # queue of (basefilename, count)
        self.stop_event = stop_event
        self.picam2 = None
        self.interp = None
        self.in_d = None
        self.out_ds = None

        # Shared state (protected by lock)
        self._lock = threading.Lock()
        self._frame = None       # BGR frame (latest)
        self._detections = []    # latest detections
        self._inference_ms = 0.0
        self.ready = threading.Event()

    def setup(self):
        # Load TFLite model
        print("[YOLO] Loading model:", self.model_path)
        self.interp = tflite.Interpreter(model_path=self.model_path)
        self.interp.allocate_tensors()
        self.in_d = self.interp.get_input_details()[0]
        self.out_ds = self.interp.get_output_details()
        print("[YOLO] Model input:", self.in_d['shape'], self.in_d['dtype'])
        for i, od in enumerate(self.out_ds):
            print(f"[YOLO] Output[{i}] shape: {od['shape']} dtype: {od['dtype']}")

        # Setup camera
        print("[YOLO] Initializing Picamera2...")
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            main={"size": (DISPLAY_WIDTH, DISPLAY_HEIGHT), "format": "RGB888"}
        )
        self.picam2.configure(config)
        self.picam2.start()
        print("[YOLO] Camera started.")
        self.ready.set()

    def get_state(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None, list(self._detections), self._inference_ms

    def run(self):
        try:
            self.setup()
            fps_counter = 0
            fps_start = time.time()
            fps = 0
            while not self.stop_event.is_set():
                frame_rgb = self.picam2.capture_array()
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                t0 = time.time()
                detections = process_frame(self.interp, self.in_d, self.out_ds, frame_bgr)
                t1 = time.time()
                inference_ms = (t1 - t0) * 1000.0

                # Save to shared state
                with self._lock:
                    self._frame = frame_bgr.copy()
                    self._detections = detections
                    self._inference_ms = inference_ms

                # Handle snapshot requests (non-blocking)
                try:
                    base_name, count = self.snapshot_queue.get_nowait()
                except queue.Empty:
                    base_name = None
                    count = 0

                if base_name is not None and count > 0:
                    # Save `count` snapshots spaced by SNAPSHOT_DELAY
                    for i in range(count):
                        with self._lock:
                            save_frame = self._frame.copy() if self._frame is not None else None
                        if save_frame is None:
                            time.sleep(0.05)
                            continue
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        fname = f"{base_name}_{timestamp}_{i+1}.jpg"
                        fullpath = os.path.join(SNAPSHOT_DIR, fname)
                        cv2.imwrite(fullpath, save_frame)
                        print(f"[YOLO] Saved snapshot: {fullpath}")
                        # small delay between snapshots
                        time.sleep(SNAPSHOT_DELAY)

                # small sleep to prevent 100% busy loop (tune if needed)
                # but keep it small so camera capture remains high-rate
                time.sleep(0.001)

        except Exception as e:
            print("[YOLO] Exception in YOLO thread:", str(e))
        finally:
            print("[YOLO] Stopping camera and cleaning up...")
            try:
                if self.picam2:
                    self.picam2.stop()
            except Exception:
                pass
            print("[YOLO] Thread exiting.")

# -----------------------------
# Main (UI) thread
# -----------------------------
def main():
    print("Loading model (main thread will only start runner)...")
    # Pre-check: make sure model exists
    if not os.path.isfile(MODEL):
        print("Model file not found:", MODEL)
        return

    snapshot_q = queue.Queue()
    stop_event = threading.Event()
    yolo_runner = YOLORunner(MODEL, snapshot_q, stop_event)

    print("[Main] Starting YOLO runner thread...")
    yolo_runner.start()
    # Wait for camera/model ready
    if not yolo_runner.ready.wait(timeout=10):
        print("[Main] YOLO thread did not become ready in 10s — aborting.")
        stop_event.set()
        yolo_runner.join(timeout=2)
        return
    print("[Main] YOLO thread is ready. Showing display window...")

    fps_counter = 0
    fps = 0
    fps_start = time.time()

    try:
        while True:
            state = yolo_runner.get_state()
            frame, detections, inference_ms = state
            if frame is None:
                # no frame yet
                time.sleep(0.01)
                continue

            display = frame.copy()
            # Draw detections
            for box, score in detections:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 2)
                label = f"fire {score:.2f}"
                cv2.putText(display, label, (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # FPS calc (display thread)
            fps_counter += 1
            if time.time() - fps_start >= 1.0:
                fps = fps_counter
                fps_counter = 0
                fps_start = time.time()

            info_text = f"FPS: {fps} | Inference: {inference_ms:.1f}ms | Detections: {len(detections)}"
            cv2.putText(display, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if SHOW_DISPLAY:
                cv2.imshow("Fire Detection - Press 'q' to quit, 's' to save", display)
                key = cv2.waitKey(1) & 0xFF
            else:
                # When no display attached, poll keyboard less aggressively
                key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("[Main] Quit requested.")
                break
            elif key == ord('s'):
                # Request snapshot(s)
                base = f"snapshot"
                snapshot_q.put((base, SNAPSHOTS_PER_PRESS))
                print(f"[Main] Snapshot requested: {SNAPSHOTS_PER_PRESS} saved to {SNAPSHOT_DIR}")

            # tiny sleep to yield
            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user")
    finally:
        print("[Main] Stopping YOLO thread...")
        stop_event.set()
        yolo_runner.join(timeout=5)
        if SHOW_DISPLAY:
            cv2.destroyAllWindows()
        print("[Main] Exiting.")

if __name__ == "__main__":
    main()
