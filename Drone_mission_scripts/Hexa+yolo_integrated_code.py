#!/usr/bin/env python3
"""
drone_yolo_integration.py
- Integrates DroneKit hexagon mission with YOLOv8 TFLite detector using Picamera2.
- Starts the YOLO thread when first vertex is reached, keeps it running,
  and at each vertex requests 3 snapshots to be saved.
- Run this inside your virtualenv where tflite_runtime & picamera2 are installed.

Usage:
    source fire_env/bin/activate
    python3 drone_yolo_integration.py --connect /dev/ttyACM0
"""

import time
import math
import argparse
import threading
import queue
import os
import cv2
import numpy as np

# Drone libs
from dronekit import connect, VehicleMode, LocationGlobalRelative

# TFLite + Picamera2 libs (ensure installed inside your venv)
import tflite_runtime.interpreter as tflite
from picamera2 import Picamera2

# -----------------------------
# User-configurable parameters (edit these manually)
# -----------------------------
SIDE_LENGTH = 10.0   # meters (change manually)
TARGET_ALTITUDE = 15.0  # meters (change manually)
CENTER_LAT = 21.1600811   # centre latitude (change manually)
CENTER_LON = 72.7860258   # centre longitude (change manually)

# YOLO config
MODEL = "/home/pi/Desktop/fire_detection/best_yolov8n_float16.tflite"
IMG_SIZE = 320
CONF_THRESH = 0.32
IOU_THRESH = 0.45
MAX_DET = 100
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480

# Snapshots config
SNAPSHOTS_PER_POINT = 3
SNAPSHOT_DIR = "/home/pi/Desktop/fire_detection/snapshots"  # ensure exists or will be created
SNAPSHOT_DELAY = 0.6  # seconds between each snapshot

# Camera display (optional)
SHOW_DISPLAY = False  # set True if you have display connected and want to view frames

# -----------------------------
# Utility functions (from your YOLO script)
# -----------------------------
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
        self.snapshot_queue = snapshot_queue  # queue of (basefilename, count) requests
        self.stop_event = stop_event
        self.picam2 = None
        self.interp = None
        self.in_d = None
        self.out_ds = None
        self.frame = None  # last frame (BGR)
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
    
    def run(self):
        try:
            self.setup()
            fps_counter = 0
            fps_start = time.time()
            fps = 0
            while not self.stop_event.is_set():
                frame_rgb = self.picam2.capture_array()
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                self.frame = frame_bgr.copy()
                
                # Run detection (non-blocking-ish — this is blocking but fast on pi with tflite)
                t0 = time.time()
                detections = process_frame(self.interp, self.in_d, self.out_ds, frame_bgr)
                t1 = time.time()
                inference_ms = (t1 - t0) * 1000
                
                # Optional draw boxes
                for box, score in detections:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    label = f"fire {score:.2f}"
                    cv2.putText(frame_bgr, label, (x1, max(0, y1 - 6)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                fps_counter += 1
                if time.time() - fps_start >= 1.0:
                    fps = fps_counter
                    fps_counter = 0
                    fps_start = time.time()
                
                info_text = f"FPS: {fps} | Inference: {inference_ms:.1f}ms | Dets: {len(detections)}"
                cv2.putText(frame_bgr, info_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                
                if SHOW_DISPLAY:
                    cv2.imshow("YOLO Live", frame_bgr)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("[YOLO] Quit requested from display.")
                        self.stop_event.set()
                        break
                
                # Handle snapshot requests (non-blocking)
                try:
                    base_name, count = self.snapshot_queue.get_nowait()
                except queue.Empty:
                    base_name = None
                    count = 0
                
                if base_name is not None and count > 0:
                    # Save `count` snapshots spaced by SNAPSHOT_DELAY
                    for i in range(count):
                        if self.frame is None:
                            time.sleep(0.05)
                            continue
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        fname = f"{base_name}_{timestamp}_{i+1}.jpg"
                        fullpath = os.path.join(SNAPSHOT_DIR, fname)
                        cv2.imwrite(fullpath, self.frame)
                        print(f"[YOLO] Saved snapshot: {fullpath}")
                        time.sleep(SNAPSHOT_DELAY)
                
            # End loop
        except Exception as e:
            print("[YOLO] Exception in YOLO thread:", str(e))
        finally:
            print("[YOLO] Stopping camera and cleaning up...")
            try:
                if self.picam2:
                    self.picam2.stop()
            except Exception:
                pass
            if SHOW_DISPLAY:
                cv2.destroyAllWindows()
            print("[YOLO] Thread exiting.")


# -----------------------------
# Drone helper functions (from your drone script)
# -----------------------------
def arm_and_takeoff(vehicle, aTargetAltitude):
    """
    Arms vehicle and fly to target altitude.
    """
    print("Basic pre-arm checks...")
    while not vehicle.is_armable:
        print(" Waiting for vehicle to initialise...")
        time.sleep(1)

    print("Arming motors...")
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True

    while not vehicle.armed:
        print(" Waiting for arming...")
        time.sleep(1)

    print("Taking off!")
    vehicle.simple_takeoff(aTargetAltitude)

    while True:
        alt = vehicle.location.global_relative_frame.alt
        print(f" Altitude: {alt:.2f} m")
        if alt >= aTargetAltitude * 0.95:
            print("Reached target altitude")
            break
        time.sleep(1)

def get_location_metres(original_location, dNorth, dEast):
    """
    Returns a LocationGlobalRelative object moved by dNorth and dEast meters.
    """
    earth_radius = 6378137.0  # Radius of "spherical" earth
    dLat = dNorth / earth_radius
    dLon = dEast / (earth_radius * math.cos(math.pi * original_location.lat / 180))

    newlat = original_location.lat + (dLat * 180 / math.pi)
    newlon = original_location.lon + (dLon * 180 / math.pi)
    return LocationGlobalRelative(newlat, newlon, original_location.alt)

def hexagon_vertices_center(center_location, side_length):
    """
    Returns 6 vertices (LocationGlobalRelative) of a regular hexagon around center_location.
    side_length is the distance from centre to vertex (circumradius).
    """
    vertices = []
    for i in range(6):
        angle_deg = 60 * i
        angle_rad = math.radians(angle_deg)
        dNorth = side_length * math.cos(angle_rad)
        dEast = side_length * math.sin(angle_rad)
        new_point = get_location_metres(center_location, dNorth, dEast)
        vertices.append(new_point)
    return vertices

def goto_with_delay(vehicle, location, delay_time=5):
    print(f"Going to: Lat {location.lat:.6f}, Lon {location.lon:.6f}, Alt {location.alt}m")
    vehicle.simple_goto(location)
    # Wait loop until roughly at location or delay expires
    start = time.time()
    while time.time() - start < delay_time:
        # Optionally: add more robust proximity checks using distance
        time.sleep(0.5)

# -----------------------------
# Main integration
# -----------------------------
def ensure_snapshot_dir():
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--connect', default='/dev/ttyACM0', help='vehicle connection string')
    args = parser.parse_args()

    ensure_snapshot_dir()

    print("Connecting to vehicle on:", args.connect)
    vehicle = connect(args.connect, wait_ready=True)

    print(f"Taking off to {TARGET_ALTITUDE}m altitude")
    arm_and_takeoff(vehicle, TARGET_ALTITUDE)

    home_location = vehicle.location.global_frame
    print(f"Home set at: {home_location.lat:.6f}, {home_location.lon:.6f}")

    vehicle.airspeed = 3

    # Given target coordinate is treated as the CENTER of the hexagon (now editable)
    center = LocationGlobalRelative(CENTER_LAT, CENTER_LON, TARGET_ALTITUDE)
    print("Going to centre first")
    goto_with_delay(vehicle, center, delay_time=5)

    # Generate hexagon vertices (side = SIDE_LENGTH)
    vertices = hexagon_vertices_center(center, SIDE_LENGTH)

    # Setup YOLO thread objects but don't start until first vertex reached
    snapshot_q = queue.Queue()
    stop_event = threading.Event()
    yolo_thread = YOLORunner(MODEL, snapshot_q, stop_event)
    yolo_started = False

    print("Starting hexagon path around centre...")
    for i, vertex in enumerate(vertices):
        print(f"Visiting vertex {i+1} / 6")
        goto_with_delay(vehicle, vertex, delay_time=10)

        # When we reach the FIRST vertex, start YOLO thread (if not started)
        if not yolo_started:
            print("[Main] Starting YOLO thread now (first vertex reached).")
            yolo_thread.start()
            # Wait until YOLO camera & model ready
            yolo_thread.ready.wait(timeout=10)
            print("[Main] YOLO thread reported ready.")
            yolo_started = True

        # Request snapshots at this vertex
        basefile = f"vertex{i+1}"
        # Put a request: (basefilename, count)
        snapshot_q.put((basefile, SNAPSHOTS_PER_POINT))
        print(f"[Main] Requested {SNAPSHOTS_PER_POINT} snapshots at vertex {i+1} (queued).")

        # optional: small hover wait so snapshots can be captured
        time.sleep(2 + SNAPSHOTS_PER_POINT * SNAPSHOT_DELAY)

    # after visiting all vertices
    print("Hexagon completed. Requesting YOLO thread to stop after finishing queued snapshots.")
    # Give some time for remaining snapshot saves
    time.sleep(2)
    if yolo_started:
        stop_event.set()
        yolo_thread.join(timeout=10)
        print("[Main] YOLO thread stopped.")

    # Return to Launch (RTL)
    print("Returning to home (RTL)...")
    vehicle.mode = VehicleMode("RTL")

    # Wait until disarmed
    while vehicle.armed:
        print(" Waiting for drone to land and disarm...")
        time.sleep(2)

    print("Mission completed. Closing vehicle object.")
    vehicle.close()
    print("All done.")

if __name__ == "__main__":
    main()
