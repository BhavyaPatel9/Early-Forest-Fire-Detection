#!/usr/bin/env python3
"""
yolov8n_picam_live.py
- Real-time fire detection using Raspberry Pi Camera Module 3
- Decodes (1,6,N) YOLOv8 tflite outputs: [cx,cy,w,h, v5, v6]
- Uses score = sigmoid(v5)*sigmoid(v6)
- Filters by CONF_THRESH, applies NMS, displays live feed with boxes
"""
import tflite_runtime.interpreter as tflite
import numpy as np
import math
import cv2
import time
from picamera2 import Picamera2

# ---- USER CONFIG ----
MODEL = "/home/pi/Desktop/fire_detection/best_yolov8n_float16.tflite"
IMG_SIZE = 320
CONF_THRESH = 0.32
IOU_THRESH = 0.45
MAX_DET = 100
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480
# ---------------------

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
    """Process a single frame and return detections"""
    img, ratio, (dw, dh), (orig_w, orig_h) = letterbox(frame, new_shape=(IMG_SIZE, IMG_SIZE))
    inp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    inp = np.expand_dims(inp, 0)
    
    # Handle dtype
    if in_d['dtype'] == np.float16:
        inp = inp.astype(np.float16)
    elif in_d['dtype'] in (np.uint8, np.int8):
        scale, zp = in_d.get('quantization', (1.0, 0))
        if scale == 0:
            scale = 1.0
        inp = (inp / scale + zp).astype(in_d['dtype'])
    else:
        inp = inp.astype(in_d['dtype'])
    
    interp.set_tensor(in_d['index'], inp)
    interp.invoke()
    
    # Find the (1,6,N) output
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
        # Normalized xywh -> normalized xyxy
        x1n, y1n, x2n, y2n = cx - w/2, cy - h/2, cx + w/2, cy + h/2
        # To original pixels (undo letterbox)
        x1 = (x1n * IMG_SIZE - dw) / ratio
        x2 = (x2n * IMG_SIZE - dw) / ratio
        y1 = (y1n * IMG_SIZE - dh) / ratio
        y2 = (y2n * IMG_SIZE - dh) / ratio
        # Clip
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

def main():
    # Initialize TFLite model
    print("Loading model...")
    interp = tflite.Interpreter(model_path=MODEL)
    interp.allocate_tensors()
    in_d = interp.get_input_details()[0]
    out_ds = interp.get_output_details()
    
    print("Input:", in_d['shape'], in_d['dtype'])
    for i, od in enumerate(out_ds):
        print(f"Output[{i}] shape: {od['shape']} dtype: {od['dtype']}")
    
    # Initialize Picamera2
    print("Initializing camera...")
    picam2 = Picamera2()
    
    # Configure camera for video mode
    config = picam2.create_preview_configuration(
        main={"size": (DISPLAY_WIDTH, DISPLAY_HEIGHT), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    
    print("Camera started. Press 'q' to quit, 's' to save snapshot.")
    
    # FPS calculation
    fps_start_time = time.time()
    fps_counter = 0
    fps = 0
    
    try:
        while True:
            # Capture frame (already in RGB888 format)
            frame = picam2.capture_array()
            
            # No color conversion needed - frame is already in correct format
            frame_bgr = frame.copy()
            
            # Process frame
            t0 = time.time()
            detections = process_frame(interp, in_d, out_ds, frame_bgr)
            t1 = time.time()
            inference_time = (t1 - t0) * 1000  # Convert to ms
            
            # Draw detections
            for box, score in detections:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
                label = f"fire {score:.2f}"
                cv2.putText(frame_bgr, label, (x1, max(0, y1 - 6)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Calculate FPS
            fps_counter += 1
            if time.time() - fps_start_time >= 1.0:
                fps = fps_counter
                fps_counter = 0
                fps_start_time = time.time()
            
            # Display info
            info_text = f"FPS: {fps} | Inference: {inference_time:.1f}ms | Detections: {len(detections)}"
            cv2.putText(frame_bgr, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow("Fire Detection - Press 'q' to quit, 's' to save", frame_bgr)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"/home/pi/Desktop/fire_detection/snapshot_{timestamp}.jpg"
                cv2.imwrite(filename, frame_bgr)
                print(f"Saved snapshot to {filename}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        picam2.stop()
        cv2.destroyAllWindows()
        print("Camera stopped and windows closed.")

if __name__ == "__main__":
    main()
