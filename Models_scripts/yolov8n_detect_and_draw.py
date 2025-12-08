#!/usr/bin/env python3
"""
yolov8n_detect_and_draw.py
- Decodes (1,6,N) YOLOv8 tflite outputs: [cx,cy,w,h, v5, v6] (v5/v6 are logits)
- Uses score = sigmoid(v5)*sigmoid(v6)
- Filters by CONF_THRESH, applies NMS, draws boxes, saves output.
"""
import tflite_runtime.interpreter as tflite
import numpy as np, math, cv2, time, argparse

# ---- USER CONFIG ----
MODEL = "/home/pi/Desktop/fire_detection/best_yolov8n_float16.tflite"
IMAGE = "/home/pi/Desktop/fire_detection/fire_detect.mp4"
OUT   = "/home/pi/Desktop/fire_detection/detected_result1.mp4"
IMG_SIZE = 320
CONF_THRESH = 0.32   # try 0.30-0.35 if needed
IOU_THRESH = 0.45
MAX_DET = 100
# ---------------------

def sigmoid(x): return 1.0/(1.0+math.exp(-x))

def letterbox(img, new_shape=(IMG_SIZE,IMG_SIZE), color=(114,114,114)):
    h0,w0 = img.shape[:2]
    nh, nw = new_shape
    r = min(nw / w0, nh / h0)
    new_unpad_w, new_unpad_h = int(round(w0 * r)), int(round(h0 * r))
    dw, dh = nw - new_unpad_w, nh - new_unpad_h
    dw /= 2; dh /= 2
    resized = cv2.resize(img, (new_unpad_w, new_unpad_h))
    top, bottom = int(round(dh-0.5)), int(round(dh+0.5))
    left, right = int(round(dw-0.5)), int(round(dw+0.5))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded, r, (dw, dh), (w0, h0)

def nms(boxes, scores, iou_thresh=0.45):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes); scores = np.array(scores)
    x1 = boxes[:,0]; y1 = boxes[:,1]; x2 = boxes[:,2]; y2 = boxes[:,3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1); h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-8)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return keep

def main(model_path=MODEL, image_path=IMAGE, out_path=OUT):
    interp = tflite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    in_d = interp.get_input_details()[0]
    out_ds = interp.get_output_details()

    print("Input:", in_d['shape'], in_d['dtype'])
    for i,od in enumerate(out_ds):
        print(f"Output[{i}] shape: {od['shape']} dtype: {od['dtype']}")

    img0 = cv2.imread(image_path)
    if img0 is None:
        raise FileNotFoundError("Image not found: "+image_path)

    img, ratio, (dw, dh), (orig_w, orig_h) = letterbox(img0, new_shape=(IMG_SIZE, IMG_SIZE))
    inp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    inp = np.expand_dims(inp, 0)
    # handle dtype
    if in_d['dtype'] == np.float16:
        inp = inp.astype(np.float16)
    elif in_d['dtype'] in (np.uint8, np.int8):
        scale, zp = in_d.get('quantization', (1.0,0))
        if scale == 0: scale = 1.0
        inp = (inp/scale + zp).astype(in_d['dtype'])
    else:
        inp = inp.astype(in_d['dtype'])

    interp.set_tensor(in_d['index'], inp)
    t0 = time.time(); interp.invoke(); t1 = time.time()
    print("Inference time: %.3f s" % (t1 - t0))

    # find the (1,6,N) output
    out_arr = None
    for od in out_ds:
        arr = interp.get_tensor(od['index'])
        if arr.ndim == 3 and arr.shape[1] == 6:
            out_arr = arr; break
    if out_arr is None:
        out_arr = interp.get_tensor(out_ds[0]['index'])
        print("Warning: used first output:", out_arr.shape)

    s = np.squeeze(out_arr)   # (6,N)
    props = s.T               # (N,6)

    boxes_norm = []; scores = []
    for p in props:
        cx, cy, w, h = float(p[0]), float(p[1]), float(p[2]), float(p[3])
        v5, v6 = float(p[4]), float(p[5])
        sc = sigmoid(v5) * sigmoid(v6)
        if sc < CONF_THRESH:
            continue
        # normalized xywh -> normalized xyxy
        x1n, y1n, x2n, y2n = cx - w/2, cy - h/2, cx + w/2, cy + h/2
        # to original pixels (undo letterbox)
        x1 = (x1n * IMG_SIZE - dw) / ratio
        x2 = (x2n * IMG_SIZE - dw) / ratio
        y1 = (y1n * IMG_SIZE - dh) / ratio
        y2 = (y2n * IMG_SIZE - dh) / ratio
        # clip
        x1 = max(0, min(orig_w-1, x1)); x2 = max(0, min(orig_w-1, x2))
        y1 = max(0, min(orig_h-1, y1)); y2 = max(0, min(orig_h-1, y2))
        boxes_norm.append([x1, y1, x2, y2]); scores.append(sc)

    print("Candidates after threshold:", len(scores))
    if len(scores) == 0:
        print("No detections above CONF_THRESH (try lowering to 0.25). Saving original image.")
        cv2.imwrite(out_path, img0); return

    keep = nms(boxes_norm, scores, IOU_THRESH)[:MAX_DET]
    print("Kept after NMS:", len(keep))

    out_img = img0.copy()
    for i in keep:
        x1,y1,x2,y2 = map(int, boxes_norm[i])
        sc = scores[i]
        cv2.rectangle(out_img, (x1,y1), (x2,y2), (0,0,255), 2)
        cv2.putText(out_img, f"fire {sc:.2f}", (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imwrite(out_path, out_img)
    print("Saved result to", out_path)

if __name__ == "__main__":
    main()
