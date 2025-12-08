import time
import cv2
import numpy as np
from picamera2 import Picamera2
from PIL import Image
import tflite_runtime.interpreter as tflite

# Path to model
MODEL_PATH = "/home/pi/Desktop/fire_detection/mobilenetv2_fire_detection.tflite"

# Initialize TFLite interpreter
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Model loaded successfully!")
print("Input shape:", input_details[0]['shape'])

# Initialize Pi Camera
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(config)
picam2.start()
time.sleep(2)  # warm-up

print(" Camera started! Press 'q' to quit.")

# Function to preprocess frames
def preprocess(frame):
    img = Image.fromarray(frame).resize((224, 224))
    input_data = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)
    if input_details[0]['dtype'] == np.float16:
        input_data = input_data.astype(np.float16)
    return input_data

# Real-time detection loop
while True:
    frame = picam2.capture_array()
    input_data = preprocess(frame)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = float(output_data[0][0])

    # Display result on frame
    label = "Fire detected!" if prediction > 0.5 else "Safe"
    color = (0, 0, 255) if prediction > 0.5 else (0, 255, 0)

    cv2.putText(frame, label, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3, cv2.LINE_AA)
    cv2.imshow("Live Fire Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cv2.destroyAllWindows()
picam2.stop()
print("Exiting...")
