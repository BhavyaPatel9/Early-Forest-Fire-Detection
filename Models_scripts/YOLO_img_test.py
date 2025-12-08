import tflite_runtime.interpreter as tflite
import numpy as np
from PIL import Image

# Load the model
interpreter = tflite.Interpreter(model_path="/home/pi/Desktop/fire_detection/best_yolov8n_float32.tflite")
interpreter.allocate_tensors()

# Get details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input shape:", input_details[0]['shape'])
print("Input type:", input_details[0]['dtype'])
print("Output shape:", output_details[0]['shape'])
print("Output type:", output_details[0]['dtype'])

# Load and preprocess an image
img = Image.open("/home/pi/Desktop/fire_detection/photo/f3.jpg").resize((224, 224))
input_data = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)

if input_details[0]['dtype'] == np.float16:
    input_data = input_data.astype(np.float16)

# Run inference
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# Get results
output_data = interpreter.get_tensor(output_details[0]['index'])
print("Model output:", output_data)

if output_data[0][0] > 0.5:
    print("🔥 Fire detected!")
else:
    print("✅ No fire detected.")
