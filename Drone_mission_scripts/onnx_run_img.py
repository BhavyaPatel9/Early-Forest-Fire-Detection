import cv2

# Load the ONNX model
net = cv2.dnn.readNetFromONNX("/home/pi/Desktop/fire_model.onnx")

# Read image from file
image_path = "/home/pi/Desktop/pexels-pixabay-51951.jpg"  # <-- change this path to your image
frame = cv2.imread(image_path)

if frame is None:
    print("Error: Could not load image.")
    exit()

# Preprocess the image
blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255.0, size=(224, 224))

# Set input to the network
net.setInput(blob)

# Perform forward pass
output = net.forward()

# Print the output
print("Model Output:", output)


# Display the image
cv2.imshow("Input Image", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

