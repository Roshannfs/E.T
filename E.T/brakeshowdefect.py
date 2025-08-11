import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('D:/git roshan/E.T/E.T/defect_detection_model.h5')  # Use your trained model filename

# Set image size same as training
img_height, img_width = 128, 128

# Class names in order as used during training
class_names = ['noal', 'Defects']

# Open webcam feed (0 is usually default webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Resize frame to model input shape
    img = cv2.resize(frame, (img_width, img_height))

    # Convert BGR (OpenCV default) to RGB to match training images
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize pixel values to [0,1]
    img = img.astype("float32") / 255.0

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    # Predict class (output between 0 and 1)
    pred = model.predict(img)[0][0]

    # Determine label based on threshold 0.5
    if pred < 0.5:
        label = class_names[0]  # "normal"
        color = (0, 255, 0)  # Green
    else:
        label = class_names[1]  # "Defects"
        color = (0, 0, 255)  # Red

    # Display label and prediction confidence
    label_text = f"{label}: {pred:.2f}"

    # Put label text on frame
    cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, color, 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('Defect Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release and close
cap.release()
cv2.destroyAllWindows()
