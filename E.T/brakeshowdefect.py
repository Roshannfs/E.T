import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('D:/git roshan/E.T/E.T/defect_detection_model.h5')

# Define class names
class_names = ['Normal', 'Defect']

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame for model prediction
    img = cv2.resize(frame, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    pred = model.predict(img)[0][0]
    label = class_names[int(pred > 0.5)]

    # Display prediction on live frame
    cv2.putText(frame, f"Prediction: {label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,200,0) if label == 'Normal' else (0,0,255), 2)

    cv2.imshow('Defect Detection (Press q to quit)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
