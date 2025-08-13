import cv2
import numpy as np

# Open the default webcam
cap = cv2.VideoCapture(0)

# Storage for manually marked points
points = []

# Mouse callback function to record points
def mark_point(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point marked at: ({x}, {y})")

cv2.namedWindow('Webcam with Pixel Range')
cv2.setMouseCallback('Webcam with Pixel Range', mark_point)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Calculate min/max pixel values for each color channel
    ranges_text = []
    for i, color in enumerate(['Blue', 'Green', 'Red']):
        min_val = np.min(frame[:, :, i])
        max_val = np.max(frame[:, :, i])
        ranges_text.append(f"{color}: {min_val} - {max_val}")

    # Display the pixel ranges on the frame
    y0, dy = 30, 25
    for i, text in enumerate(ranges_text):
        y = y0 + i * dy
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Draw the marked points and their coordinates
    for point in points:
        cv2.circle(frame, point, 5, (0, 255, 0), -1)
        cv2.putText(frame, f"{point}", (point[0] + 10, point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Webcam with Pixel Range', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
