import cv2
import imutils
from imutils import perspective
import numpy as np

def midpoint(x, y):
    return ((x[0] + y[0]) * 0.5, (x[1] + y[1]) * 0.5)

KNOWN_WIDTH = 8.5  # Reference object width in cm
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Cannot open camera")
    exit()

pixelsPerMetric = None

while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame")
        break

    image = imutils.resize(frame, width=600)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(blurred, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    for c in cnts:
        if cv2.contourArea(c) < 100:
            continue

        # Create overlay for transparency
        overlay = image.copy()
        
        # Shade the actual object shape (contour) with light transparent color
        cv2.fillPoly(overlay, [c], color=(0, 255, 0))  # Light green fill
        
        # Apply transparency (alpha blending)
        alpha = 0.3  # Transparency level (0.0 = fully transparent, 1.0 = fully opaque)
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

        box = cv2.minAreaRect(c)
        box_points = cv2.boxPoints(box)
        box_points = np.array(box_points, dtype="int")
        box_points = perspective.order_points(box_points)

        # Draw rectangle outline
        cv2.drawContours(image, [box_points.astype("int")], -1, (0, 255, 0), 2)
            
        (tl, tr, br, bl) = box_points
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        dA = np.linalg.norm([tltrX - blbrX, tltrY - blbrY])  # height in pixels
        dB = np.linalg.norm([tlblX - trbrX, tlblY - trbrY])  # width in pixels

        if pixelsPerMetric is None:
            pixelsPerMetric = dB / KNOWN_WIDTH

        width = dB / pixelsPerMetric
        height = dA / pixelsPerMetric

        # Draw measurement lines
        # Height line (vertical)
        cv2.line(image, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 0), 2)
        # Width line (horizontal)
        cv2.line(image, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (0, 0, 255), 2)

        # Overlay measurements
        cv2.putText(image, "{:.2f}cm".format(height),
                    (int(tltrX) - 40, int(tltrY)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(image, "{:.2f}cm".format(width),
                    (int(trbrX), int(trbrY) + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Measurement", image)
    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()
