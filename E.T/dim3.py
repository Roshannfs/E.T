import cv2
import imutils
from imutils import perspective
from imutils import contours
import numpy as np

def midpoint(x, y):
    return ((x[0] + y[0]) * 0.5, (x[1] + y[1]) * 0.5)

KNOWN_WIDTH = 8.5
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Cannot open camera")
    exit()

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
    (cnts, _) = contours.sort_contours(cnts)
    pixelsPerMetric = None

    largest_contour = None
    largest_area = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 100:
            continue
        if area > largest_area:
            largest_area = area
            largest_contour = c

    if largest_contour is not None:
        box = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
        cv2.drawContours(image, [box.astype("int")], -1, (0, 255, 0), 2)
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        dA = np.linalg.norm(np.array([tltrX, tltrY]) - np.array([blbrX, blbrY]))
        dB = np.linalg.norm(np.array([tlblX, tlblY]) - np.array([trbrX, trbrY]))

        if pixelsPerMetric is None:
            pixelsPerMetric = dB / KNOWN_WIDTH

        dimA = dB / pixelsPerMetric  # width in cm
        dimB = dA / pixelsPerMetric  # height in cm

        cv2.putText(image, "{:.1f}cm".format(dimB),
                    (int(tltrX - 10), int(tltrY - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(image, "{:.1f}cm".format(dimA),
                    (int(trbrX + 10), int(trbrY)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        if abs(dimA - dimB) < 0.15 * max(dimA, dimB):
            ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
            diameter = (2 * radius) / pixelsPerMetric
            cv2.circle(image, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.putText(image, "Diam: {:.2f}cm".format(diameter),
                        (int(x), int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Measurement", image)
    key = cv2.waitKey(1)
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()
