import cv2
import imutils
import numpy as np

# Reference values
KNOWN_WIDTH_PIXELS = 404.08
KNOWN_WIDTH_CM = 13.3
TARGET_WIDTH_CM = 5.56
TARGET_HEIGHT_CM = 12.54
TOLERANCE = 0.2

def midpoint(x, y):
    return ((x[0] + y[0]) * 0.5, (x[1] + y[1]) * 0.5)

def check_dimension(width_cm, height_cm, target_w, target_h, tolerance=TOLERANCE):
    return abs(width_cm - target_w) <= tolerance and abs(height_cm - target_h) <= tolerance

def measure_object(image, pixels_per_metric):
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
        x, y, w, h = cv2.boundingRect(c)
        width_cm = w / pixels_per_metric
        height_cm = h / pixels_per_metric
        return width_cm, height_cm, x, y, w, h, c
    return None, None, None, None, None, None, None

def calibrate_pixels_per_metric(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(blurred, 50, 100)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    for c in cnts:
        if cv2.contourArea(c) < 100:
            continue
        x, y, w, h = cv2.boundingRect(c)
        return w / KNOWN_WIDTH_CM, x, y, w, h, c
    return KNOWN_WIDTH_PIXELS / KNOWN_WIDTH_CM, None, None, None, None, None

def shade_object(image, x, y, w, h, contour=None, alpha=0.3):
    overlay = image.copy()
    # Orange color (BGR): (0, 165, 255)
    if contour is not None:
        cv2.drawContours(overlay, [contour], -1, (0, 165, 255), thickness=cv2.FILLED)
    else:
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 165, 255), thickness=cv2.FILLED)
    # Blend overlay
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Cannot open camera")
    exit()

print("Show the reference object for calibration and press any key")
while True:
    ret, frame = cam.read()
    image = imutils.resize(frame, width=600)
    # Attempt to find object for shading
    ppm, x, y, w, h, c = calibrate_pixels_per_metric(image)
    if x is not None:
        image = shade_object(image, x, y, w, h, c)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("Calibration", image)
    if cv2.waitKey(1) != -1 and x is not None:
        pixels_per_metric = ppm
        break

results = []
for view in ["Front", "Side"]:
    print(f"Show the {view} of the object and press any key")
    while True:
        ret, frame = cam.read()
        image = imutils.resize(frame, width=600)
        width_cm, height_cm, x, y, w, h, c = measure_object(image, pixels_per_metric)
        if x is not None:
            image = shade_object(image, x, y, w, h, c)
            color = (0,255,0) if check_dimension(width_cm, height_cm, TARGET_WIDTH_CM, TARGET_HEIGHT_CM) else (0,0,255)
            status = "Not Defected" if color==(0,255,0) else "Defected"
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, f"W:{width_cm:.2f}cm H:{height_cm:.2f}cm", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(image, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow(f"{view} View", image)
        if cv2.waitKey(1) != -1 and x is not None:
            width_cm, height_cm, x, y, w, h, c = measure_object(image, pixels_per_metric)
            color = (0,255,0) if check_dimension(width_cm, height_cm, TARGET_WIDTH_CM, TARGET_HEIGHT_CM) else (0,0,255)
            status = "Not Defected" if color==(0,255,0) else "Defected"
            image = shade_object(image, x, y, w, h, c)
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
            cv2.putText(image, f"W:{width_cm:.2f}cm H:{height_cm:.2f}cm", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(image, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.imshow(f"{view} Result", image)
            cv2.waitKey(2000)
            results.append(status)
            break

cam.release()
cv2.destroyAllWindows()
