import cv2
import imutils
import numpy as np

# Reference object width in pixels for calibration (e.g., known object)
KNOWN_WIDTH_PIXELS = 404.08
KNOWN_WIDTH_CM = 13.3

# Target width and height in cm
TARGET_WIDTH_CM = 5.56
TARGET_HEIGHT_CM = 12.54
TOLERANCE = 0.2  # Acceptable deviation in cm

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
        return width_cm, height_cm, x, y, w, h
    return None, None, None, None, None, None

def calibrate_pixels_per_metric(image):
    # Find contours, use known reference object for calibration
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(blurred, 50, 100)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    for c in cnts:
        if cv2.contourArea(c) < 100:
            continue
        x, y, w, h = cv2.boundingRect(c)
        pixels_per_metric = w / KNOWN_WIDTH_CM
        return pixels_per_metric
    return KNOWN_WIDTH_PIXELS / KNOWN_WIDTH_CM

cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Cannot open camera")
    exit()

# Calibration Phase: User should show reference object first
print("Show the reference object for calibration and press any key")
while True:
    ret, frame = cam.read()
    image = imutils.resize(frame, width=600)
    cv2.imshow("Calibration", image)
    if cv2.waitKey(1) != -1:
        pixels_per_metric = calibrate_pixels_per_metric(image)
        break

# Measurement Phase: Iterate twice for front and then side
results = []
for view in ["Front", "Side"]:
    print(f"Show the {view} of the object and press any key")
    while True:
        ret, frame = cam.read()
        image = imutils.resize(frame, width=600)
        cv2.imshow(f"{view} View", image)
        if cv2.waitKey(1) != -1:
            width_cm, height_cm, x, y, w, h = measure_object(image, pixels_per_metric)
            color = (0,255,0) if check_dimension(width_cm, height_cm, TARGET_WIDTH_CM, TARGET_HEIGHT_CM) else (0,0,255)
            status = "Not Defected" if color==(0,255,0) else "Defected"
            # Overlay
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
