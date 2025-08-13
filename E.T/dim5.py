import cv2
import imutils
import numpy as np

KNOWN_WIDTH = 13.3/404.08 # Reference object width in cm

def midpoint(x, y):
    return ((x[0] + y[0]) * 0.5, (x[1] + y[1]) * 0.5)

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
        cv2.fillPoly(overlay, [c], color=(0, 255, 0))
        alpha = 0.3
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

        # Measure the contour's dimensions
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0,255,0), 2)

        # Calibration with reference object
        if pixelsPerMetric is None:
            # Use width of first detected object as reference
            pixelsPerMetric = w * KNOWN_WIDTH
            pixelsPerMetrich = h * KNOWN_WIDTH
        # Convert pixels to cm
        width = pixelsPerMetric
        height = pixelsPerMetrich

        if(width==5.56 , height==12.54):
            print("it is correct dimention")
        else:
            print("prodect is defected")
            


        # Draw measurement lines
        # Height line (vertical)
        cv2.line(image, (x + w//2, y), (x + w//2, y + h), (255, 0, 0), 2)
        # Width line (horizontal)
        cv2.line(image, (x, y + h//2), (x + w, y + h//2), (0, 0, 255), 2)

        # Overlay dimension text
        cv2.putText(image, "{:.2f}cm".format(height),
                    (x + w//2 - 30, y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(image, "{:.2f}cm".format(width),
                    (x + w - 80, y + h//2 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        #if(width==5.56 , height==12.54):
        #    print("it is correct dimention")
        #else:
        #   print("prodect is defected")

    cv2.imshow("Measurement", image)
    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()
