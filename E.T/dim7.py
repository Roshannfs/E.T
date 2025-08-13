import cv2
import imutils
import numpy as np
from scipy.spatial import distance as dist

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

def find_rectangles_and_measure(image, pixels_per_metric):
    """Find rectangular regions and measure objects within them"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(blurred, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    
    # Find contours
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    measurements = []
    
    for c in cnts:
        # Filter by area to find significant contours
        if cv2.contourArea(c) < 500:
            continue
            
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(c)
        
        # Check if this looks like a rectangular region (aspect ratio and area)
        aspect_ratio = w / float(h)
        
        # Look for rectangular shapes that could contain objects
        if 0.5 < aspect_ratio < 3.0 and w > 50 and h > 50:
            # Create ROI (Region of Interest) within this rectangle
            roi = image[y:y+h, x:x+w]
            
            # Find the actual object within this ROI
            object_width, object_height = measure_object_in_roi(roi, pixels_per_metric)
            
            if object_width is not None and object_height is not None:
                measurements.append({
                    'x': x, 'y': y, 'w': w, 'h': h,
                    'object_width_cm': object_width,
                    'object_height_cm': object_height,
                    'roi_x': x, 'roi_y': y
                })
    
    return measurements

def measure_object_in_roi(roi, pixels_per_metric):
    """Measure the actual object within a rectangular ROI"""
    if roi.size == 0:
        return None, None
        
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)
    
    # Use adaptive thresholding to better separate objects from background
    thresh = cv2.adaptiveThreshold(blurred_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find contours of objects within the ROI
    cnts_roi = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts_roi = imutils.grab_contours(cnts_roi)
    
    if len(cnts_roi) > 0:
        # Find the largest contour (assuming it's the main object)
        largest_contour = max(cnts_roi, key=cv2.contourArea)
        
        if cv2.contourArea(largest_contour) > 100:
            # Get the rotated rectangle for more accurate measurement
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            box = np.array(box, dtype="int")
            
            # Calculate width and height from the rotated rectangle
            (tl, tr, br, bl) = box
            
            # Compute the width and height of the object
            width_pixels = max(dist.euclidean(tl, tr), dist.euclidean(bl, br))
            height_pixels = max(dist.euclidean(tl, bl), dist.euclidean(tr, br))
            
            # Convert to centimeters
            width_cm = width_pixels / pixels_per_metric
            height_cm = height_pixels / pixels_per_metric
            
            return width_cm, height_cm
    
    return None, None

def calibrate_pixels_per_metric(image):
    """Calibrate using reference object"""
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

def draw_measurements(image, measurements):
    """Draw measurements and annotations on the image"""
    result_image = image.copy()
    
    for measurement in measurements:
        x, y, w, h = measurement['x'], measurement['y'], measurement['w'], measurement['h']
        width_cm = measurement['object_width_cm']
        height_cm = measurement['object_height_cm']
        
        # Check if object meets specifications
        is_good = check_dimension(width_cm, height_cm, TARGET_WIDTH_CM, TARGET_HEIGHT_CM)
        color = (0, 255, 0) if is_good else (0, 0, 255)  # Green if good, Red if defective
        status = "Not Defected" if is_good else "Defected"
        
        # Draw rectangle around the ROI
        cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
        
        # Add measurement text
        text_y = y - 10 if y > 30 else y + h + 25
        cv2.putText(result_image, f"W: {width_cm:.2f}cm", (x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(result_image, f"H: {height_cm:.2f}cm", (x, text_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(result_image, status, (x, text_y + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return result_image

# Main execution
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
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, f"Reference: {KNOWN_WIDTH_CM}cm", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
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
        
        # Find rectangles and measure objects within them
        measurements = find_rectangles_and_measure(image, pixels_per_metric)
        
        if measurements:
            # Draw measurements on image
            image = draw_measurements(image, measurements)
            
        cv2.imshow(f"{view} View", image)
        
        if cv2.waitKey(1) != -1 and measurements:
            # Process results for this view
            for measurement in measurements:
                width_cm = measurement['object_width_cm']
                height_cm = measurement['object_height_cm']
                is_good = check_dimension(width_cm, height_cm, TARGET_WIDTH_CM, TARGET_HEIGHT_CM)
                status = "Not Defected" if is_good else "Defected"
                results.append(status)
                print(f"{view} - Width: {width_cm:.2f}cm, Height: {height_cm:.2f}cm - {status}")
            
            cv2.imshow(f"{view} Result", image)
            cv2.waitKey(2000)
            break

# Final results
print("\nFinal Results:")
for i, result in enumerate(results):
    print(f"Measurement {i+1}: {result}")

overall_status = "PASS" if all(r == "Not Defected" for r in results) else "FAIL"
print(f"Overall Status: {overall_status}")

cam.release()
cv2.destroyAllWindows()
