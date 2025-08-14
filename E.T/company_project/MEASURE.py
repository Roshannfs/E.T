import cv2
import imutils
from imutils import perspective
import numpy as np

def midpoint(x, y):
    return ((x[0] + y[0]) * 0.5, (x[1] + y[1]) * 0.5)

def check_contour_inside_reference(contour, reference_contour):
    """Check if the detected contour fits inside the reference contour"""
    try:
        # Get bounding rectangles for both contours
        x1, y1, w1, h1 = cv2.boundingRect(contour)
        x2, y2, w2, h2 = cv2.boundingRect(reference_contour)
        
        # Check if detected object's bounding box fits inside reference bounding box
        fits_inside = (x1 >= x2 and y1 >= y2 and 
                       x1 + w1 <= x2 + w2 and y1 + h1 <= y2 + h2)
        
        # Check if all points of detected contour are inside reference contour
        points_inside = True
        for point in contour.reshape(-1, 2):
            # Convert numpy integers to Python native floats
            point_tuple = (float(point[0]), float(point[1]))
            if cv2.pointPolygonTest(reference_contour, point_tuple, False) < 0:
                points_inside = False
                break
        
        return fits_inside and points_inside
    
    except Exception as e:
        print(f"Error in containment check: {e}")
        return False  # Default to false if error occurs

def new_func(image, overlay, alpha):
    blended = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    return blended

def add_reference_shape_from_image(reference_image_path):
    ref_img = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
    if ref_img is None:
        return None
    _, thresh = cv2.threshold(ref_img, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        return max(contours, key=cv2.contourArea)
    return None

KNOWN_WIDTH = 14 / 362  # Reference object width in cm
cam = cv2.VideoCapture(0)

# Set window to resizable and specify initial size
cv2.namedWindow("Measurement", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Measurement", 800, 600)  # Width, Height in pixels

if not cam.isOpened():
    print("Cannot open camera")
    exit()

pixelsPerMetric = None

# Load reference contour once outside the loop (optimization)
reference_image_path = "D:\git roshan\E.T\E.T\company_project\Removebackgroundproject.png"
reference_box = add_reference_shape_from_image(reference_image_path)

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

    # FILTER TO GET ONLY THE LARGEST CONTOUR (SINGLE OBJECT DETECTION)
    if len(cnts) > 0:
        # Find the largest contour by area
        largest_contour = max(cnts, key=cv2.contourArea)
        
        # Only process if the largest contour meets minimum area requirement
        if cv2.contourArea(largest_contour) >= 100:
            c = largest_contour  # Use only the largest contour
            
            if reference_box is None:
                continue  # Skip if no reference contour found

            # Scale reference contour to fit inside the display image
            ref_img = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
            if ref_img is None:
                continue  # Skip if reference image not found

            ref_h, ref_w = ref_img.shape
            img_h, img_w = image.shape[:2]
            scale_x = img_w / ref_w * 0.55  # display width
            scale_y = img_h / ref_h * 0.9# display height

            # FIX: Proper contour reshaping and scaling
            reference_box_reshaped = reference_box.reshape(-1, 2).astype("float")
            
            # Scale the coordinates
            reference_box_reshaped[:, 0] *= scale_x  # Scale x coordinates
            reference_box_reshaped[:, 1] *= scale_y  # Scale y coordinates

            # Center the reference box in the display image
            min_x, min_y = np.min(reference_box_reshaped, axis=0)
            max_x, max_y = np.max(reference_box_reshaped, axis=0)
            offset_x = (img_w - (max_x - min_x)) / 2 - min_x
            offset_y = (img_h - (max_y - min_y)) / 2 - min_y
            reference_box_reshaped[:, 0] += offset_x
            reference_box_reshaped[:, 1] += offset_y

            # Convert back to OpenCV contour format (n, 1, 2)
            reference_box_scaled = reference_box_reshaped.astype("int").reshape(-1, 1, 2)

            # Create overlay image for drawing
            overlay = image.copy()

            # Draw reference outline for comparison
            cv2.polylines(overlay, [reference_box_scaled], isClosed=True, color=(255, 255, 0), thickness=3)

            # Calculate bounding box for measurements
            box = cv2.minAreaRect(c)
            box_points = cv2.boxPoints(box)
            box_points = np.array(box_points, dtype="int")
            box_points = perspective.order_points(box_points)

            (tl, tr, br, bl) = box_points

            # Calculate midpoints for measurements
            (tltrX, tltrY) = midpoint(tl, tr)
            (blbrX, blbrY) = midpoint(bl, br)
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)

            dA = np.linalg.norm([tltrX - blbrX, tltrY - blbrY])  # height in pixels
            dB = np.linalg.norm([tlblX - trbrX, tlblY - trbrY])  # width in pixels

            if pixelsPerMetric is None:
                pixelsPerMetric = dB * KNOWN_WIDTH
                pixelsPerMetrich = dA * KNOWN_WIDTH
            else:
                pixelsPerMetrich = dA * KNOWN_WIDTH

            width = pixelsPerMetric
            height = pixelsPerMetrich

            # Check if detected object fits inside reference box
            fits_inside = check_contour_inside_reference(c, reference_box_scaled)
            
            # Create overlay with detected object
            cv2.fillPoly(overlay, [c], color=(0, 255, 0) if fits_inside else (0, 0, 255))
            
            # Determine dimension status based on containment
            if fits_inside:
                dimension_status = "Correct Dimension"
                status_color = (0, 128, 0)  # Green
                
                # Draw measurement lines only for correct dimensions
                cv2.line(overlay, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 0), 2)
                cv2.line(overlay, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (0, 0, 255), 2)
                
                # Show measurements
                cv2.putText(overlay, "{:.2f}cm".format(height),
                            (int(tltrX) - 40, int(tltrY)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cv2.putText(overlay, "{:.2f}cm".format(width),
                            (int(trbrX), int(trbrY) + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                dimension_status = "Not Correct Dimension"
                status_color = (139, 0, 0)  # Red

            # Show dimension status on image
            cv2.putText(overlay, dimension_status, (int(tl[0]), int(tl[1]) - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

            # Additional shape matching for defect detection
            #match = cv2.matchShapes(c, reference_box_scaled, cv2.CONTOURS_MATCH_I1, 0.0)
            #defect_status = "" if match > 0.1 else "OK"
            
            #cv2.putText(overlay, defect_status, (int(tl[0]), int(tl[1]) - 10),
                   # cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    #(0, 0, 255) if defect_status == "" else (0, 255, 0), 2)

            # Apply transparency (alpha blending)
            alpha = 0.6
            image = new_func(image, overlay, alpha)

            # Draw rectangle outline around detected object
            cv2.drawContours(image, [box_points.astype("int")], -1, status_color, 2)

    # Always draw reference box even if no object is detected
    if reference_box is not None:
        ref_img = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
        if ref_img is not None:
            ref_h, ref_w = ref_img.shape
            img_h, img_w = image.shape[:2]
            scale_x = img_w / ref_w * 0.55
            scale_y = img_h / ref_h * 0.9

            reference_box_reshaped = reference_box.reshape(-1, 2).astype("float")
            reference_box_reshaped[:, 0] *= scale_x
            reference_box_reshaped[:, 1] *= scale_y

            min_x, min_y = np.min(reference_box_reshaped, axis=0)
            max_x, max_y = np.max(reference_box_reshaped, axis=0)
            offset_x = (img_w - (max_x - min_x)) / 2 - min_x
            offset_y = (img_h - (max_y - min_y)) / 2 - min_y
            reference_box_reshaped[:, 0] += offset_x
            reference_box_reshaped[:, 1] += offset_y

            reference_box_scaled = reference_box_reshaped.astype("int").reshape(-1, 1, 2)
            cv2.polylines(image, [reference_box_scaled], isClosed=True, color=(255, 255, 0), thickness=2)

    # Show the result
    cv2.imshow("Measurement", image)
    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()
