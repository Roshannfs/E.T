import cv2
import numpy as np
import math
from scipy import spatial

class ObjectMeasurement:
    def __init__(self, reference_width_cm=21.0, reference_height_cm=29.7):
        """
        Initialize the measurement system
        reference_width_cm: Width of A4 paper in cm (default reference object)
        reference_height_cm: Height of A4 paper in cm
        """
        self.reference_width_cm = reference_width_cm
        self.reference_height_cm = reference_height_cm
        self.pixels_per_cm = None
        self.reference_detected = False
        
    def preprocess_image(self, image):
        """Preprocess the image for better contour detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # Morphological operations to clean up the image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        return thresh
    
    def get_contours(self, image, min_area=1000, filter_corners=4):
        """Find and filter contours based on area and corner count"""
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                # Approximate contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Filter by number of corners if specified
                if filter_corners == 0 or len(approx) == filter_corners:
                    filtered_contours.append((contour, approx, area))
        
        # Sort by area (largest first)
        filtered_contours.sort(key=lambda x: x[2], reverse=True)
        return filtered_contours
    
    def order_points(self, pts):
        """Order points in clockwise order: top-left, top-right, bottom-right, bottom-left"""
        xSorted = pts[np.argsort(pts[:, 0]), :]
        
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]
        
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost
        
        D = spatial.distance.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
        (br, tr) = rightMost[np.argsort(D)[::-1], :]
        
        return np.array([tl, tr, br, bl], dtype="float32")
    
    def calculate_distance(self, p1, p2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
    
    def detect_reference_object(self, image, contours):
        """Detect reference object (A4 paper) and calculate pixels per cm"""
        for contour, approx, area in contours:
            if len(approx) == 4:  # Rectangle/square
                # Order the points
                rect = self.order_points(approx.reshape(4, 2))
                
                # Calculate dimensions in pixels
                width_px = max(
                    self.calculate_distance(rect[0], rect[1]),
                    self.calculate_distance(rect[2], rect[3])
                )
                height_px = max(
                    self.calculate_distance(rect[0], rect[3]),
                    self.calculate_distance(rect[1], rect[2])
                )
                
                
                    
                    # Draw reference object
                cv2.drawContours(image, [approx], -1, (0, 255, 0), 3)
                cv2.putText(image, "Reference A4", 
                              (int(rect[0][0]), int(rect[0][1]) - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                return True
        return False
    
    def measure_object_dimensions(self, image, contours, skip_first=True):
        """Measure dimensions of objects (excluding reference object)"""
        measurements = []
        start_index = 1 if skip_first else 0
        
        for i, (contour, approx, area) in enumerate(contours[start_index:], start_index):
            if not self.reference_detected:
                continue
                
            # Get bounding rectangle
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            # Calculate dimensions
            width_px = rect[1][0]
            height_px = rect[1][1]
            
            # Convert to centimeters
            width_cm = width_px / self.pixels_per_cm
            height_cm = height_px / self.pixels_per_cm
            
            # Draw bounding box and measurements
            cv2.drawContours(image, [box], 0, (255, 0, 0), 2)
            
            # Calculate center for text placement
            center_x = int(rect[0][0])
            center_y = int(rect[0][1])
            
            # Display measurements
            cv2.putText(image, f"W: {width_cm:.1f}cm", 
                       (center_x - 50, center_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(image, f"H: {height_cm:.1f}cm", 
                       (center_x - 50, center_y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            measurements.append({
                'width_cm': width_cm,
                'height_cm': height_cm,
                'area_cm2': (width_cm * height_cm),
                'center': (center_x, center_y)
            })
            
        return measurements
    
    def process_frame(self, frame):
        """Process a single frame for object measurement"""
        # Create a copy for processingimport cv2
import numpy as np
from cv2 import aruco

class ArUcoMeasurementSystem:
    def __init__(self, marker_size_cm=5.0):
        """
        Initialize with ArUco marker as reference
        marker_size_cm: Real-world size of ArUco marker in cm
        """
        self.marker_size_cm = marker_size_cm
        self.pixels_per_cm = None
        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        self.aruco_params = aruco.DetectorParameters_create()
        
    def detect_aruco_reference(self, image):
        """Detect ArUco marker and calculate pixels per cm"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        
        if ids is not None and len(corners) > 0:
            # Use first detected marker
            corner = corners[0][0]
            
            # Calculate marker size in pixels
            side1 = np.linalg.norm(corner[0] - corner[1])
            side2 = np.linalg.norm(corner[1] - corner[2])
            marker_size_pixels = (side1 + side2) / 2
            
            # Calculate pixels per cm
            self.pixels_per_cm = marker_size_pixels / self.marker_size_cm
            
            # Draw detected marker
            aruco.drawDetectedMarkers(image, corners, ids)
            cv2.putText(image, f"Marker Size: {self.marker_size_cm}cm", 
                       (int(corner[0][0]), int(corner[0][1]) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            return True
        return False
    
    def measure_objects_with_aruco(self, image):
        """Measure objects using ArUco marker as reference"""
        if not self.detect_aruco_reference(image):
            return image, []
        
        # Convert to grayscale and find contours
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        measurements = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Filter small contours
                # Get minimum area rectangle
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                
                # Calculate dimensions in pixels and convert to cm
                width_px, height_px = rect[1]
                width_cm = width_px / self.pixels_per_cm
                height_cm = height_px / self.pixels_per_cm
                
                # Draw measurement
                cv2.drawContours(image, [box], 0, (255, 0, 0), 2)
                center = tuple(map(int, rect[0]))
                cv2.putText(image, f"{width_cm:.1f}x{height_cm:.1f}cm", 
                           (center[0]-40, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                
                measurements.append({'width_cm': width_cm, 'height_cm': height_cm})
        
        return image, measurements

        processed_frame = frame.copy()
        
        # Preprocess image
        thresh = self.preprocess_image(frame)
        
        # Find contours
        contours = self.get_contours(thresh, min_area=1000, filter_corners=0)
        
        if not contours:
            return processed_frame, []
        
        # Detect reference object if not already detected
        if not self.reference_detected:
            self.detect_reference_object(processed_frame, contours)
        
        # Measure object dimensions
        measurements = self.measure_object_dimensions(processed_frame, contours)
        
        # Display status
        status_text = "Reference: Detected" if self.reference_detected else "Reference: Place A4 paper"
        cv2.putText(processed_frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if self.reference_detected else (0, 0, 255), 2)
        
        return processed_frame, measurements

def main():
    """Main function to run the object measurement system"""
    # Initialize measurement system
    measurement_system = ObjectMeasurement()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Set camera properties for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.6)
    
    print("Object Dimension Measurement System")
    print("Instructions:")
    print("1. Place an A4 paper in the frame as reference")
    print("2. Place objects (like brake pads) to measure")
    print("3. Press 'r' to reset reference")
    print("4. Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        processed_frame, measurements = measurement_system.process_frame(frame)
        
        # Display results
        cv2.imshow('Object Dimension Measurement', processed_frame)
        
        # Print measurements to console
        if measurements:
            print("\n--- Current Measurements ---")
            for i, measurement in enumerate(measurements):
                print(f"Object {i+1}: {measurement['width_cm']:.1f}cm x {measurement['height_cm']:.1f}cm "
                      f"(Area: {measurement['area_cm2']:.1f}cm²)")
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Reset reference detection
            measurement_system.reference_detected = False
            measurement_system.pixels_per_cm = None
            print("Reference reset. Please place A4 paper again.")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
