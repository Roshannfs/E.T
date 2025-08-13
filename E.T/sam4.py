import cv2
import numpy as np
import math

class BrakePadMeasurement:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)  # Initialize webcam
        self.pixel_per_cm = None
        self.calibration_points = []
        self.calibrated = False
        
        # Set camera resolution for better accuracy
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("Brake Pad Measurement Tool")
        print("Instructions:")
        print("1. Place brake pad on a flat surface under good lighting")
        print("2. For calibration, click two points with known distance (e.g., ruler marks)")
        print("3. Enter the actual distance in centimeters when prompted")
        print("4. Press 'c' to calibrate, 'r' to reset, 'q' to quit")

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks for calibration"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.calibration_points) < 2:
                self.calibration_points.append((x, y))
                cv2.circle(param, (x, y), 5, (0, 0, 255), -1)
            else:
                self.calibration_points = [(x, y)]

    def calculate_distance(self, pt1, pt2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

    def calibrate(self, known_distance_cm):
        """Calibrate pixel-to-cm ratio using two reference points"""
        if len(self.calibration_points) == 2:
            pixel_distance = self.calculate_distance(
                self.calibration_points[0], 
                self.calibration_points[1]
            )
            self.pixel_per_cm = pixel_distance / known_distance_cm
            self.calibrated = True
            print(f"Calibration successful: {self.pixel_per_cm:.2f} pixels per cm")
            return True
        return False

    def preprocess_image(self, frame):
        """Preprocess image for better contour detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive threshold for better edge detection
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Morphological operations to clean up the image
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        return gray, thresh

    def find_brake_pad_contour(self, thresh):
        """Find the brake pad contour (largest meaningful contour)"""
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return None
            
        # Filter contours by area (brake pad should be reasonably large)
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 5000]
        
        if not valid_contours:
            return None
            
        # Return the largest contour (assuming it's the brake pad)
        return max(valid_contours, key=cv2.contourArea)

    def measure_brake_pad(self, contour, frame):
        """Measure brake pad dimensions and draw results"""
        if contour is None or not self.calibrated:
            return frame
        
        # Fill the contour to shade the inside of the brake pad
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [contour], 255)
        
        # Create colored overlay for the brake pad area
        overlay = frame.copy()
        overlay[mask == 255] = [0, 255, 0]  # Green overlay
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Draw contour outline
        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
        
        # Get minimum area rectangle (handles rotation)
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # Draw the rotated bounding box
        cv2.drawContours(frame, [box], 0, (255, 0, 0), 2)
        
        # Calculate dimensions
        (center_x, center_y), (width_px, height_px), angle = rect
        
        # Convert to real-world measurements
        length_cm = max(width_px, height_px) / self.pixel_per_cm
        width_cm = min(width_px, height_px) / self.pixel_per_cm
        
        # Calculate area
        area_px = cv2.contourArea(contour)
        area_cm2 = area_px / (self.pixel_per_cm ** 2)
        
        # Calculate perimeter
        perimeter_px = cv2.arcLength(contour, True)
        perimeter_cm = perimeter_px / self.pixel_per_cm
        
        # Display measurements on the image
        text_y = 30
        cv2.putText(frame, f"Length: {length_cm:.2f} cm", 
                    (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        text_y += 30
        cv2.putText(frame, f"Width: {width_cm:.2f} cm", 
                    (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        text_y += 30
        cv2.putText(frame, f"Area: {area_cm2:.2f} sq.cm", 
                    (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        text_y += 30
        cv2.putText(frame, f"Perimeter: {perimeter_cm:.2f} cm", 
                    (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Additional contour analysis for curves
        # Approximate the contour to reduce noise
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Calculate convex hull for better curve analysis
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area_px / hull_area if hull_area > 0 else 0
        
        text_y += 30
        cv2.putText(frame, f"Solidity: {solidity:.3f}", 
                    (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame

    def run(self):
        """Main measurement loop"""
        cv2.namedWindow("Brake Pad Measurement")
        cv2.setMouseCallback("Brake Pad Measurement", self.mouse_callback, None)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            original_frame = frame.copy()
            
            # Preprocess image
            gray, thresh = self.preprocess_image(frame)
            
            # Find brake pad contour
            brake_pad_contour = self.find_brake_pad_contour(thresh)
            
            # Measure and display results
            if self.calibrated and brake_pad_contour is not None:
                frame = self.measure_brake_pad(brake_pad_contour, frame)
            
            # Draw calibration points
            for i, point in enumerate(self.calibration_points):
                cv2.circle(frame, point, 8, (0, 0, 255), -1)
                cv2.putText(frame, f"P{i+1}", 
                           (point[0] + 10, point[1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Draw line between calibration points
            if len(self.calibration_points) == 2:
                cv2.line(frame, self.calibration_points[0], 
                        self.calibration_points[1], (255, 255, 0), 2)
                
                pixel_dist = self.calculate_distance(
                    self.calibration_points[0], 
                    self.calibration_points[1]
                )
                cv2.putText(frame, f"Distance: {pixel_dist:.1f} px", 
                           (10, frame.shape[0] - 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Status information
            status = "Calibrated" if self.calibrated else "Not Calibrated"
            cv2.putText(frame, f"Status: {status}", 
                       (10, frame.shape[0] - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.putText(frame, "Press 'c' to calibrate, 'r' to reset, 'q' to quit", 
                       (10, frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("Brake Pad Measurement", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c') and len(self.calibration_points) == 2:
                # Prompt for known distance
                print("Enter the actual distance between the two points in centimeters:")
                try:
                    known_distance = float(input())
                    if self.calibrate(known_distance):
                        print("Calibration successful!")
                    else:
                        print("Calibration failed. Please select two points first.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
            elif key == ord('r'):
                # Reset calibration
                self.calibration_points = []
                self.calibrated = False
                self.pixel_per_cm = None
                print("Reset complete. Please recalibrate.")
        
        self.cap.release()
        cv2.destroyAllWindows()

# Alternative method without reference object (using manual scaling)
def brake_pad_measurement_no_reference():
    """Alternative approach when no reference object is available"""
    cap = cv2.VideoCapture(0)
    
    # Estimated pixel-to-cm ratio (needs manual adjustment based on camera distance)
    # This is a rough estimate - adjust based on your setup
    estimated_pixel_per_cm = 10  # Adjust this value based on your camera setup
    
    print("Alternative measurement without reference object")
    print("Note: Measurements are estimates. Adjust 'estimated_pixel_per_cm' variable for accuracy")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale and threshold
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and get largest contour
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 3000]
        
        if valid_contours:
            largest_contour = max(valid_contours, key=cv2.contourArea)
            
            # Shade the inside of the brake pad
            cv2.fillPoly(frame, [largest_contour], (0, 255, 0))
            cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
            
            # Get dimensions
            rect = cv2.minAreaRect(largest_contour)
            (center_x, center_y), (width_px, height_px), angle = rect
            
            length_cm = max(width_px, height_px) / estimated_pixel_per_cm
            width_cm = min(width_px, height_px) / estimated_pixel_per_cm
            area_cm2 = cv2.contourArea(largest_contour) / (estimated_pixel_per_cm ** 2)
            
            # Display measurements
            cv2.putText(frame, f"Length: {length_cm:.2f} cm (est)", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Width: {width_cm:.2f} cm (est)", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Area: {area_cm2:.2f} sq.cm (est)", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow("Brake Pad Measurement (No Reference)", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Usage
if __name__ == "__main__":
    measurement_tool = BrakePadMeasurement()
    measurement_tool.run()
    
    # Uncomment below line to use alternative method without reference
    # brake_pad_measurement_no_reference()
