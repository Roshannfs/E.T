import cv2
import numpy as np
import math

class BrakePadShapeMeasurer:
    def __init__(self):
        self.pixels_per_cm = None
        self.calibrated = False
        
    def calibrate_with_known_object(self, frame, known_width_cm):
        """
        Calibrate the system using an object of known width
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        edged = cv2.Canny(blurred, 50, 100)
        edged = cv2.dilate(edged, None, iterations=1)
        edged = cv2.erode(edged, None, iterations=1)
        
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            self.pixels_per_cm = w / known_width_cm
            self.calibrated = True
            
            print(f"Calibrated: {self.pixels_per_cm:.2f} pixels per cm")
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Calibration: {known_width_cm}cm", 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame
    
    def find_brake_pad_shape(self, frame):
        """
        Find and measure the actual brake pad shape with curves and edges
        """
        if not self.calibrated:
            cv2.putText(frame, "SYSTEM NOT CALIBRATED", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Enhanced preprocessing for brake pad detection
        # Apply bilateral filter to reduce noise while keeping edges sharp
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Use adaptive threshold for better shape detection
        thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Invert if brake pad is darker than background
        thresh_inv = cv2.bitwise_not(thresh)
        
        # Combine both thresholds to handle different lighting conditions
        combined = cv2.bitwise_or(thresh, thresh_inv)
        
        # Apply morphological operations to clean up the shape
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find contours with full hierarchy
        contours, hierarchy = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area (brake pads have significant size)
            if area < 3000:
                continue
            
            # Calculate contour properties
            perimeter = cv2.arcLength(contour, True)
            
            # Calculate solidity (area/convex_hull_area) to filter brake pad shapes
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # Brake pads typically have solidity between 0.7-0.95 (not too irregular)
            if solidity < 0.6 or solidity > 0.98:
                continue
            
            # Get the actual shape measurements
            measurements = self.measure_brake_pad_shape(contour, frame)
            
            if measurements:
                # Draw the actual brake pad shape (not rectangle)
                cv2.drawContours(frame, [contour], -1, (0, 255, 0), 3)
                
                # Fill the brake pad area with semi-transparent color
                overlay = frame.copy()
                cv2.fillPoly(overlay, [contour], (0, 255, 0))
                frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)
                
                # Draw measurements
                self.draw_shape_measurements(frame, contour, measurements)
        
        return frame
    
    def measure_brake_pad_shape(self, contour, frame):
        """
        Measure the actual brake pad shape dimensions
        """
        # Get bounding rectangle for reference
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate actual shape area
        shape_area_pixels = cv2.contourArea(contour)
        shape_area_cm2 = shape_area_pixels / (self.pixels_per_cm ** 2)
        
        # Calculate perimeter
        perimeter_pixels = cv2.arcLength(contour, True)
        perimeter_cm = perimeter_pixels / self.pixels_per_cm
        
        # Find extreme points for actual shape dimensions
        leftmost = tuple(contour[contour[:, :, 0].argmin()][0])
        rightmost = tuple(contour[contour[:, :, 0].argmax()][0])
        topmost = tuple(contour[contour[:, :, 1].argmin()][0])
        bottommost = tuple(contour[contour[:, :, 1].argmax()][0])
        
        # Calculate actual width and height based on shape extremes
        actual_width_pixels = abs(rightmost[0] - leftmost[0])
        actual_height_pixels = abs(bottommost[1] - topmost[1])
        
        actual_width_cm = actual_width_pixels / self.pixels_per_cm
        actual_height_cm = actual_height_pixels / self.pixels_per_cm
        
        # Calculate shape complexity (useful for brake pad identification)
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        complexity = len(approx)
        
        # Fit ellipse to get curved measurements
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            ellipse_width_cm = ellipse[1][0] / self.pixels_per_cm
            ellipse_height_cm = ellipse[1][1] / self.pixels_per_cm
        else:
            ellipse_width_cm = actual_width_cm
            ellipse_height_cm = actual_height_cm
        
        return {
            'area_cm2': shape_area_cm2,
            'perimeter_cm': perimeter_cm,
            'width_cm': actual_width_cm,
            'height_cm': actual_height_cm,
            'ellipse_width_cm': ellipse_width_cm,
            'ellipse_height_cm': ellipse_height_cm,
            'complexity': complexity,
            'leftmost': leftmost,
            'rightmost': rightmost,
            'topmost': topmost,
            'bottommost': bottommost
        }
    
    def draw_shape_measurements(self, frame, contour, measurements):
        """
        Draw measurements on the actual brake pad shape
        """
        # Get contour center
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = measurements['leftmost']
        
        # Draw extreme points
        cv2.circle(frame, measurements['leftmost'], 8, (255, 0, 0), -1)
        cv2.circle(frame, measurements['rightmost'], 8, (255, 0, 0), -1)
        cv2.circle(frame, measurements['topmost'], 8, (0, 0, 255), -1)
        cv2.circle(frame, measurements['bottommost'], 8, (0, 0, 255), -1)
        cv2.circle(frame, (cx, cy), 6, (255, 255, 0), -1)
        
        # Draw width and height lines on actual shape
        cv2.line(frame, measurements['leftmost'], measurements['rightmost'], (255, 0, 0), 2)
        cv2.line(frame, measurements['topmost'], measurements['bottommost'], (0, 0, 255), 2)
        
        # Display measurements near the brake pad
        y_offset = 0
        measurements_text = [
            f"Brake Pad Shape Analysis",
            f"Width: {measurements['width_cm']:.1f}cm",
            f"Height: {measurements['height_cm']:.1f}cm",
            f"Area: {measurements['area_cm2']:.1f}cm²",
            f"Perimeter: {measurements['perimeter_cm']:.1f}cm",
            f"Ellipse W: {measurements['ellipse_width_cm']:.1f}cm",
            f"Ellipse H: {measurements['ellipse_height_cm']:.1f}cm"
        ]
        
        for i, text in enumerate(measurements_text):
            color = (0, 255, 0) if i == 0 else (255, 255, 255)
            thickness = 2 if i == 0 else 1
            cv2.putText(frame, text, (cx + 20, cy - 80 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Initialize brake pad measurer
    measurer = BrakePadShapeMeasurer()
    
    # Calibration settings
    calibration_mode = False
    known_object_width = 5.0  # Change this to your known object width in cm
    
    print("=== Brake Pad Shape Measurement System ===")
    print("This system measures the ACTUAL brake pad shape with curves and edges")
    print("Controls:")
    print("'c' - Enter calibration mode")
    print("'space' - Calibrate with current largest object")
    print("'r' - Reset calibration")
    print("'q' - Quit")
    print(f"Calibration object width: {known_object_width}cm")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Handle different modes
        if calibration_mode:
            frame = measurer.calibrate_with_known_object(frame, known_object_width)
            cv2.putText(frame, "CALIBRATION MODE - Press SPACE to calibrate", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            frame = measurer.find_brake_pad_shape(frame)
        
        # Display status
        status = "CALIBRATED - SHAPE MODE" if measurer.calibrated else "NOT CALIBRATED"
        color = (0, 255, 0) if measurer.calibrated else (0, 0, 255)
        cv2.putText(frame, f"Status: {status}", (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Show frame
        cv2.imshow('Brake Pad Shape Measurement', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('c'):
            calibration_mode = not calibration_mode
            print(f"Calibration mode: {'ON' if calibration_mode else 'OFF'}")
        elif key == ord(' ') and calibration_mode:
            print("Calibrating...")
        elif key == ord('r'):
            measurer.calibrated = False
            measurer.pixels_per_cm = None
            print
