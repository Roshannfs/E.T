import cv2
import numpy as np
import math

class ObjectMeasurer:
    def __init__(self):
        # This will be calibrated once with a known object
        self.pixels_per_cm = None
        self.calibrated = False
        
    def calibrate_with_known_object(self, frame, known_width_cm):
        """
        Calibrate the system using an object of known width
        Args:
            frame: Current frame from webcam
            known_width_cm: Known width of calibration object in cm
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
        # Apply edge detection
        edged = cv2.Canny(blurred, 50, 100)
        
        # Dilate and erode to close gaps in edges
        edged = cv2.dilate(edged, None, iterations=1)
        edged = cv2.erode(edged, None, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour (assuming it's our calibration object)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Calculate pixels per cm using the width
            self.pixels_per_cm = w / known_width_cm
            self.calibrated = True
            
            print(f"Calibrated: {self.pixels_per_cm:.2f} pixels per cm")
            
            # Draw calibration object
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Calibration: {known_width_cm}cm", 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame
    
    def find_and_measure_objects(self, frame):
        """
        Find and measure all objects in the frame
        """
        if not self.calibrated:
            cv2.putText(frame, "SYSTEM NOT CALIBRATED", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
        # Apply edge detection
        edged = cv2.Canny(blurred, 50, 100)
        
        # Dilate and erode to close gaps
        edged = cv2.dilate(edged, None, iterations=1)
        edged = cv2.erode(edged, None, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area to remove noise
        min_area = 1000  # Minimum area threshold
        filtered_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        
        for i, contour in enumerate(filtered_contours):
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate dimensions in cm
            width_cm = w / self.pixels_per_cm
            height_cm = h / self.pixels_per_cm
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Draw measurements
            cv2.putText(frame, f"W: {width_cm:.1f}cm", 
                       (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(frame, f"H: {height_cm:.1f}cm", 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Draw center point
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.circle(frame, (center_x, center_y), 5, (0, 255, 255), -1)
        
        return frame

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Set camera properties for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Initialize measurer
    measurer = ObjectMeasurer()
    
    # Calibration mode flag
    calibration_mode = False
    known_object_width = 5.0  # Change this to your known object width in cm
    
    print("=== Object Dimension Measurement System ===")
    print("Controls:")
    print("'c' - Enter calibration mode")
    print("'space' - Calibrate with current largest object")
    print("'r' - Reset calibration")
    print("'q' - Quit")
    print(f"Set known_object_width to your calibration object size (currently {known_object_width}cm)")
    
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
            frame = measurer.find_and_measure_objects(frame)
        
        # Display status
        status = "CALIBRATED" if measurer.calibrated else "NOT CALIBRATED"
        color = (0, 255, 0) if measurer.calibrated else (0, 0, 255)
        cv2.putText(frame, f"Status: {status}", (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Show frame
        cv2.imshow('Object Measurement', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('c'):
            calibration_mode = not calibration_mode
            print(f"Calibration mode: {'ON' if calibration_mode else 'OFF'}")
        elif key == ord(' ') and calibration_mode:
            # This will trigger calibration in the next frame
            print("Calibrating...")
        elif key == ord('r'):
            measurer.calibrated = False
            measurer.pixels_per_cm = None
            print("Calibration reset")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

# Enhanced version with better edge detection for specific objects
class BrakePadMeasurer(ObjectMeasurer):
    def __init__(self):
        super().__init__()
    
    def find_brake_pads(self, frame):
        """
        Specialized function to detect brake pad-like objects
        """
        if not self.calibrated:
            cv2.putText(frame, "SYSTEM NOT CALIBRATED", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply stronger blur for brake pads
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        
        # Use adaptive threshold for better edge detection
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Apply morphological operations
        kernel = np.ones((3,3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter for rectangular shapes (brake pads are usually rectangular)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 2000:  # Filter small objects
                continue
            
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if it's roughly rectangular (4 corners)
            if len(approx) >= 4:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                
                # Brake pads usually have specific aspect ratios
                if 0.3 < aspect_ratio < 3.0:
                    # Calculate dimensions
                    width_cm = w / self.pixels_per_cm
                    height_cm = h / self.pixels_per_cm
                    
                    # Draw detection
                    cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    
                    # Display measurements
                    cv2.putText(frame, f"Brake Pad", (x, y - 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"W: {width_cm:.1f}cm", (x, y - 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    cv2.putText(frame, f"H: {height_cm:.1f}cm", (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        return frame

if __name__ == "__main__":
    main()
