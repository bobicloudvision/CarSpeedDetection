import cv2
import numpy as np
import argparse

class CalibrationTool:
    def __init__(self, video_path=None, camera_index=0):
        """
        Tool for calibrating the pixel-to-meter ratio for speed estimation
        """
        self.video_path = video_path
        self.camera_index = camera_index
        self.cap = None
        self.calibration_points = []
        self.known_distance = 0  # Known distance in meters
        self.pixel_distance = 0  # Distance in pixels
        
        self._init_video_capture()
        
    def _init_video_capture(self):
        """Initialize video capture"""
        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
        else:
            self.cap = cv2.VideoCapture(self.camera_index)
            
        if not self.cap.isOpened():
            raise ValueError("Could not open video source")
        
        # Get video properties and validate
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if self.frame_width <= 0 or self.frame_height <= 0:
            raise ValueError("Invalid video dimensions")
    
    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for selecting calibration points"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.calibration_points) < 2:
                self.calibration_points.append((x, y))
                print(f"Point {len(self.calibration_points)}: ({x}, {y})")
                
                if len(self.calibration_points) == 2:
                    self.calculate_ratio()
    
    def calculate_ratio(self):
        """Calculate pixel-to-meter ratio"""
        if len(self.calibration_points) == 2:
            p1 = self.calibration_points[0]
            p2 = self.calibration_points[1]
            
            # Calculate pixel distance
            self.pixel_distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            
            # Get known distance from user
            try:
                self.known_distance = float(input("Enter the known distance in meters: "))
                ratio = self.known_distance / self.pixel_distance
                print(f"Pixel-to-meter ratio: {ratio:.6f}")
                print(f"Use this value for pixel_to_meter_ratio in car_tracker.py")
            except ValueError:
                print("Invalid distance value")
    
    def run(self):
        """Run calibration tool"""
        cv2.namedWindow('Calibration')
        cv2.setMouseCallback('Calibration', self.mouse_callback)
        
        print("Click two points to measure distance")
        print("Press 'r' to reset, 'q' to quit")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                # Loop video if it's a file
                if self.video_path:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    if not ret:
                        print("Failed to read video frame, exiting...")
                        break
                else:
                    break
            
            # Ensure frame is valid
            if frame is None or frame.size == 0:
                print("Invalid frame received, skipping...")
                continue
            
            # Draw calibration points
            for i, point in enumerate(self.calibration_points):
                cv2.circle(frame, point, 5, (0, 255, 0), -1)
                cv2.putText(frame, f"P{i+1}", (point[0] + 10, point[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw line between points
            if len(self.calibration_points) == 2:
                cv2.line(frame, self.calibration_points[0], self.calibration_points[1], (0, 255, 0), 2)
                
                # Display distance information
                if self.pixel_distance > 0 and self.known_distance > 0:
                    ratio = self.known_distance / self.pixel_distance
                    cv2.putText(frame, f"Ratio: {ratio:.6f} m/pixel", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"Pixel dist: {self.pixel_distance:.1f}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"Real dist: {self.known_distance:.2f} m", (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display instructions
            cv2.putText(frame, "Click 2 points to calibrate", (10, self.frame_height - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, "Press 'r' to reset, 'q' to quit", (10, self.frame_height - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Calibration', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.calibration_points = []
                self.pixel_distance = 0
                self.known_distance = 0
                print("Calibration reset")
        
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Calibration Tool for Car Tracker')
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    
    args = parser.parse_args()
    
    try:
        calibrator = CalibrationTool(video_path=args.video, camera_index=args.camera)
        calibrator.run()
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 