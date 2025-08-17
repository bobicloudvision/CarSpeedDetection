import cv2
import numpy as np
import pytesseract
from PIL import Image
import time
import math
import os
from collections import deque
import argparse

class CarTracker:
    def __init__(self, video_path=None, camera_index=0):
        """
        Initialize the car tracker with video source and tracking parameters
        """
        self.video_path = video_path
        self.camera_index = camera_index
        self.cap = None
        
        # Tracking parameters
        self.car_tracks = {}  # Dictionary to store car tracks
        self.next_track_id = 0
        self.max_disappeared = 30  # Max frames a car can disappear
        self.min_distance = 50  # Minimum distance to consider as same car
        
        # Speed estimation parameters
        self.fps = 30  # Assumed FPS, will be updated from video
        self.pixel_to_meter_ratio = 0.1  # Calibration factor
        self.speed_history = deque(maxlen=10)
        
        # Number plate detection parameters
        self.min_plate_area = 1000
        self.max_plate_area = 50000
        
        # Initialize background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=False
        )
        
        # Car detection cascade (optional)
        self.car_cascade = None
        try:
            # First try the local cars.xml file
            if os.path.exists('cars.xml'):
                self.car_cascade = cv2.CascadeClassifier('cars.xml')
                if not self.car_cascade.empty():
                    print("✓ Loaded local car cascade file (cars.xml)")
                else:
                    self.car_cascade = None
                    print("Warning: Local car cascade file is empty.")
            # Fallback to OpenCV's built-in cascade
            elif os.path.exists(cv2.data.haarcascades + 'haarcascade_cars.xml'):
                self.car_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_cars.xml')
                if not self.car_cascade.empty():
                    print("✓ Loaded OpenCV built-in car cascade file")
                else:
                    self.car_cascade = None
                    print("Warning: Built-in car cascade file is empty.")
            else:
                print("No car cascade file found. Using background subtraction only.")
        except Exception as e:
            print(f"Warning: Could not load car cascade: {e}")
            self.car_cascade = None
        
        # Initialize video capture
        self._init_video_capture()
        
    def _init_video_capture(self):
        """Initialize video capture from file or camera"""
        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
        else:
            self.cap = cv2.VideoCapture(self.camera_index)
            
        if not self.cap.isOpened():
            raise ValueError("Could not open video source")
            
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
    def preprocess_frame(self, frame):
        """Preprocess frame for better detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        morph = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)
        
        return morph
        
    def detect_cars(self, frame):
        """Detect cars in the frame using multiple methods"""
        cars = []
        
        # Method 1: Background subtraction (primary method)
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 800:  # Lower threshold for better detection
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h)
                
                # More flexible aspect ratio for cars
                if 0.8 < aspect_ratio < 4.0:
                    # Additional filtering: minimum width and height
                    if w > 30 and h > 20:
                        cars.append((x, y, w, h))
        
        # Method 2: Haar cascade (if available, as backup)
        if self.car_cascade is not None:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cascade_cars = self.car_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30)
                )
                
                for (x, y, w, h) in cascade_cars:
                    cars.append((x, y, w, h))
            except Exception as e:
                # Silently continue with background subtraction only
                pass
        
        # Remove duplicate detections (simple overlap check)
        if len(cars) > 1:
            filtered_cars = []
            for i, car1 in enumerate(cars):
                is_duplicate = False
                for j, car2 in enumerate(cars):
                    if i != j:
                        # Check overlap
                        x1, y1, w1, h1 = car1
                        x2, y2, w2, h2 = car2
                        
                        # Calculate overlap area
                        overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                        overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                        overlap_area = overlap_x * overlap_y
                        
                        if overlap_area > 0.5 * min(w1 * h1, w2 * h2):
                            is_duplicate = True
                            break
                
                if not is_duplicate:
                    filtered_cars.append(car1)
            
            cars = filtered_cars
        
        return cars
        
    def detect_number_plate(self, frame, car_bbox):
        """Detect number plate within car bounding box"""
        x, y, w, h = car_bbox
        
        # Extract car region
        car_region = frame[y:y+h, x:x+w]
        
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(car_region, cv2.COLOR_BGR2HSV)
        
        # Define range for white color (typical for number plates)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        
        # Create mask for white regions
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Find contours in the white mask
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        plates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_plate_area < area < self.max_plate_area:
                x_plate, y_plate, w_plate, h_plate = cv2.boundingRect(contour)
                aspect_ratio = w_plate / float(h_plate)
                
                # Number plates typically have aspect ratio around 2.5-4.0
                if 2.0 < aspect_ratio < 5.0:
                    plates.append((x_plate, y_plate, w_plate, h_plate))
        
        return plates
        
    def read_number_plate(self, frame, plate_bbox, car_bbox):
        """Read text from detected number plate using OCR"""
        x_car, y_car, w_car, h_car = car_bbox
        x_plate, y_plate, w_plate, h_plate = plate_bbox
        
        # Extract plate region (relative to original frame)
        plate_x = x_car + x_plate
        plate_y = y_car + y_plate
        plate_region = frame[plate_y:plate_y+h_plate, plate_x:plate_x+w_plate]
        
        # Preprocess plate image for better OCR
        gray_plate = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding
        _, thresh = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Resize for better OCR
        processed = cv2.resize(processed, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        
        try:
            # OCR configuration for number plates
            config = '--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            text = pytesseract.image_to_string(processed, config=config)
            
            # Clean up the text
            text = ''.join(c for c in text if c.isalnum())
            
            return text if len(text) >= 4 else None
        except Exception as e:
            print(f"OCR Error: {e}")
            return None
            
    def update_tracks(self, cars):
        """Update car tracks and assign IDs"""
        # If no cars detected, mark all tracks as disappeared
        if len(cars) == 0:
            for track_id in list(self.car_tracks.keys()):
                self.car_tracks[track_id]['disappeared'] += 1
                
                # Remove tracks that have disappeared for too long
                if self.car_tracks[track_id]['disappeared'] > self.max_disappeared:
                    del self.car_tracks[track_id]
            return
            
        # If no existing tracks, create new ones
        if len(self.car_tracks) == 0:
            for car in cars:
                self.car_tracks[self.next_track_id] = {
                    'bbox': car,
                    'center': (car[0] + car[2]//2, car[1] + car[3]//2),
                    'disappeared': 0,
                    'speed': 0,
                    'plate_number': None,
                    'first_seen': time.time()
                }
                self.next_track_id += 1
        else:
            # Match existing tracks with new detections
            track_ids = list(self.car_tracks.keys())
            car_centers = [(car[0] + car[2]//2, car[1] + car[3]//2) for car in cars]
            
            # Calculate distances between existing tracks and new detections
            distances = []
            for track_id in track_ids:
                for i, car_center in enumerate(car_centers):
                    distance = math.sqrt(
                        (self.car_tracks[track_id]['center'][0] - car_center[0])**2 +
                        (self.car_tracks[track_id]['center'][1] - car_center[1])**2
                    )
                    distances.append((distance, track_id, i))
            
            # Sort by distance
            distances.sort()
            
            # Track which cars and tracks have been matched
            matched_cars = set()
            matched_tracks = set()
            
            for distance, track_id, car_idx in distances:
                if track_id not in matched_tracks and car_idx not in matched_cars and distance < self.min_distance:
                    # Update existing track
                    old_center = self.car_tracks[track_id]['center']
                    new_center = car_centers[car_idx]
                    
                    # Calculate speed
                    speed = self._calculate_speed(old_center, new_center)
                    self.car_tracks[track_id]['speed'] = speed
                    
                    # Update track
                    self.car_tracks[track_id]['bbox'] = cars[car_idx]
                    self.car_tracks[track_id]['center'] = new_center
                    self.car_tracks[track_id]['disappeared'] = 0
                    
                    matched_cars.add(car_idx)
                    matched_tracks.add(track_id)
            
            # Handle unmatched tracks
            for track_id in track_ids:
                if track_id not in matched_tracks:
                    self.car_tracks[track_id]['disappeared'] += 1
                    
                    if self.car_tracks[track_id]['disappeared'] > self.max_disappeared:
                        del self.car_tracks[track_id]
            
            # Handle unmatched cars (new cars)
            for i, car in enumerate(cars):
                if i not in matched_cars:
                    self.car_tracks[self.next_track_id] = {
                        'bbox': car,
                        'center': car_centers[i],
                        'disappeared': 0,
                        'speed': 0,
                        'plate_number': None,
                        'first_seen': time.time()
                    }
                    self.next_track_id += 1
                    
    def _calculate_speed(self, old_center, new_center):
        """Calculate speed based on pixel movement"""
        dx = new_center[0] - old_center[0]
        dy = new_center[1] - old_center[1]
        
        # Calculate distance in pixels
        distance_pixels = math.sqrt(dx**2 + dy**2)
        
        # Convert to meters using calibration factor
        distance_meters = distance_pixels * self.pixel_to_meter_ratio
        
        # Calculate speed in m/s
        speed = distance_meters * self.fps
        
        # Convert to km/h
        speed_kmh = speed * 3.6
        
        return speed_kmh
        
    def process_frame(self, frame):
        """Process a single frame for car tracking and number plate recognition"""
        # Detect cars
        cars = self.detect_cars(frame)
        
        # Update tracks
        self.update_tracks(cars)
        
        # Process each tracked car
        for track_id, track_info in self.car_tracks.items():
            bbox = track_info['bbox']
            center = track_info['center']
            speed = track_info['speed']
            plate_number = track_info['plate_number']
            
            # Draw bounding box
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw track ID
            cv2.putText(frame, f"ID: {track_id}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Draw speed
            cv2.putText(frame, f"Speed: {speed:.1f} km/h", (x, y + h + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Try to detect number plate if not already found
            if plate_number is None:
                plates = self.detect_number_plate(frame, bbox)
                if plates:
                    # Use the largest plate
                    largest_plate = max(plates, key=lambda p: p[2] * p[3])
                    plate_text = self.read_number_plate(frame, largest_plate, bbox)
                    
                    if plate_text:
                        track_info['plate_number'] = plate_text
                        plate_number = plate_text
            
            # Draw plate number if found
            if plate_number:
                cv2.putText(frame, f"Plate: {plate_number}", (x, y + h + 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Draw plate bounding box
                plates = self.detect_number_plate(frame, bbox)
                if plates:
                    largest_plate = max(plates, key=lambda p: p[2] * p[3])
                    x_plate, y_plate, w_plate, h_plate = largest_plate
                    cv2.rectangle(frame, 
                                (x + x_plate, y + y_plate),
                                (x + x_plate + w_plate, y + y_plate + h_plate),
                                (0, 0, 255), 2)
        
        return frame
        
    def run(self):
        """Main loop for processing video"""
        frame_count = 0
        start_time = time.time()
        
        print("Starting video processing...")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print(f"End of video reached at frame {frame_count}")
                break
                
            frame_count += 1
            
            if frame_count % 30 == 0:  # Print every 30 frames
                print(f"Processing frame {frame_count}")
            
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Add frame counter and FPS
            elapsed_time = time.time() - start_time
            current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            cv2.putText(processed_frame, f"Frame: {frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(processed_frame, f"FPS: {current_fps:.1f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(processed_frame, f"Cars: {len(self.car_tracks)}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow('Car Tracker', processed_frame)
            
            # Break on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("User pressed 'q', exiting...")
                break
                
        print(f"Total frames processed: {frame_count}")
        self.cap.release()
        cv2.destroyAllWindows()
        
    def __del__(self):
        """Cleanup"""
        if self.cap:
            self.cap.release()

def main():
    parser = argparse.ArgumentParser(description='Car Tracking with Number Plate Recognition')
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    
    args = parser.parse_args()
    
    try:
        tracker = CarTracker(video_path=args.video, camera_index=args.camera)
        tracker.run()
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 