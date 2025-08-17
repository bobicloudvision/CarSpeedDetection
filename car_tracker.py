import cv2
import numpy as np
import pytesseract
from PIL import Image
import time
import math
import os
from collections import deque
import argparse

class VirtualLine:
    def __init__(self, x1, y1, x2, y2, name="Line"):
        """Initialize a virtual line for speed detection"""
        self.x1, self.y1 = x1, y1
        self.x2, self.y2 = x2, y2
        self.name = name
        self.crossed_cars = {}  # Track cars that crossed this line
        
    def draw(self, frame):
        """Draw the virtual line on the frame"""
        cv2.line(frame, (self.x1, self.y1), (self.x2, self.y2), (0, 255, 255), 2)
        cv2.putText(frame, self.name, (self.x1, self.y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    def check_crossing(self, car_center, car_id, frame_time):
        """Check if a car has crossed this line"""
        # Calculate which side of the line the car is on
        # Using cross product to determine side
        line_vector = (self.x2 - self.x1, self.y2 - self.y1)
        car_vector = (car_center[0] - self.x1, car_center[1] - self.y1)
        
        # Cross product to determine side
        cross_product = line_vector[0] * car_vector[1] - line_vector[1] * car_vector[0]
        
        # If car hasn't been tracked for this line yet, initialize
        if car_id not in self.crossed_cars:
            self.crossed_cars[car_id] = {
                'side': 'left' if cross_product > 0 else 'right',
                'crossed': False,
                'cross_time': None
            }
            return False
        
        current_side = 'left' if cross_product > 0 else 'right'
        previous_side = self.crossed_cars[car_id]['side']
        
        # Check if car crossed the line (changed sides)
        if current_side != previous_side and not self.crossed_cars[car_id]['crossed']:
            self.crossed_cars[car_id]['crossed'] = True
            self.crossed_cars[car_id]['cross_time'] = frame_time
            self.crossed_cars[car_id]['side'] = current_side
            return True
        
        # Update current side
        self.crossed_cars[car_id]['side'] = current_side
        return False

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
        
        # Virtual lines for speed detection
        self.virtual_lines = []
        self.line_distance_meters = 10.0  # Distance between lines in meters
        self.line1_y_ratio = 0.4  # Line 1 position (40% from top)
        self.line2_y_ratio = 0.6  # Line 2 position (60% from top)
        
        # Interactive line setup
        self.interactive_mode = False
        self.line_setup_points = []
        self.current_setup_line = 0  # 0 for line 1, 1 for line 2
        
        # Performance optimization flags
        self.enable_plate_detection = True
        self.plate_detection_frequency = 10  # Check plates every N frames
        self.enable_haar_cascade = False  # Disable by default for better performance
        self.frame_count = 0  # Track frame count for processing decisions

        
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
        
        # Setup virtual lines
        self._setup_virtual_lines()
        
    def _setup_virtual_lines(self):
        """Setup virtual lines for speed detection"""
        if self.frame_width > 0 and self.frame_height > 0:
            # Create two horizontal lines across the road
            line1_y = int(self.frame_height * self.line1_y_ratio)
            line2_y = int(self.frame_height * self.line2_y_ratio)
            
            self.virtual_lines = [
                VirtualLine(0, line1_y, self.frame_width, line1_y, "Start Line"),
                VirtualLine(0, line2_y, self.frame_width, line2_y, "Finish Line")
            ]
            print(f"✓ Virtual lines set up: {self.line_distance_meters}m apart")
            print(f"  Line 1 at {self.line1_y_ratio*100:.0f}% of frame height")
            print(f"  Line 2 at {self.line2_y_ratio*100:.0f}% of frame height")
        else:
            print("Warning: Cannot setup virtual lines - video dimensions unknown")
    
    def start_interactive_setup(self):
        """Start interactive line setup mode"""
        self.interactive_mode = True
        self.line_setup_points = []
        self.current_setup_line = 0
        print("🎯 Interactive line setup mode activated!")
        print("Click to set Line 1 (Start Line) position")
        print("Press 'r' to reset, 'q' to quit setup mode")
    
    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for interactive line setup"""
        if not self.interactive_mode:
            return
            
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.current_setup_line == 0:
                # Setting Line 1 (Start Line)
                self.line_setup_points = [(0, y), (self.frame_width, y)]
                self.current_setup_line = 1
                print(f"✓ Line 1 set at y={y} ({(y/self.frame_height)*100:.1f}% of frame)")
                print("Now click to set Line 2 (Finish Line) position")
            elif self.current_setup_line == 1:
                # Setting Line 2 (Finish Line)
                line2_points = [(0, y), (self.frame_width, y)]
                
                # Ensure Line 2 is below Line 1
                if y <= self.line_setup_points[0][1]:
                    print("⚠️  Line 2 must be below Line 1. Please click below Line 1.")
                    return
                
                # Update virtual lines
                self.virtual_lines[0] = VirtualLine(*self.line_setup_points[0], *self.line_setup_points[1], "Start Line")
                self.virtual_lines[1] = VirtualLine(*line2_points[0], *line2_points[1], "Finish Line")
                
                # Update ratios
                self.line1_y_ratio = self.line_setup_points[0][1] / self.frame_height
                self.line2_y_ratio = y / self.frame_height
                
                print(f"✓ Line 2 set at y={y} ({(y/self.frame_height)*100:.1f}% of frame)")
                print(f"✓ Virtual lines updated! Distance: {self.line_distance_meters}m")
                print("Press 's' to start tracking or 'q' to quit")
                
                self.interactive_mode = False
    
    def set_line_distance(self, distance_meters):
        """Set the distance between virtual lines in meters"""
        self.line_distance_meters = distance_meters
        print(f"Line distance updated to {distance_meters}m")
    
    def set_line_positions(self, line1_ratio, line2_ratio):
        """Set the vertical positions of the lines (0.0 to 1.0)"""
        self.line1_y_ratio = max(0.1, min(0.9, line1_ratio))
        self.line2_y_ratio = max(0.1, min(0.9, line2_ratio))
        
        # Ensure line2 is below line1
        if self.line2_y_ratio <= self.line1_y_ratio:
            self.line2_y_ratio = min(0.9, self.line1_y_ratio + 0.1)
        
        # Recalculate line positions
        if self.frame_width > 0 and self.frame_height > 0:
            line1_y = int(self.frame_height * self.line1_y_ratio)
            line2_y = int(self.frame_height * self.line2_y_ratio)
            
            self.virtual_lines[0] = VirtualLine(0, line1_y, self.frame_width, line1_y, "Start Line")
            self.virtual_lines[1] = VirtualLine(0, line2_y, self.frame_width, line2_y, "Finish Line")
            
            print(f"Line positions updated: Line 1 at {self.line1_y_ratio*100:.0f}%, Line 2 at {self.line2_y_ratio*100:.0f}%")
    
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
        
        # Method 1: Background subtraction (primary method) - OPTIMIZED
        fg_mask = self.bg_subtractor.apply(frame)
        
        # OPTIMIZATION: Use smaller kernel and reduce morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # Reduced from (5,5)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        # OPTIMIZATION: Removed MORPH_OPEN operation for better performance
        
        # Find contours with optimized parameters
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # OPTIMIZATION: Process only first 20 contours to avoid slowdown
        max_contours = min(20, len(contours))
        for i in range(max_contours):
            contour = contours[i]
            area = cv2.contourArea(contour)
            if area > 800:  # Lower threshold for better detection
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h)
                
                # More flexible aspect ratio for cars
                if 0.8 < aspect_ratio < 4.0:
                    # Additional filtering: minimum width and height
                    if w > 30 and h > 20:
                        cars.append((x, y, w, h))
        
        # Method 2: Haar cascade (if available, as backup) - OPTIMIZED
        if self.enable_haar_cascade and self.car_cascade is not None:
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
        
        # OPTIMIZATION: Simplified duplicate removal - only check if more than 5 cars
        if len(cars) > 5:
            # Simple area-based filtering instead of complex overlap calculation
            cars = sorted(cars, key=lambda c: c[2] * c[3], reverse=True)[:5]
        
        return cars
        
    def detect_number_plate(self, frame, car_bbox):
        """Detect number plate within car bounding box - OPTIMIZED"""
        x, y, w, h = car_bbox
        
        # OPTIMIZATION: Skip small cars to avoid processing tiny regions
        if w < 50 or h < 30:
            return []
        
        # Extract car region
        car_region = frame[y:y+h, x:x+w]
        
        # OPTIMIZATION: Resize car region to reduce processing time
        if w > 200 or h > 150:
            scale_factor = min(200.0/w, 150.0/h)
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            car_region = cv2.resize(car_region, (new_w, new_h))
        
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(car_region, cv2.COLOR_BGR2HSV)
        
        # Define range for white color (typical for number plates)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        
        # Create mask for white regions
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # OPTIMIZATION: Limit contour search to avoid slowdown
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        plates = []
        # OPTIMIZATION: Process only first 10 contours
        max_plate_contours = min(10, len(contours))
        for i in range(max_plate_contours):
            contour = contours[i]
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
                    'first_seen': time.time(),
                    'line1_crossed': False,
                    'line2_crossed': False,
                    'line1_time': None,
                    'line2_time': None
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
                        'first_seen': time.time(),
                        'line1_crossed': False,
                        'line2_crossed': False,
                        'line1_time': None,
                        'line2_time': None
                    }
                    self.next_track_id += 1
                    
    def calculate_speed_from_lines(self, car_id):
        """Calculate speed based on time between crossing two virtual lines"""
        track_info = self.car_tracks[car_id]
        
        if track_info['line1_crossed'] and track_info['line2_crossed']:
            if track_info['line1_time'] and track_info['line2_time']:
                time_diff = abs(track_info['line2_time'] - track_info['line1_time'])
                if time_diff > 0:
                    # Speed = distance / time
                    speed_mps = self.line_distance_meters / time_diff
                    speed_kmh = speed_mps * 3.6
                    return speed_kmh
        return 0
    
    def check_line_crossings(self, frame_time):
        """Check if cars have crossed the virtual lines"""
        for car_id, track_info in self.car_tracks.items():
            car_center = track_info['center']
            
            # Check line 1 crossing
            if not track_info['line1_crossed']:
                if self.virtual_lines[0].check_crossing(car_center, car_id, frame_time):
                    track_info['line1_crossed'] = True
                    track_info['line1_time'] = frame_time
                    print(f"Car {car_id} crossed Line 1 at {frame_time:.2f}s")
            
            # Check line 2 crossing
            if track_info['line1_crossed'] and not track_info['line2_crossed']:
                if self.virtual_lines[1].check_crossing(car_center, car_id, frame_time):
                    track_info['line2_crossed'] = True
                    track_info['line2_time'] = frame_time
                    print(f"Car {car_id} crossed Line 2 at {frame_time:.2f}s")
                    
                    # Calculate speed
                    speed = self.calculate_speed_from_lines(car_id)
                    track_info['speed'] = speed
                    print(f"Car {car_id} speed: {speed:.1f} km/h")
        
    def process_frame(self, frame):
        """Process a single frame for car tracking and number plate recognition"""
        # Detect cars
        cars = self.detect_cars(frame)
        
        # Update tracks
        self.update_tracks(cars)
        
        # Check line crossings
        frame_time = time.time()
        self.check_line_crossings(frame_time)
        
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
            if speed > 0:
                cv2.putText(frame, f"Speed: {speed:.1f} km/h", (x, y + h + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Try to detect number plate if not already found - OPTIMIZED
            if plate_number is None and self.enable_plate_detection:
                # OPTIMIZATION: Only check plates every N frames for better performance
                if self.frame_count % self.plate_detection_frequency == 0:
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
        
        # Draw virtual lines
        for line in self.virtual_lines:
            line.draw(frame)
        
        return frame
        
    def run(self):
        """Main loop for processing video"""
        self.frame_count = 0
        start_time = time.time()
        
        # Setup mouse callback for interactive line setup
        cv2.namedWindow('Car Tracker')
        cv2.setMouseCallback('Car Tracker', self.mouse_callback)
        
        # Start in interactive setup mode
        self.start_interactive_setup()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                # Loop video if it's a file
                if self.video_path:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                else:
                    break
                
            self.frame_count += 1
            
            # Process every frame for detection and tracking
            processed_frame = self.process_frame(frame)
            
            # Add FPS counter
            elapsed_time = time.time() - start_time
            current_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
            
            # Add status information
            cv2.putText(processed_frame, f"Frame: {self.frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(processed_frame, f"FPS: {current_fps:.1f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(processed_frame, f"Cars: {len(self.car_tracks)}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(processed_frame, f"Line Distance: {self.line_distance_meters}m", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add performance mode info
            cv2.putText(processed_frame, f"Original Timing: ON", (10, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Add interactive mode instructions
            if self.interactive_mode:
                if self.current_setup_line == 0:
                    cv2.putText(processed_frame, "Click to set Line 1 (Start Line)", (10, self.frame_height - 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                elif self.current_setup_line == 1:
                    cv2.putText(processed_frame, "Click to set Line 2 (Finish Line)", (10, self.frame_height - 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(processed_frame, "Press 'r' to reset, 'q' to quit", (10, self.frame_height - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                cv2.putText(processed_frame, "Press 'i' for interactive setup, 'r' to reset lines, 'q' to quit", 
                           (10, self.frame_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow('Car Tracker', processed_frame)
            
            # Always maintain original video timing
            if self.video_path and self.fps > 0:
                # Calculate delay to maintain original video speed
                delay = int(1000 / self.fps)  # Convert to milliseconds
                key = cv2.waitKey(delay) & 0xFF
            else:
                # For camera input, use minimal delay
                key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset lines to default positions
                self.line1_y_ratio = 0.4
                self.line2_y_ratio = 0.6
                self._setup_virtual_lines()
                print("✓ Lines reset to default positions")
            elif key == ord('i') and not self.interactive_mode:
                # Start interactive setup
                self.start_interactive_setup()
            elif key == ord('s') and self.interactive_mode:
                # Skip to next frame (useful during setup)
                continue
                
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
    parser.add_argument('--line-distance', type=float, default=10.0, 
                       help='Distance between virtual lines in meters (default: 10.0)')
    parser.add_argument('--line1-pos', type=float, default=0.4,
                       help='Position of first line as ratio of frame height (0.0-1.0, default: 0.4)')
    parser.add_argument('--line2-pos', type=float, default=0.6,
                       help='Position of second line as ratio of frame height (0.0-1.0, default: 0.6)')
    parser.add_argument('--no-interactive', action='store_true',
                       help='Disable interactive line setup mode')
    parser.add_argument('--performance-mode', choices=['fast', 'balanced', 'accurate'], 
                       default='balanced', help='Performance mode (default: balanced)')
    parser.add_argument('--no-plate-detection', action='store_true',
                       help='Disable number plate detection for better performance')
    parser.add_argument('--enable-haar', action='store_true',
                       help='Enable Haar cascade detection (slower but more accurate)')
    
    args = parser.parse_args()
    
    try:
        tracker = CarTracker(video_path=args.video, camera_index=args.camera)
        
        # Customize virtual lines if specified
        if args.line_distance != 10.0:
            tracker.set_line_distance(args.line_distance)
        if args.line1_pos != 0.4 or args.line2_pos != 0.6:
            tracker.set_line_positions(args.line1_pos, args.line2_pos)
        
        # Performance tuning - simplified
        if args.performance_mode == 'fast':
            tracker.enable_plate_detection = False
            tracker.plate_detection_frequency = 30
            tracker.enable_haar_cascade = False
            print("🚀 Fast performance mode enabled")
        elif args.performance_mode == 'accurate':
            tracker.enable_plate_detection = True
            tracker.plate_detection_frequency = 5
            tracker.enable_haar_cascade = True
            print("🎯 Accurate performance mode enabled")
        else:  # balanced
            tracker.enable_plate_detection = True
            tracker.plate_detection_frequency = 10
            tracker.enable_haar_cascade = False
            print("⚖️  Balanced performance mode enabled")
        
        # Override with specific flags
        if args.no_plate_detection:
            tracker.enable_plate_detection = False
            print("📱 Number plate detection disabled")
        if args.enable_haar:
            tracker.enable_haar_cascade = True
            print("🔍 Haar cascade detection enabled")
        
        # Disable interactive mode if requested
        if args.no_interactive:
            tracker.interactive_mode = False
            print("Interactive mode disabled - using preset line positions")
        
        tracker.run()
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 