import cv2
import numpy as np
import time
import math
import argparse
from ultralytics import YOLO

class VirtualLine:
    def __init__(self, x1, y1, x2, y2, name="Line"):
        """Initialize a virtual line for speed detection"""
        self.x1, self.y1 = x1, y1
        self.x2, self.y2 = x2, y2
        self.name = name
        self.crossed_cars = {}  # Track cars that crossed this line
        self.max_crossing_history = 50  # Maximum number of crossing records to keep
        
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
    
    def cleanup_old_crossings(self):
        """Clean up old crossing records to prevent memory bloat"""
        if len(self.crossed_cars) <= self.max_crossing_history:
            return
        
        # Remove oldest crossing records
        current_time = time.time()
        crossing_times = []
        
        for car_id, crossing_data in self.crossed_cars.items():
            if crossing_data['cross_time']:
                crossing_times.append((car_id, crossing_data['cross_time']))
        
        # Sort by crossing time (oldest first)
        crossing_times.sort(key=lambda x: x[1])
        
        # Remove oldest records
        records_to_remove = len(self.crossed_cars) - self.max_crossing_history
        removed_count = 0
        
        for car_id, cross_time in crossing_times:
            if removed_count >= records_to_remove:
                break
            
            del self.crossed_cars[car_id]
            removed_count += 1
        
        if removed_count > 0:
            if hasattr(self, 'debug_logging') and self.debug_logging:
                print(f"🧹 Cleaned up {removed_count} old crossing records from {self.name}")

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
        self.max_track_history = 100  # Maximum number of tracks to keep in memory
        self.cleanup_interval = 60  # Cleanup every N frames
        
        # Speed estimation parameters
        self.fps = 30  # Assumed FPS, will be updated from video
        
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
        self.enable_plate_detection = False  # Disabled for performance
        self.frame_count = 0  # Track frame count for processing decisions
        self.confidence_threshold = 0.5  # YOLO detection confidence threshold
        
        # GPU acceleration and timing optimization
        self.use_gpu_acceleration = False  # Disabled by default for better compatibility
        self.frame_timing = True  # Maintain original video timing
        self.last_frame_time = 0  # For precise timing
        self.performance_monitoring = False  # Disable performance monitoring by default for performance
        self.debug_logging = False  # Disable debug logging by default for performance
        
        # Car detection with YOLO
        try:
            # Load YOLO model for car detection
            self.yolo_model = YOLO('yolov8n.pt')  # Use nano model for speed
            print("✓ YOLO model loaded (yolov8n.pt)")
        except Exception as e:
            print(f"Error: Could not load YOLO model: {e}")
            print("Please install ultralytics: pip install ultralytics")
            exit(1)
        
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
        
        # Validate FPS
        if self.fps <= 0:
            self.fps = 30.0  # Default FPS if not detected
            print(f"⚠️  FPS not detected, using default: {self.fps}")
        else:
            print(f"✓ Video FPS: {self.fps:.2f}")
        
        print(f"✓ Video dimensions: {self.frame_width}x{self.frame_height}")
        
        # Calculate frame interval for precise timing
        self.frame_interval = 1.0 / self.fps if self.fps > 0 else 1.0 / 30.0
        

        
    def detect_cars(self, frame):
        """Detect cars in the frame using YOLO model"""
        cars = []
        raw_detections = 0
        filtered_detections = 0
        
        # Use YOLO model for car detection
        if self.yolo_model is not None:
            try:
                # Run YOLO inference on the frame
                results = self.yolo_model(frame, verbose=False)
                
                # Process results
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = box.conf[0].cpu().numpy()
                            class_id = int(box.cls[0].cpu().numpy())
                            
                            # Filter for cars (class 2 in COCO dataset) and high confidence
                            if class_id == 2 and confidence > self.confidence_threshold:  # Car class with configurable confidence
                                x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                                
                                # Additional size validation
                                if w >= 30 and h >= 30 and w <= 800 and h <= 800:
                                    # Check aspect ratio (cars are typically wider than tall)
                                    aspect_ratio = w / float(h)
                                    if 0.8 <= aspect_ratio <= 4.0:  # Flexible car proportions
                                        cars.append((x, y, w, h))
                
                # Debug info (uncomment to see detection counts)
                # print(f"🚗 YOLO detected {len(cars)} cars")
                    
            except Exception as e:
                print(f"Error in YOLO detection: {e}")
                pass
        
        return cars
    

        
    def cleanup_old_tracks(self):
        """Clean up old tracks to prevent memory bloat and improve performance"""
        if len(self.car_tracks) <= self.max_track_history:
            return
        
        # Sort tracks by last seen time (most recent first)
        current_time = time.time()
        track_times = []
        
        for track_id, track_info in self.car_tracks.items():
            # Calculate time since last seen
            if track_info['disappeared'] > 0:
                # Track is currently disappeared
                last_seen = current_time - (track_info['disappeared'] / self.fps)
            else:
                # Track is currently visible
                last_seen = current_time
            
            track_times.append((track_id, last_seen, track_info['disappeared']))
        
        # Sort by last seen time (oldest first)
        track_times.sort(key=lambda x: x[1])
        
        # Remove oldest tracks until we're under the limit
        tracks_to_remove = len(self.car_tracks) - self.max_track_history
        removed_count = 0
        
        for track_id, last_seen, disappeared in track_times:
            if removed_count >= tracks_to_remove:
                break
            
            # Remove the track
            del self.car_tracks[track_id]
            removed_count += 1
            
            # Also clean up from virtual lines
            for line in self.virtual_lines:
                if track_id in line.crossed_cars:
                    del line.crossed_cars[track_id]
        
        if removed_count > 0:
            if self.debug_logging:
                print(f"🧹 Cleaned up {removed_count} old tracks (memory optimization)")
        
        # Also cleanup virtual lines
        for line in self.virtual_lines:
            line.cleanup_old_crossings()
    
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
                    if self.debug_logging:
                        print(f"🚗 Car {car_id} crossed Line 1 at {frame_time:.2f}s")
            
            # Check line 2 crossing
            if track_info['line1_crossed'] and not track_info['line2_crossed']:
                if self.virtual_lines[1].check_crossing(car_center, car_id, frame_time):
                    track_info['line2_crossed'] = True
                    track_info['line2_time'] = frame_time
                    
                    # Calculate speed
                    speed = self.calculate_speed_from_lines(car_id)
                    track_info['speed'] = speed
                    if self.debug_logging:
                        print(f"🏁 Car {car_id} crossed Line 2 at {frame_time:.2f}s")
                        print(f"⚡ Car {car_id} speed: {speed:.1f} km/h")
                    
            # Debug: Show line crossing status
            if self.debug_logging and track_info['line1_crossed'] and not track_info['line2_crossed']:
                print(f"🔄 Car {car_id} waiting to cross Line 2...")
        
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
            
            # Draw bounding box
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw track ID
            cv2.putText(frame, f"ID: {track_id}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Draw speed (always show, even if 0)
            speed_text = f"Speed: {speed:.1f} km/h" if speed > 0 else "Speed: -- km/h"
            speed_color = (255, 0, 0) if speed > 0 else (128, 128, 128)  # Red if speed > 0, gray if 0
            cv2.putText(frame, speed_text, (x, y + h + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, speed_color, 2)
        
        # Draw virtual lines
        for line in self.virtual_lines:
            line.draw(frame)
        
        return frame
        
    def run(self):
        """Main loop for processing video"""
        self.frame_count = 0
        start_time = time.time()
        
        # Setup mouse callback for interactive line setup
        cv2.namedWindow('Car Tracker', cv2.WINDOW_NORMAL)
        
        cv2.setMouseCallback('Car Tracker', self.mouse_callback)
        
        # Start in interactive setup mode
        self.start_interactive_setup()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                # Video ended, exit the program
                break
                
            self.frame_count += 1
            
            # Process every frame for detection and tracking
            processed_frame = self.process_frame(frame)
            
            # Periodic cleanup to prevent memory bloat
            if self.frame_count % self.cleanup_interval == 0:
                self.cleanup_old_tracks()
            
            # Calculate timing information
            current_time = time.time()
            elapsed_time = current_time - start_time
            current_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
            
            # Add FPS counter
            cv2.putText(processed_frame, f"Frame: {self.frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(processed_frame, f"FPS: {current_fps:.1f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(processed_frame, f"Cars: {len(self.car_tracks)}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(processed_frame, f"Line Distance: {self.line_distance_meters}m", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add speed tracking info
            cars_with_speed = sum(1 for track in self.car_tracks.values() if track['speed'] > 0)
            cv2.putText(processed_frame, f"Cars with Speed: {cars_with_speed}/{len(self.car_tracks)}", (10, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
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
            
            # Precise timing control for video playback
            if self.video_path and self.frame_timing:
                # Calculate when the next frame should be displayed
                target_frame_time = start_time + (self.frame_count * self.frame_interval)
                current_time = time.time()
                
                # Calculate delay needed
                if target_frame_time > current_time:
                    delay_ms = int((target_frame_time - current_time) * 1000)
                    delay_ms = max(1, min(delay_ms, 100))  # Clamp between 1-100ms
                else:
                    delay_ms = 1  # Minimal delay if we're behind
                
                key = cv2.waitKey(delay_ms) & 0xFF
            else:
                # For camera input, use minimal delay
                key = cv2.waitKey(1) & 0xFF
            
            # Handle key presses
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
            elif key == ord('g'):
                # Toggle GPU acceleration
                self.use_gpu_acceleration = not self.use_gpu_acceleration
                print(f"GPU acceleration: {'ON' if self.use_gpu_acceleration else 'OFF'}")
            elif key == ord('t'):
                # Toggle frame timing
                self.frame_timing = not self.frame_timing
                print(f"Frame timing: {'ON' if self.frame_timing else 'OFF'}")
                
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
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='YOLO detection confidence threshold (0.0-1.0, default: 0.5)')
    parser.add_argument('--no-gpu', action='store_true',
                       help='Disable GPU acceleration for better compatibility')
    parser.add_argument('--no-timing', action='store_true',
                       help='Disable frame timing for maximum performance')
    parser.add_argument('--max-tracks', type=int, default=100,
                       help='Maximum number of car tracks to keep in memory (default: 100)')
    parser.add_argument('--cleanup-interval', type=int, default=60,
                       help='Cleanup old tracks every N frames (default: 60)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging (may impact performance)')
    parser.add_argument('--monitor', action='store_true',
                       help='Enable performance monitoring (may impact performance)')

    
    args = parser.parse_args()
    
    try:
        tracker = CarTracker(video_path=args.video, camera_index=args.camera)
        
        # Customize virtual lines if specified
        if args.line_distance != 10.0:
            tracker.set_line_distance(args.line_distance)
        if args.line1_pos != 0.4 or args.line2_pos != 0.6:
            tracker.set_line_positions(args.line1_pos, args.line2_pos)
        
        # Performance tuning - simplified
        if args.confidence != 0.5:
            tracker.confidence_threshold = args.confidence
            print(f"🎯 YOLO confidence threshold set to {args.confidence}")
        
        # Disable interactive mode if requested
        if args.no_interactive:
            tracker.interactive_mode = False
            print("Interactive mode disabled - using preset line positions")
        
        # Disable GPU acceleration if requested
        if args.no_gpu:
            tracker.use_gpu_acceleration = False
            print("GPU acceleration disabled.")
        
        # Disable frame timing if requested
        if args.no_timing:
            tracker.frame_timing = False
            print("Frame timing disabled.")
        
        # Set new parameters
        tracker.max_track_history = args.max_tracks
        tracker.cleanup_interval = args.cleanup_interval
        print(f"🎯 Car tracker parameters updated:")
        print(f"  Max Tracks: {tracker.max_track_history}")
        print(f"  Cleanup Interval: {tracker.cleanup_interval} frames")
        
        # Enable debug logging if requested
        if args.debug:
            tracker.debug_logging = True
            print("🐛 Debug logging enabled (may impact performance)")
        
        # Enable performance monitoring if requested
        if args.monitor:
            tracker.performance_monitoring = True
            print("📊 Performance monitoring enabled (may impact performance)")
        
        tracker.run()
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 