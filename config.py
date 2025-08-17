# Configuration file for Car Tracking and Number Plate Recognition

# Video Source Configuration
VIDEO_SOURCE = {
    'default_camera': 0,  # Default camera index
    'video_path': None,   # Path to video file (None for camera)
}

# Car Detection Parameters
CAR_DETECTION = {
    'min_area': 1000,           # Minimum contour area for car detection
    'max_area': 50000,          # Maximum contour area for car detection
    'aspect_ratio_min': 1.2,    # Minimum width/height ratio for cars
    'aspect_ratio_max': 3.0,    # Maximum width/height ratio for cars
    'haar_scale_factor': 1.1,   # Haar cascade scale factor
    'haar_min_neighbors': 5,    # Haar cascade minimum neighbors
    'haar_min_size': (30, 30),  # Haar cascade minimum size
}

# Background Subtraction Parameters
BACKGROUND_SUBTRACTION = {
    'history': 500,             # Number of frames for background model
    'var_threshold': 50,        # Variance threshold for background subtraction
    'detect_shadows': False,    # Whether to detect shadows
}

# Number Plate Detection Parameters
PLATE_DETECTION = {
    'min_area': 1000,           # Minimum plate area
    'max_area': 50000,          # Maximum plate area
    'aspect_ratio_min': 2.0,    # Minimum plate aspect ratio
    'aspect_ratio_max': 5.0,    # Maximum plate aspect ratio
    
    # HSV color range for white plates
    'white_lower': [0, 0, 200],
    'white_upper': [180, 30, 255],
    
    # OCR Configuration
    'ocr_config': '--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
    'min_plate_length': 4,      # Minimum characters for valid plate
}

# Tracking Parameters
TRACKING = {
    'max_disappeared': 30,      # Maximum frames a car can disappear
    'min_distance': 50,         # Minimum distance to consider as same car
    'speed_history_length': 10, # Number of speed measurements to keep
}

# Speed Estimation Parameters
SPEED_ESTIMATION = {
    'pixel_to_meter_ratio': 0.1,  # Calibration factor (m/pixel)
    'fps': 30,                     # Assumed FPS (will be updated from video)
}

# Display Parameters
DISPLAY = {
    'bbox_color': (0, 255, 0),     # BGR color for car bounding boxes
    'plate_color': (0, 0, 255),    # BGR color for plate bounding boxes
    'text_color': (255, 255, 255), # BGR color for text
    'line_thickness': 2,           # Line thickness for drawings
    'font_scale': 0.6,             # Font scale for text
    'font_thickness': 2,           # Font thickness for text
}

# Processing Parameters
PROCESSING = {
    'gaussian_blur_kernel': (5, 5),    # Gaussian blur kernel size
    'morphology_kernel': (3, 3),       # Morphology kernel size
    'plate_resize_factor': 2,          # Factor to resize plate for OCR
}

# File Paths
PATHS = {
    'car_cascade': 'haarcascade_cars.xml',  # Haar cascade file for car detection
    'output_dir': 'output',                 # Output directory for processed videos
    'log_file': 'car_tracker.log',          # Log file path
} 