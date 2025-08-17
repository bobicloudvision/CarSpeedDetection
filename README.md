# Car Speed Detection with Number Plate Recognition

A comprehensive OpenCV-based system for tracking cars in video streams, detecting number plates, and estimating vehicle speeds using computer vision and OCR techniques.

## Features

- **Car Detection**: Multi-method car detection using background subtraction and Haar cascades
- **Object Tracking**: Robust car tracking with unique ID assignment
- **Number Plate Recognition**: Automatic detection and OCR of license plates
- **Speed Estimation**: Real-time speed calculation based on pixel movement
- **Multi-source Input**: Support for video files and live camera feeds
- **Calibration Tools**: Built-in calibration for accurate speed measurements

## Project Structure

```
CarSpeedDetection/
├── car_tracker.py          # Main car tracking application
├── calibration.py          # Calibration tool for speed estimation
├── config.py              # Configuration parameters
├── test_installation.py   # Dependency testing script
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Requirements

### System Requirements
- Python 3.7+
- OpenCV 4.8+
- Tesseract OCR engine

### Python Dependencies
- opencv-python
- opencv-contrib-python
- numpy
- pytesseract
- Pillow
- imutils
- scikit-image
- matplotlib

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd CarSpeedDetection
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Install Tesseract OCR

#### macOS
```bash
brew install tesseract
```

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

#### Windows
Download and install from: https://github.com/UB-Mannheim/tesseract/wiki

### 4. Test Installation
```bash
python test_installation.py
```

## Usage

### Basic Car Tracking

#### From Video File
```bash
python car_tracker.py --video path/to/your/video.mp4
```

#### From Camera
```bash
python car_tracker.py --camera 0
```

### Calibration for Speed Estimation

Before getting accurate speed measurements, calibrate the system:

```bash
python calibration.py --video path/to/calibration_video.mp4
```

1. Click two points on the video that represent a known distance
2. Enter the actual distance in meters
3. Use the calculated ratio in `config.py`

### Configuration

Edit `config.py` to adjust parameters for your specific use case:

- Car detection sensitivity
- Number plate detection parameters
- Speed estimation calibration
- Display settings

## How It Works

### 1. Car Detection
The system uses two complementary methods:
- **Background Subtraction**: Detects moving objects by comparing frames
- **Haar Cascade**: Uses pre-trained classifiers for car detection

### 2. Object Tracking
- Assigns unique IDs to detected cars
- Tracks cars across frames using distance-based matching
- Handles occlusions and temporary disappearances

### 3. Number Plate Recognition
- Detects white regions within car bounding boxes
- Filters candidates based on aspect ratio and size
- Uses Tesseract OCR to read plate text

### 4. Speed Estimation
- Calculates pixel movement between frames
- Converts to real-world units using calibration
- Provides speed in km/h

## Key Parameters

### Car Detection
- `min_area`: Minimum contour area (default: 1000)
- `aspect_ratio_min/max`: Width/height ratio range (default: 1.2-3.0)

### Number Plate Detection
- `min_plate_area`: Minimum plate area (default: 1000)
- `aspect_ratio_min/max`: Plate aspect ratio range (default: 2.0-5.0)

### Tracking
- `max_disappeared`: Frames before removing track (default: 30)
- `min_distance`: Distance threshold for matching (default: 50)

### Speed Estimation
- `pixel_to_meter_ratio`: Calibration factor (default: 0.1)

## Troubleshooting

### Common Issues

1. **Tesseract not found**
   - Ensure Tesseract is installed and in your PATH
   - On Windows, set `pytesseract.pytesseract.tesseract_cmd` to the executable path

2. **Poor car detection**
   - Adjust `min_area` and `max_area` in config
   - Modify aspect ratio constraints
   - Check video quality and lighting

3. **Inaccurate speed estimation**
   - Use calibration tool to get proper `pixel_to_meter_ratio`
   - Ensure camera is stationary
   - Check video FPS settings

4. **Number plate not detected**
   - Adjust HSV color ranges for your lighting conditions
   - Modify plate area constraints
   - Check OCR configuration

### Performance Tips

- Use GPU-accelerated OpenCV for better performance
- Reduce video resolution for faster processing
- Adjust detection parameters based on your specific scenario
- Use SSD storage for video files

## Customization

### Adding New Detection Methods
Extend the `detect_cars` method in `CarTracker` class:

```python
def detect_cars(self, frame):
    cars = []
    
    # Your custom detection logic here
    # Add detected cars to the list: cars.append((x, y, w, h))
    
    return cars
```

### Custom Number Plate Detection
Modify the `detect_number_plate` method for different plate types:

```python
def detect_number_plate(self, frame, car_bbox):
    # Custom plate detection logic
    # Return list of (x, y, w, h) tuples
    pass
```

## Output

The system provides real-time visualization including:
- Car bounding boxes with unique IDs
- Number plate bounding boxes
- Speed estimates in km/h
- Frame counter and FPS
- Car count

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenCV community for computer vision tools
- Tesseract OCR for text recognition
- Contributors and testers

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review configuration parameters
3. Test with the provided test script
4. Open an issue on the repository

---

**Note**: This system is designed for educational and research purposes. Ensure compliance with local laws and regulations when using for traffic monitoring or law enforcement applications. 