#!/bin/bash
# Activation script for Car Speed Detection project

echo "Activating Car Speed Detection environment..."
source venv/bin/activate

echo "Environment activated! You can now run:"
echo "  python car_tracker.py --video your_video.mp4"
echo "  python car_tracker.py --camera 0"
echo "  python demo.py --interactive"
echo "  python calibration.py --video your_video.mp4"
echo ""
echo "To deactivate, run: deactivate" 