#!/usr/bin/env python3
"""
Demo script for Car Speed Detection with Number Plate Recognition
"""

import os
import sys
import argparse
from car_tracker import CarTracker
from calibration import CalibrationTool

def demo_camera():
    """Demo with live camera feed"""
    print("Starting camera demo...")
    print("Press 'q' to quit")
    
    try:
        tracker = CarTracker(camera_index=0)
        tracker.run()
    except Exception as e:
        print(f"Camera demo failed: {e}")
        print("Make sure you have a camera connected and accessible")

def demo_video(video_path):
    """Demo with video file"""
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return
    
    print(f"Starting video demo with: {video_path}")
    print("Press 'q' to quit")
    
    try:
        tracker = CarTracker(video_path=video_path)
        tracker.run()
    except Exception as e:
        print(f"Video demo failed: {e}")

def demo_calibration(video_path):
    """Demo calibration tool"""
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return
    
    print(f"Starting calibration demo with: {video_path}")
    print("Click two points to measure distance")
    print("Press 'r' to reset, 'q' to quit")
    
    try:
        calibrator = CalibrationTool(video_path=video_path)
        calibrator.run()
    except Exception as e:
        print(f"Calibration demo failed: {e}")

def show_menu():
    """Show interactive menu"""
    print("\n" + "="*50)
    print("Car Speed Detection Demo")
    print("="*50)
    print("1. Camera Demo (live feed)")
    print("2. Video Demo (from file)")
    print("3. Calibration Demo")
    print("4. Test Installation")
    print("5. Exit")
    print("="*50)

def main():
    parser = argparse.ArgumentParser(description='Car Tracker Demo')
    parser.add_argument('--mode', choices=['camera', 'video', 'calibration'], 
                       help='Demo mode')
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--interactive', action='store_true', 
                       help='Run in interactive mode')
    
    args = parser.parse_args()
    
    if args.interactive or (not args.mode and not args.video):
        # Interactive mode
        while True:
            show_menu()
            try:
                choice = input("Enter your choice (1-5): ").strip()
                
                if choice == '1':
                    demo_camera()
                elif choice == '2':
                    video_path = input("Enter video file path: ").strip()
                    if video_path:
                        demo_video(video_path)
                    else:
                        print("No video path provided")
                elif choice == '3':
                    video_path = input("Enter video file path for calibration: ").strip()
                    if video_path:
                        demo_calibration(video_path)
                    else:
                        print("No video path provided")
                elif choice == '4':
                    os.system('python test_installation.py')
                elif choice == '5':
                    print("Goodbye!")
                    break
                else:
                    print("Invalid choice. Please enter 1-5.")
                    
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                
    else:
        # Command line mode
        if args.mode == 'camera':
            demo_camera()
        elif args.mode == 'video':
            if args.video:
                demo_video(args.video)
            else:
                print("Please provide video path with --video")
        elif args.mode == 'calibration':
            if args.video:
                demo_calibration(args.video)
            else:
                print("Please provide video path with --video")

if __name__ == "__main__":
    main() 