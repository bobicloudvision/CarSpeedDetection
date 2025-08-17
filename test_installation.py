#!/usr/bin/env python3
"""
Test script to verify OpenCV and other dependencies are properly installed
"""

import sys
import importlib

def test_import(module_name, package_name=None):
    """Test if a module can be imported"""
    try:
        if package_name:
            module = importlib.import_module(module_name)
            print(f"✓ {package_name} imported successfully")
            return True
        else:
            module = importlib.import_module(module_name)
            print(f"✓ {module_name} imported successfully")
            return True
    except ImportError as e:
        print(f"✗ Failed to import {module_name}: {e}")
        return False

def test_opencv():
    """Test OpenCV functionality"""
    try:
        import cv2
        print(f"✓ OpenCV version: {cv2.__version__}")
        
        # Test basic OpenCV functions
        img = cv2.imread("test_image.jpg") if cv2.os.path.exists("test_image.jpg") else None
        if img is not None:
            print("✓ OpenCV image reading works")
        else:
            print("! OpenCV image reading test skipped (no test image)")
        
        # Test video capture
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✓ OpenCV camera access works")
            cap.release()
        else:
            print("! OpenCV camera access test skipped (no camera available)")
            
        return True
    except Exception as e:
        print(f"✗ OpenCV test failed: {e}")
        return False

def test_tesseract():
    """Test Tesseract OCR"""
    try:
        import pytesseract
        print("✓ pytesseract imported successfully")
        
        # Test if tesseract is available
        try:
            version = pytesseract.get_tesseract_version()
            print(f"✓ Tesseract version: {version}")
            return True
        except Exception as e:
            print(f"! Tesseract not found in PATH: {e}")
            print("  Please install Tesseract OCR and ensure it's in your PATH")
            return False
            
    except ImportError as e:
        print(f"✗ pytesseract import failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing Car Tracker Dependencies...")
    print("=" * 40)
    
    tests_passed = 0
    total_tests = 0
    
    # Test basic Python modules
    basic_modules = [
        ('numpy', 'NumPy'),
        ('PIL', 'Pillow (PIL)'),
        ('cv2', 'OpenCV'),
        ('pytesseract', 'pytesseract'),
        ('imutils', 'imutils'),
        ('skimage', 'scikit-image'),
        ('matplotlib', 'matplotlib')
    ]
    
    for module, name in basic_modules:
        total_tests += 1
        if test_import(module, name):
            tests_passed += 1
    
    print("\n" + "=" * 40)
    
    # Test OpenCV specifically
    total_tests += 1
    if test_opencv():
        tests_passed += 1
    
    # Test Tesseract specifically
    total_tests += 1
    if test_tesseract():
        tests_passed += 1
    
    print("\n" + "=" * 40)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! You're ready to run the car tracker.")
        print("\nNext steps:")
        print("1. Run: python car_tracker.py --video your_video.mp4")
        print("2. Or run: python car_tracker.py --camera 0")
        print("3. For calibration: python calibration.py --video your_video.mp4")
    else:
        print("❌ Some tests failed. Please install missing dependencies:")
        print("   pip install -r requirements.txt")
        print("\nFor Tesseract OCR:")
        print("   macOS: brew install tesseract")
        print("   Ubuntu: sudo apt-get install tesseract-ocr")
        print("   Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")

if __name__ == "__main__":
    main() 