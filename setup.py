#!/usr/bin/env python3
"""
Setup script for Car Speed Detection project
"""

import os
import sys
import subprocess
import platform

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print(f"✗ Python {version.major}.{version.minor} is not supported. Please use Python 3.7+")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_python_dependencies():
    """Install Python dependencies"""
    print("\nInstalling Python dependencies...")
    
    # Upgrade pip first
    if not run_command("pip install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing requirements"):
        return False
    
    return True

def install_tesseract():
    """Install Tesseract OCR based on platform"""
    system = platform.system().lower()
    
    print(f"\nInstalling Tesseract OCR for {system}...")
    
    if system == "darwin":  # macOS
        if not run_command("brew --version", "Checking Homebrew"):
            print("Homebrew not found. Installing...")
            install_command = '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
            if not run_command(install_command, "Installing Homebrew"):
                return False
        
        if not run_command("brew install tesseract", "Installing Tesseract"):
            return False
            
    elif system == "linux":
        # Try different package managers
        if os.path.exists("/etc/debian_version"):  # Debian/Ubuntu
            if not run_command("sudo apt-get update", "Updating package list"):
                return False
            if not run_command("sudo apt-get install -y tesseract-ocr", "Installing Tesseract"):
                return False
        elif os.path.exists("/etc/redhat-release"):  # RHEL/CentOS/Fedora
            if not run_command("sudo yum install -y tesseract", "Installing Tesseract"):
                return False
        else:
            print("Unsupported Linux distribution. Please install Tesseract manually.")
            return False
            
    elif system == "windows":
        print("Please install Tesseract manually from: https://github.com/UB-Mannheim/tesseract/wiki")
        print("After installation, ensure it's added to your PATH")
        return True
        
    else:
        print(f"Unsupported operating system: {system}")
        return False
    
    return True

def create_output_directory():
    """Create output directory for processed videos"""
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✓ Created output directory: {output_dir}")
    else:
        print(f"✓ Output directory already exists: {output_dir}")

def test_installation():
    """Test if everything is working"""
    print("\nTesting installation...")
    
    if not run_command("python test_installation.py", "Running installation test"):
        return False
    
    return True

def main():
    """Main setup function"""
    print("Car Speed Detection - Setup Script")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        print("Setup cannot continue. Please upgrade Python.")
        return
    
    # Install Python dependencies
    if not install_python_dependencies():
        print("Failed to install Python dependencies.")
        return
    
    # Install Tesseract
    if not install_tesseract():
        print("Failed to install Tesseract OCR.")
        return
    
    # Create output directory
    create_output_directory()
    
    # Test installation
    if not test_installation():
        print("Installation test failed. Please check the errors above.")
        return
    
    print("\n" + "=" * 40)
    print("🎉 Setup completed successfully!")
    print("\nYou can now:")
    print("1. Run camera demo: python demo.py --mode camera")
    print("2. Run video demo: python demo.py --mode video --video your_video.mp4")
    print("3. Run calibration: python demo.py --mode calibration --video your_video.mp4")
    print("4. Use interactive mode: python demo.py --interactive")
    print("\nFor more information, see README.md")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSetup interrupted by user")
    except Exception as e:
        print(f"Setup failed with error: {e}")
        sys.exit(1) 