#!/usr/bin/env python3
"""
r/vision Setup Script
Automated installation script for all r/vision dependencies.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Error: {e}")
        if e.stdout:
            print(f"   Output: {e.stdout}")
        if e.stderr:
            print(f"   Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+")
        return False

def main():
    """Main setup function."""
    print("🚀 r/vision Setup Script")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        print("\n❌ Please upgrade Python to version 3.8 or higher")
        return 1
    
    # Get current directory
    current_dir = Path(__file__).parent
    requirements_file = current_dir / "requirements.txt"
    
    if not requirements_file.exists():
        print(f"❌ requirements.txt not found in {current_dir}")
        return 1
    
    print(f"\n📁 Working directory: {current_dir}")
    print(f"📋 Requirements file: {requirements_file}")
    
    # Upgrade pip first
    if not run_command(f"{sys.executable} -m pip install --upgrade pip", 
                      "Upgrading pip"):
        print("⚠️  Pip upgrade failed, continuing anyway...")
    
    # Install requirements
    install_cmd = f"{sys.executable} -m pip install -r {requirements_file}"
    if not run_command(install_cmd, "Installing r/vision dependencies"):
        print("\n❌ Failed to install dependencies")
        print("💡 Try running manually:")
        print(f"   {install_cmd}")
        return 1
    
    print("\n🧪 Running installation test...")
    test_cmd = f"{sys.executable} test_installation.py"
    if run_command(test_cmd, "Testing installation"):
        print("\n🎉 Setup completed successfully!")
        print("\n📚 Quick start:")
        print("1. Place a video file in this directory")
        print("2. Run: python r_vision.py your_video.mp4")
        print("3. Check the generated output.mp4")
        return 0
    else:
        print("\n⚠️  Installation test failed")
        print("💡 Try running the test manually:")
        print(f"   {test_cmd}")
        return 1

if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\n⏹️  Setup interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        exit(1)
