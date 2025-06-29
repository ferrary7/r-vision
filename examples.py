#!/usr/bin/env python3
"""
r/vision Examples
Demonstration script showing various use cases and configurations.
"""

import os
import sys
from pathlib import Path

def print_example(title, description, command):
    """Print a formatted example."""
    print(f"\n📋 {title}")
    print("-" * (len(title) + 4))
    print(f"Description: {description}")
    print(f"Command: {command}")
    print()

def main():
    """Display example usage scenarios."""
    print("🎯 r/vision Usage Examples")
    print("=" * 50)
    
    # Basic examples
    print_example(
        "Basic Processing",
        "Process a video with default settings (YOLO + Pose)",
        "python r_vision.py input.mp4"
    )
    
    print_example(
        "Custom Output",
        "Specify custom output filename",
        "python r_vision.py input.mp4 --output my_processed_video.mp4"
    )
    
    # Feature toggles
    print_example(
        "Object Detection Only",
        "Disable pose detection, only run YOLO",
        "python r_vision.py input.mp4 --no-pose"
    )
    
    print_example(
        "Pose Detection Only",
        "Disable object detection, only run pose estimation",
        "python r_vision.py input.mp4 --no-yolo"
    )
    
    # Confidence tuning
    print_example(
        "High Precision Mode",
        "Use high confidence thresholds for fewer, more accurate detections",
        "python r_vision.py input.mp4 --yolo-confidence 0.8 --pose-confidence 0.7"
    )
    
    print_example(
        "High Recall Mode",
        "Use low confidence thresholds for more detections",
        "python r_vision.py input.mp4 --yolo-confidence 0.3 --pose-confidence 0.3"
    )
    
    # Advanced combinations
    print_example(
        "Sports Analysis",
        "Optimized for sports videos with multiple people",
        "python r_vision.py sports_game.mp4 --yolo-confidence 0.6 --pose-confidence 0.4"
    )
    
    print_example(
        "Security Footage",
        "High precision for security camera analysis",
        "python r_vision.py security_cam.mp4 --yolo-confidence 0.7 --no-pose"
    )
    
    print_example(
        "Fitness Video",
        "Focus on pose detection for fitness analysis",
        "python r_vision.py workout.mp4 --no-yolo --pose-confidence 0.6"
    )
    
    print_example(
        "Batch Processing",
        "Process multiple videos with a simple loop",
        """for video in *.mp4; do
    python r_vision.py "$video" --output "processed_$video"
done"""
    )
    
    # Performance examples
    print("\n⚡ Performance Optimization Tips")
    print("=" * 35)
    
    tips = [
        "Use --no-yolo or --no-pose to disable unused features",
        "Increase confidence thresholds to reduce processing load",
        "Process lower resolution videos for faster results",
        "Close other applications to free up system resources",
        "Use SSD storage for better I/O performance"
    ]
    
    for i, tip in enumerate(tips, 1):
        print(f"{i}. {tip}")
    
    # File format support
    print("\n📁 Supported Video Formats")
    print("=" * 30)
    
    formats = [
        ".mp4 (Recommended)",
        ".avi",
        ".mov",
        ".mkv",
        ".wmv",
        ".flv"
    ]
    
    for fmt in formats:
        print(f"✅ {fmt}")
    
    # Sample workflow
    print("\n🔄 Sample Workflow")
    print("=" * 20)
    
    workflow_steps = [
        "1. Install dependencies: python setup.py",
        "2. Test installation: python test_installation.py",
        "3. Process video: python r_vision.py input.mp4",
        "4. Check output: output.mp4",
        "5. Adjust settings if needed and reprocess"
    ]
    
    for step in workflow_steps:
        print(step)
    
    print("\n💡 For more options, run: python r_vision.py --help")
    print("📖 Full documentation: README.md")

if __name__ == "__main__":
    main()
