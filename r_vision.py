#!/usr/bin/env python3
"""
r/vision - Advanced Visual Processing Tool
A Python-based tool for object detection and human pose estimation on videos.

Features:
- YOLOv8 object detection with stylized bounding boxes
- MediaPipe human pose detection with custom keypoint connections
- FPS monitoring and performance logging
- Modular design with toggle flags
- Clean, minimalist overlay styling

Author: r/vision Team
Version: 1.0.0
"""

import cv2
import numpy as np
import mediapipe as mp
import time
import argparse
import sys
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
from ultralytics import YOLO

class RVisionProcessor:
    """Main processor class for r/vision video analysis."""
    
    def __init__(self, 
                 enable_yolo: bool = True, 
                 enable_pose: bool = True,
                 confidence_threshold: float = 0.25,  # Lower threshold to detect more objects
                 pose_confidence: float = 0.5):
        """
        Initialize the r/vision processor.
        
        Args:
            enable_yolo: Enable YOLO object detection
            enable_pose: Enable MediaPipe pose detection
            confidence_threshold: YOLO confidence threshold (lowered to detect more objects)
            pose_confidence: MediaPipe pose confidence threshold
        """
        self.enable_yolo = enable_yolo
        self.enable_pose = enable_pose
        self.confidence_threshold = confidence_threshold
        self.pose_confidence = pose_confidence
        
        # Initialize models
        self.yolo_model = None
        self.mp_pose = None
        self.pose_detector = None
        
        # Enhanced color system with more variety (BGR format for OpenCV)
        self.colors = {
            'person': (0, 255, 0),          # Bright Green
            'vehicle': (0, 0, 255),         # Red
            'animal': (255, 0, 255),        # Magenta
            'object': (255, 255, 0),        # Cyan
            'food': (0, 165, 255),          # Orange
            'electronics': (255, 0, 0),     # Blue
            'sports': (128, 0, 128),        # Purple
            'furniture': (0, 255, 255),     # Yellow
            'text': (255, 255, 255),        # White
            'background': (0, 0, 0)         # Black
        }
        
        # Additional varied colors for specific objects (more vibrant and diverse)
        self.specific_colors = {
            # People - Green variations
            'person': (50, 255, 50),
            
            # Vehicles - Red/Orange variations  
            'car': (0, 69, 255),           # Bright Orange-Red
            'truck': (0, 0, 255),          # Pure Red
            'bus': (0, 140, 255),          # Dark Orange
            'motorcycle': (0, 165, 255),   # Orange
            'bicycle': (0, 255, 127),      # Spring Green
            'airplane': (255, 0, 127),     # Deep Pink
            'train': (255, 69, 0),         # Blue-Red
            'boat': (255, 191, 0),         # Deep Sky Blue
            
            # Animals - Bright and varied
            'cat': (255, 20, 147),         # Deep Pink
            'dog': (255, 105, 180),        # Hot Pink  
            'horse': (138, 43, 226),       # Blue Violet
            'cow': (75, 0, 130),           # Indigo
            'sheep': (221, 160, 221),      # Plum
            'bird': (0, 255, 255),         # Yellow
            'elephant': (128, 128, 128),   # Gray
            'bear': (139, 69, 19),         # Saddle Brown
            'zebra': (255, 255, 255),      # White
            'giraffe': (255, 140, 0),      # Dark Orange
            
            # Food - Warm colors
            'banana': (0, 255, 255),       # Yellow
            'apple': (0, 100, 0),          # Dark Green
            'orange': (0, 165, 255),       # Orange
            'pizza': (0, 69, 255),         # Red-Orange
            'cake': (255, 192, 203),       # Pink
            'sandwich': (210, 180, 140),   # Tan
            'bottle': (0, 128, 0),         # Green
            'cup': (139, 0, 139),          # Dark Magenta
            'wine glass': (128, 0, 128),   # Purple
            
            # Electronics - Cool blues and teals
            'laptop': (255, 0, 0),         # Blue
            'tv': (255, 127, 0),           # Azure  
            'cell phone': (255, 255, 0),   # Cyan
            'mouse': (128, 255, 255),      # Light Cyan
            'keyboard': (255, 20, 147),    # Deep Pink
            'remote': (72, 61, 139),       # Dark Slate Blue
            
            # Sports - Energetic colors
            'sports ball': (0, 255, 0),    # Lime
            'frisbee': (255, 69, 0),       # Red-Orange  
            'tennis racket': (255, 255, 0), # Yellow
            'skateboard': (255, 0, 255),   # Magenta
            'surfboard': (255, 140, 0),    # Dark Orange
            'skis': (173, 216, 230),       # Light Blue
            
            # Furniture - Earth tones and pastels
            'chair': (160, 82, 45),        # Saddle Brown
            'couch': (128, 0, 128),        # Purple
            'bed': (255, 182, 193),        # Light Pink
            'dining table': (139, 69, 19), # Saddle Brown
            'potted plant': (34, 139, 34), # Forest Green
            'toilet': (255, 255, 255),     # White
        }
        
        # Object category mapping for color coding (expanded for more YOLO classes)
        self.object_categories = {
            # People
            'person': 'person',
            
            # Vehicles
            'car': 'vehicle', 'truck': 'vehicle', 'bus': 'vehicle', 'motorcycle': 'vehicle', 'bicycle': 'vehicle',
            'airplane': 'vehicle', 'train': 'vehicle', 'boat': 'vehicle',
            
            # Animals
            'cat': 'animal', 'dog': 'animal', 'horse': 'animal', 'sheep': 'animal', 'cow': 'animal',
            'elephant': 'animal', 'bear': 'animal', 'zebra': 'animal', 'giraffe': 'animal', 'bird': 'animal',
            
            # Food
            'banana': 'food', 'apple': 'food', 'sandwich': 'food', 'orange': 'food', 'broccoli': 'food',
            'carrot': 'food', 'hot dog': 'food', 'pizza': 'food', 'donut': 'food', 'cake': 'food',
            'wine glass': 'food', 'cup': 'food', 'fork': 'food', 'knife': 'food', 'spoon': 'food',
            'bowl': 'food', 'bottle': 'food',
            
            # Electronics
            'laptop': 'electronics', 'mouse': 'electronics', 'remote': 'electronics', 'keyboard': 'electronics',
            'cell phone': 'electronics', 'microwave': 'electronics', 'tv': 'electronics', 'oven': 'electronics',
            'toaster': 'electronics', 'refrigerator': 'electronics',
            
            # Sports
            'frisbee': 'sports', 'skis': 'sports', 'snowboard': 'sports', 'sports ball': 'sports',
            'kite': 'sports', 'baseball bat': 'sports', 'baseball glove': 'sports', 'skateboard': 'sports',
            'surfboard': 'sports', 'tennis racket': 'sports',
            
            # Furniture/Home
            'chair': 'furniture', 'couch': 'furniture', 'potted plant': 'furniture', 'bed': 'furniture',
            'dining table': 'furniture', 'toilet': 'furniture', 'sink': 'furniture',
            
            # Other common objects will default to 'object' category
        }
        
        # Performance tracking
        self.frame_count = 0
        self.total_time = 0
        self.fps_history = []
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize YOLO and MediaPipe models."""
        print("🚀 Initializing r/vision models...")
        
        if self.enable_yolo:
            try:
                print("📦 Loading YOLOv8 model...")
                # Try to load with safe globals to avoid PyTorch compatibility issues
                import torch
                torch.serialization.add_safe_globals([
                    'ultralytics.nn.tasks.DetectionModel',
                    'ultralytics.models.yolo.detect.DetectionPredictor'
                ])
                self.yolo_model = YOLO('yolov8n.pt')
                print("✅ YOLOv8 model loaded successfully")
            except Exception as e:
                print(f"❌ Failed to load YOLO model: {e}")
                print("💡 Trying alternative loading method...")
                try:
                    # Alternative: load with weights_only=False
                    import torch
                    original_load = torch.load
                    torch.load = lambda *args, **kwargs: original_load(*args, **kwargs, weights_only=False)
                    self.yolo_model = YOLO('yolov8n.pt')
                    torch.load = original_load
                    print("✅ YOLOv8 model loaded with alternative method")
                except Exception as e2:
                    print(f"❌ Both loading methods failed: {e2}")
                    self.enable_yolo = False
        
        if self.enable_pose:
            try:
                print("🤸 Initializing MediaPipe Pose...")
                self.mp_pose = mp.solutions.pose
                self.pose_detector = self.mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=self.pose_confidence,
                    min_tracking_confidence=self.pose_confidence
                )
                print("✅ MediaPipe Pose initialized successfully")
            except Exception as e:
                print(f"❌ Failed to initialize MediaPipe: {e}")
                self.enable_pose = False
    
    def _get_object_color(self, class_name: str) -> Tuple[int, int, int]:
        """Get color for object - try specific color first, then category color."""
        # Try specific color first for more variety
        if class_name.lower() in self.specific_colors:
            return self.specific_colors[class_name.lower()]
        
        # Fall back to category color
        category = self.object_categories.get(class_name.lower(), 'object')
        return self.colors.get(category, self.colors['object'])

    def _draw_bounding_box(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, 
                          class_name: str, confidence: float, current_fps: float) -> np.ndarray:
        """Draw a single bounding box with thin, sleek styling."""
        color = self._get_object_color(class_name)
        
        # Draw main bounding box - thin and sleek
        box_thickness = 2  # Reduced from 3 to 2 for sleeker look
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)
        
        # Draw refined corner accents - smaller and thinner
        corner_length = 15  # Reduced from 20
        corner_thickness = 3  # Reduced from 5
        
        # Top-left corner
        cv2.line(frame, (x1, y1), (x1 + corner_length, y1), color, corner_thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_length), color, corner_thickness)
        
        # Top-right corner
        cv2.line(frame, (x2, y1), (x2 - corner_length, y1), color, corner_thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_length), color, corner_thickness)
        
        # Bottom-left corner
        cv2.line(frame, (x1, y2), (x1 + corner_length, y2), color, corner_thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_length), color, corner_thickness)
        
        # Bottom-right corner
        cv2.line(frame, (x2, y2), (x2 - corner_length, y2), color, corner_thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_length), color, corner_thickness)
        
        # Prepare compact label - all info in one line
        label = f"{class_name.upper()} | {confidence:.2f} | {current_fps:.1f}fps"
        
        # Use smaller, refined font for sleek appearance
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45  # Slightly smaller for more refined look
        thickness = 1
        
        # Calculate label size
        (label_width, label_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Position label at top-left of bounding box (slightly above)
        label_x = x1
        label_y = y1 - 6  # Reduced gap for tighter spacing
        
        # Ensure label doesn't go off screen
        if label_y - label_height < 0:
            label_y = y1 + label_height + 6  # Put it below the top line instead
        
        # Draw minimal semi-transparent background for text readability
        overlay = frame.copy()
        cv2.rectangle(overlay, 
                     (label_x - 1, label_y - label_height - 1),
                     (label_x + label_width + 1, label_y + 1),
                     (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)  # More subtle background
        
        # Draw the label text with the same color as the box
        cv2.putText(frame, label, (label_x, label_y),
                   font, font_scale, color, thickness)
        
        return frame

    def _draw_yolo_detections(self, frame: np.ndarray, results, current_fps: float) -> np.ndarray:
        """Draw ALL YOLO detection results - detect everything possible in the frame."""
        if not results or len(results) == 0:
            return frame
        
        height, width = frame.shape[:2]
        detection_count = 0
        
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
                
            # Process ALL boxes in this result - be very aggressive
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self.yolo_model.names[class_id]
                
                # Use very low confidence threshold to catch everything
                # Even lower than what user sets - we want to see EVERYTHING
                min_confidence = min(0.1, self.confidence_threshold)
                if confidence < min_confidence:
                    continue
                
                # Ensure coordinates are within frame bounds
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width - 1))
                y2 = max(0, min(y2, height - 1))
                
                # Allow even small boxes (background objects, partial objects)
                box_width = x2 - x1
                box_height = y2 - y1
                
                # Only skip extremely tiny boxes (less than 5x5 pixels)
                if box_width < 5 or box_height < 5:
                    continue
                
                # Draw the bounding box with labels
                frame = self._draw_bounding_box(frame, x1, y1, x2, y2, class_name, confidence, current_fps)
                detection_count += 1
        
        return frame
    
    def _draw_pose_bounding_box(self, frame: np.ndarray, landmarks, current_fps: float) -> np.ndarray:
        """Draw bounding box around detected person pose."""
        if not landmarks:
            return frame
        
        height, width = frame.shape[:2]
        
        # Get all visible landmarks
        visible_points = []
        for landmark in landmarks.landmark:
            if landmark.visibility > self.pose_confidence:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                visible_points.append((x, y))
        
        if len(visible_points) < 5:  # Need at least 5 points for a meaningful bounding box
            return frame
        
        # Calculate bounding box from visible landmarks
        xs = [point[0] for point in visible_points]
        ys = [point[1] for point in visible_points]
        
        x1 = max(0, min(xs) - 20)  # Add some padding
        y1 = max(0, min(ys) - 20)
        x2 = min(width, max(xs) + 20)
        y2 = min(height, max(ys) + 20)
        
        # Draw the person bounding box
        frame = self._draw_bounding_box(frame, x1, y1, x2, y2, "PERSON", 1.0, current_fps)
        
        return frame
    
    def _draw_global_info(self, frame: np.ndarray, fps: float) -> np.ndarray:
        """Draw global information overlay."""
        height, width = frame.shape[:2]
        
        # Prepare text
        info_text = f"r/vision | Global FPS: {fps:.1f} | Frame: {self.frame_count}"
        
        # Calculate text size
        font = cv2.FONT_HERSHEY_DUPLEX
        text_size = cv2.getTextSize(info_text, font, 0.6, 2)[0]
        
        # Position at top center
        x = (width - text_size[0]) // 2
        y = 30
        
        # Draw background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x - 10, y - text_size[1] - 10), 
                     (x + text_size[0] + 10, y + 5), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Draw border
        cv2.rectangle(frame, (x - 10, y - text_size[1] - 10), 
                     (x + text_size[0] + 10, y + 5), (0, 255, 255), 2)
        
        # Draw text
        cv2.putText(frame, info_text, (x, y - 5),
                   font, 0.6, self.colors['text'], 2)
        
        return frame
    
    def _draw_logo(self, frame: np.ndarray) -> np.ndarray:
        """Draw compact r/vision logo/watermark."""
        logo_text = "r/vision"
        font = cv2.FONT_HERSHEY_DUPLEX
        text_size = cv2.getTextSize(logo_text, font, 0.5, 1)[0]
        
        # Position at bottom-right
        x = frame.shape[1] - text_size[0] - 15
        y = frame.shape[0] - 15
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x - 5, y - text_size[1] - 5), 
                     (x + text_size[0] + 5, y + 5), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)
        
        # Draw text
        cv2.putText(frame, logo_text, (x, y), 
                   font, 0.5, (0, 255, 255), 1)
        
        return frame
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame with all enabled features."""
        start_time = time.time()
        
        # Calculate current FPS first
        frame_time = time.time() - start_time if hasattr(self, '_last_frame_start') else 0.033
        self._last_frame_start = start_time
        
        current_fps = 1.0 / frame_time if frame_time > 0 else 30.0
        self.fps_history.append(current_fps)
        
        # Keep only last 30 FPS measurements for smooth average
        if len(self.fps_history) > 30:
            self.fps_history.pop(0)
        
        avg_fps = sum(self.fps_history) / len(self.fps_history)
        
        # YOLO object detection - detect ALL objects with aggressive settings
        if self.enable_yolo and self.yolo_model:
            try:
                # Use aggressive detection settings to catch everything
                results = self.yolo_model(
                    frame, 
                    verbose=False,
                    conf=0.1,      # Very low confidence threshold
                    iou=0.7,       # Higher IoU threshold to allow more overlapping detections
                    max_det=300,   # Allow up to 300 detections per image
                    agnostic_nms=False  # Don't use class-agnostic NMS
                )
                frame = self._draw_yolo_detections(frame, results, avg_fps)
            except Exception as e:
                print(f"⚠️  YOLO processing error: {e}")
        
        # MediaPipe pose detection (bounding boxes for people) - detect ALL people
        if self.enable_pose and self.pose_detector:
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pose_results = self.pose_detector.process(rgb_frame)
                if pose_results.pose_landmarks:
                    frame = self._draw_pose_bounding_box(frame, pose_results.pose_landmarks, avg_fps)
            except Exception as e:
                print(f"⚠️  Pose processing error: {e}")
        
        # Update frame tracking
        end_time = time.time()
        total_frame_time = end_time - start_time
        self.total_time += total_frame_time
        self.frame_count += 1
        
        # No global overlays - clean frame with just detection boxes
        return frame
    
    def process_video(self, input_path: str, output_path: str = "output.mp4") -> bool:
        """Process entire video file."""
        print(f"🎬 Processing video: {input_path}")
        
        # Open input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"❌ Error: Could not open video file: {input_path}")
            return False
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📊 Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print(f"❌ Error: Could not create output video: {output_path}")
            cap.release()
            return False
        
        print("🔄 Processing frames...")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Write frame
                out.write(processed_frame)
                
                # Progress indicator
                progress = (self.frame_count / total_frames) * 100
                if self.frame_count % 30 == 0:  # Update every 30 frames
                    avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
                    print(f"📈 Progress: {progress:.1f}% | Processing FPS: {avg_fps:.1f}")
        
        except KeyboardInterrupt:
            print("\n⏹️  Processing interrupted by user")
        except Exception as e:
            print(f"❌ Error during processing: {e}")
            return False
        finally:
            cap.release()
            out.release()
        
        # Final statistics
        if self.frame_count > 0:
            avg_processing_fps = self.frame_count / self.total_time
            print(f"\n✅ Processing complete!")
            print(f"📊 Statistics:")
            print(f"   • Frames processed: {self.frame_count}")
            print(f"   • Total time: {self.total_time:.2f}s")
            print(f"   • Average processing FPS: {avg_processing_fps:.2f}")
            print(f"   • Output saved: {output_path}")
        
        return True
    
    def cleanup(self):
        """Clean up resources."""
        if self.pose_detector:
            self.pose_detector.close()


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="r/vision - Advanced Visual Processing Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python r_vision.py input.mp4
  python r_vision.py input.mp4 --output processed.mp4
  python r_vision.py input.mp4 --no-yolo --confidence 0.7
  python r_vision.py input.mp4 --no-pose --yolo-confidence 0.3
        """
    )
    
    parser.add_argument("input", help="Input video file path")
    parser.add_argument("-o", "--output", default="output.mp4", 
                       help="Output video file path (default: output.mp4)")
    parser.add_argument("--no-yolo", action="store_true", 
                       help="Disable YOLO object detection")
    parser.add_argument("--no-pose", action="store_true", 
                       help="Disable MediaPipe pose detection")
    parser.add_argument("--yolo-confidence", type=float, default=0.25,
                       help="YOLO confidence threshold (default: 0.25 - lower for more detections)")
    parser.add_argument("--pose-confidence", type=float, default=0.5,
                       help="Pose detection confidence threshold (default: 0.5)")
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.input).exists():
        print(f"❌ Error: Input file not found: {args.input}")
        return 1
    
    # Print banner
    print("=" * 60)
    print("🔍 r/vision - Advanced Visual Processing Tool")
    print("=" * 60)
    print(f"📁 Input: {args.input}")
    print(f"💾 Output: {args.output}")
    print(f"🎯 YOLO: {'Enabled' if not args.no_yolo else 'Disabled'}")
    print(f"🤸 Pose: {'Enabled' if not args.no_pose else 'Disabled'}")
    print("=" * 60)
    
    # Initialize processor
    processor = RVisionProcessor(
        enable_yolo=not args.no_yolo,
        enable_pose=not args.no_pose,
        confidence_threshold=args.yolo_confidence,
        pose_confidence=args.pose_confidence
    )
    
    try:
        # Process video
        success = processor.process_video(args.input, args.output)
        return 0 if success else 1
    
    except KeyboardInterrupt:
        print("\n⏹️  Process interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1
    finally:
        processor.cleanup()


if __name__ == "__main__":
    exit(main())
