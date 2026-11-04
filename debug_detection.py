#!/usr/bin/env python3
"""
Debug script to test object detection on MP4 files and see what's happening.
"""

import cv2
import numpy as np
import time
from separate_inference import RGBOnlyInference, ThermalOnlyInference, visualize_detections

def debug_detection():
    print("=== DEBUG: Object Detection on MP4 Files ===")
    
    # Load models with different confidence levels
    print("Loading models...")
    try:
        # Try with lower confidence to see more detections
        rgb_infer = RGBOnlyInference('runs/exp/best_model.pth', device='cpu', conf_threshold=0.5)
        thermal_infer = ThermalOnlyInference('runs/exp/best_model.pth', device='cpu', conf_threshold=0.5)
        print("[OK] Models loaded with confidence 0.5")
    except Exception as e:
        print(f"[ERROR] Model loading failed: {e}")
        return
    
    # Open videos
    print("Opening videos...")
    rgb_cap = cv2.VideoCapture('sample_videos/rgb_sample.mp4')
    thermal_cap = cv2.VideoCapture('sample_videos/thermal_sample.mp4')
    
    if not rgb_cap.isOpened():
        print("[ERROR] Could not open RGB video")
        return
    
    if not thermal_cap.isOpened():
        print("[ERROR] Could not open Thermal video")
        return
    
    print("[OK] Videos opened!")
    
    # Process first few frames
    print("\nProcessing first 3 frames...")
    
    for frame_num in range(3):
        ret_rgb, rgb_frame = rgb_cap.read()
        ret_thermal, thermal_frame = thermal_cap.read()
        
        if not ret_rgb or not ret_thermal:
            print(f"End of video at frame {frame_num}")
            break
        
        print(f"\n--- Frame {frame_num} ---")
        
        # Convert thermal to grayscale if needed
        if thermal_frame.ndim == 3:
            thermal_gray = cv2.cvtColor(thermal_frame, cv2.COLOR_BGR2GRAY)
        else:
            thermal_gray = thermal_frame
        
        # Run detection
        try:
            rgb_dets = rgb_infer.predict(rgb_frame)
            thermal_dets = thermal_infer.predict(thermal_gray)
            
            print(f"RGB detections: {len(rgb_dets)}")
            print(f"Thermal detections: {len(thermal_dets)}")
            
            # Show first few detections
            if rgb_dets:
                print("Sample RGB detections:")
                for i, det in enumerate(rgb_dets[:3]):
                    print(f"  {i}: bbox={det['bbox']}, conf={det['confidence']:.3f}, class={det['class_id']}")
            
            if thermal_dets:
                print("Sample Thermal detections:")
                for i, det in enumerate(thermal_dets[:3]):
                    print(f"  {i}: bbox={det['bbox']}, conf={det['confidence']:.3f}, class={det['class_id']}")
            
            # Visualize and save
            rgb_vis = visualize_detections(rgb_frame.copy(), rgb_dets, rgb_infer.class_names, "RGB", 5)
            thermal_vis = visualize_detections(thermal_frame.copy(), thermal_dets, thermal_infer.class_names, "Thermal", 5)
            
            # Save debug images
            cv2.imwrite(f'debug_rgb_frame_{frame_num}.jpg', rgb_vis)
            cv2.imwrite(f'debug_thermal_frame_{frame_num}.jpg', thermal_vis)
            print(f"Saved debug images for frame {frame_num}")
            
        except Exception as e:
            print(f"[ERROR] Processing frame {frame_num}: {e}")
    
    rgb_cap.release()
    thermal_cap.release()
    
    print("\n=== Debug Complete ===")
    print("Check debug_*.jpg files to see if detections are visualized")

if __name__ == "__main__":
    debug_detection()

