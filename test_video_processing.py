#!/usr/bin/env python3
"""
Test script to verify RGB and thermal video processing works correctly.
"""

import cv2
import numpy as np
from separate_inference import RGBOnlyInference, ThermalOnlyInference, visualize_detections

def test_video_processing():
    """Test video processing with sample videos."""
    print("Loading models...")
    rgb_infer = RGBOnlyInference('runs/exp/best_model.pth', device='cpu', conf_threshold=0.9)
    thermal_infer = ThermalOnlyInference('runs/exp/best_model.pth', device='cpu', conf_threshold=0.9)
    
    print("Opening video files...")
    rgb_cap = cv2.VideoCapture('sample_videos/rgb_sample.mp4')
    thermal_cap = cv2.VideoCapture('sample_videos/thermal_sample.mp4')
    
    if not rgb_cap.isOpened():
        print("ERROR: Could not open RGB video")
        return
    
    if not thermal_cap.isOpened():
        print("ERROR: Could not open Thermal video")
        return
    
    print("Processing first few frames...")
    frame_count = 0
    
    while frame_count < 5:  # Process only first 5 frames
        ret_rgb, rgb_frame = rgb_cap.read()
        ret_thermal, thermal_frame = thermal_cap.read()
        
        if not ret_rgb or not ret_thermal:
            break
        
        # Convert thermal to grayscale if needed
        if thermal_frame.ndim == 3:
            thermal_gray = cv2.cvtColor(thermal_frame, cv2.COLOR_BGR2GRAY)
        else:
            thermal_gray = thermal_frame
        
        # Run inference
        rgb_dets = rgb_infer.predict(rgb_frame)
        thermal_dets = thermal_infer.predict(thermal_gray)
        
        print(f"Frame {frame_count}: RGB detections: {len(rgb_dets)}, Thermal detections: {len(thermal_dets)}")
        
        # Visualize and save
        rgb_vis = visualize_detections(rgb_frame.copy(), rgb_dets, rgb_infer.class_names, "RGB", 5)
        thermal_vis = visualize_detections(thermal_frame.copy(), thermal_dets, thermal_infer.class_names, "Thermal", 5)
        
        # Save visualization
        cv2.imwrite(f'test_output_rgb_frame_{frame_count}.jpg', rgb_vis)
        cv2.imwrite(f'test_output_thermal_frame_{frame_count}.jpg', thermal_vis)
        
        frame_count += 1
    
    rgb_cap.release()
    thermal_cap.release()
    print(f"Processed {frame_count} frames. Check test_output_*.jpg files for results.")

if __name__ == "__main__":
    test_video_processing()

