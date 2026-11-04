#!/usr/bin/env python3
"""
Simplified video processing script to test RGB and thermal video analysis.
"""

import cv2
import numpy as np
import time
from separate_inference import RGBOnlyInference, ThermalOnlyInference, visualize_detections

def process_videos():
    """Process RGB and thermal videos with simplified architecture."""
    print("Loading models...")
    try:
        rgb_infer = RGBOnlyInference('runs/exp/best_model.pth', device='cpu', conf_threshold=0.7)
        thermal_infer = ThermalOnlyInference('runs/exp/best_model.pth', device='cpu', conf_threshold=0.7)
        print("[OK] Models loaded successfully!")
    except Exception as e:
        print(f"[ERROR] Error loading models: {e}")
        return
    
    print("Opening video files...")
    rgb_cap = cv2.VideoCapture('sample_videos/rgb_sample.mp4')
    thermal_cap = cv2.VideoCapture('sample_videos/thermal_sample.mp4')
    
    if not rgb_cap.isOpened():
        print("[ERROR] Could not open RGB video")
        return
    
    if not thermal_cap.isOpened():
        print("[ERROR] Could not open Thermal video")
        return
    
    print("[OK] Videos opened successfully!")
    print(f"RGB: {rgb_cap.get(cv2.CAP_PROP_FRAME_COUNT)} frames")
    print(f"Thermal: {thermal_cap.get(cv2.CAP_PROP_FRAME_COUNT)} frames")
    
    frame_count = 0
    start_time = time.time()
    
    print("Starting video processing...")
    
    while True:
        ret_rgb, rgb_frame = rgb_cap.read()
        ret_thermal, thermal_frame = thermal_cap.read()
        
        if not ret_rgb or not ret_thermal:
            print(f"End of video reached at frame {frame_count}")
            break
        
        # Process every 5th frame for testing
        if frame_count % 5 != 0:
            frame_count += 1
            continue
        
        print(f"Processing frame {frame_count}...")
        
        # Convert thermal to grayscale if needed
        if thermal_frame.ndim == 3:
            thermal_gray = cv2.cvtColor(thermal_frame, cv2.COLOR_BGR2GRAY)
        else:
            thermal_gray = thermal_frame
        
        # Run inference
        try:
            rgb_dets = rgb_infer.predict(rgb_frame)
            thermal_dets = thermal_infer.predict(thermal_gray)
            
            print(f"  RGB detections: {len(rgb_dets)}")
            print(f"  Thermal detections: {len(thermal_dets)}")
            
            # Visualize detections
            rgb_vis = visualize_detections(rgb_frame.copy(), rgb_dets, rgb_infer.class_names, "RGB", 5)
            thermal_vis = visualize_detections(thermal_frame.copy(), thermal_dets, thermal_infer.class_names, "Thermal", 5)
            
            # Save sample frames
            if frame_count < 3:  # Save first 3 processed frames
                cv2.imwrite(f'video_test_rgb_frame_{frame_count}.jpg', rgb_vis)
                cv2.imwrite(f'video_test_thermal_frame_{frame_count}.jpg', thermal_vis)
                print(f"  Saved test frames {frame_count}")
            
        except Exception as e:
            print(f"  [ERROR] Error processing frame {frame_count}: {e}")
        
        frame_count += 1
        
        # Limit processing for testing
        if frame_count > 20:
            print("Stopping after 20 frames for testing")
            break
    
    # Calculate performance
    end_time = time.time()
    total_time = end_time - start_time
    fps = frame_count / total_time if total_time > 0 else 0
    
    print(f"\nProcessing Summary:")
    print(f"  Frames processed: {frame_count}")
    print(f"  Total time: {total_time:.2f} seconds")
    print(f"  Average FPS: {fps:.2f}")
    
    rgb_cap.release()
    thermal_cap.release()
    print("[OK] Video processing completed!")

if __name__ == "__main__":
    process_videos()
