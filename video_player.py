#!/usr/bin/env python3
"""
Direct video player with object detection - command line version.
"""

import cv2
import numpy as np
import time
from separate_inference import RGBOnlyInference, ThermalOnlyInference, visualize_detections

def play_videos_with_detection():
    print("=== Direct Video Player with Object Detection ===")
    
    # Load models
    print("Loading models...")
    try:
        rgb_infer = RGBOnlyInference('runs/exp/best_model.pth', device='cpu', conf_threshold=0.95)
        thermal_infer = ThermalOnlyInference('runs/exp/best_model.pth', device='cpu', conf_threshold=0.95)
        print("[OK] Models loaded!")
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
    
    # Create windows
    cv2.namedWindow('RGB Stream', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Thermal Stream', cv2.WINDOW_NORMAL)
    
    frame_count = 0
    start_time = time.time()
    
    print("Starting video playback... Press 'q' to quit, 's' to save frame")
    
    try:
        while True:
            ret_rgb, rgb_frame = rgb_cap.read()
            ret_thermal, thermal_frame = thermal_cap.read()
            
            if not ret_rgb or not ret_thermal:
                print("End of video reached")
                break
            
            # Process every 3rd frame for performance
            if frame_count % 3 != 0:
                frame_count += 1
                continue
            
            # Convert thermal to grayscale if needed
            if thermal_frame.ndim == 3:
                thermal_gray = cv2.cvtColor(thermal_frame, cv2.COLOR_BGR2GRAY)
            else:
                thermal_gray = thermal_frame
            
            # Run detection
            try:
                rgb_dets = rgb_infer.predict(rgb_frame)
                thermal_dets = thermal_infer.predict(thermal_gray)
                
                # Visualize
                rgb_vis = visualize_detections(rgb_frame.copy(), rgb_dets, rgb_infer.class_names, "RGB", 3)
                thermal_vis = visualize_detections(thermal_frame.copy(), thermal_dets, thermal_infer.class_names, "Thermal", 3)
                
                # Add info
                cv2.putText(rgb_vis, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(rgb_vis, f"Detections: {len(rgb_dets)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.putText(thermal_vis, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(thermal_vis, f"Detections: {len(thermal_dets)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Display
                cv2.imshow('RGB Stream', rgb_vis)
                cv2.imshow('Thermal Stream', thermal_vis)
                
                # Calculate FPS
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"Frame {frame_count}: RGB={len(rgb_dets)} detections, Thermal={len(thermal_dets)} detections, FPS={fps:.1f}")
                
            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")
            
            frame_count += 1
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quit requested")
                break
            elif key == ord('s'):
                cv2.imwrite(f'saved_rgb_frame_{frame_count}.jpg', rgb_vis)
                cv2.imwrite(f'saved_thermal_frame_{frame_count}.jpg', thermal_vis)
                print(f"Saved frames {frame_count}")
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    finally:
        # Cleanup
        rgb_cap.release()
        thermal_cap.release()
        cv2.destroyAllWindows()
        
        # Final stats
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        print(f"\n=== Final Stats ===")
        print(f"Frames processed: {frame_count}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average FPS: {avg_fps:.2f}")

if __name__ == "__main__":
    play_videos_with_detection()

