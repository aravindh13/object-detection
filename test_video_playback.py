#!/usr/bin/env python3
"""
Simple video playback test to verify video files work.
"""

import cv2
import time

def test_video_playback():
    print("=== Video Playback Test ===")
    
    # Test RGB video
    print("Testing RGB video...")
    rgb_cap = cv2.VideoCapture('sample_videos/rgb_sample.mp4')
    
    if not rgb_cap.isOpened():
        print("ERROR: Could not open RGB video")
        return
    
    rgb_frames = int(rgb_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rgb_fps = rgb_cap.get(cv2.CAP_PROP_FPS)
    rgb_width = int(rgb_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    rgb_height = int(rgb_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"RGB Video Info:")
    print(f"  Frames: {rgb_frames}")
    print(f"  FPS: {rgb_fps}")
    print(f"  Resolution: {rgb_width}x{rgb_height}")
    
    # Test reading first few frames
    print("Reading first 5 frames...")
    for i in range(5):
        ret, frame = rgb_cap.read()
        if ret:
            print(f"  Frame {i}: OK - Shape: {frame.shape}")
        else:
            print(f"  Frame {i}: FAILED")
            break
    
    rgb_cap.release()
    
    # Test Thermal video
    print("\nTesting Thermal video...")
    thermal_cap = cv2.VideoCapture('sample_videos/thermal_sample.mp4')
    
    if not thermal_cap.isOpened():
        print("ERROR: Could not open Thermal video")
        return
    
    thermal_frames = int(thermal_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    thermal_fps = thermal_cap.get(cv2.CAP_PROP_FPS)
    thermal_width = int(thermal_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    thermal_height = int(thermal_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Thermal Video Info:")
    print(f"  Frames: {thermal_frames}")
    print(f"  FPS: {thermal_fps}")
    print(f"  Resolution: {thermal_width}x{thermal_height}")
    
    # Test reading first few frames
    print("Reading first 5 frames...")
    for i in range(5):
        ret, frame = thermal_cap.read()
        if ret:
            print(f"  Frame {i}: OK - Shape: {frame.shape}")
        else:
            print(f"  Frame {i}: FAILED")
            break
    
    thermal_cap.release()
    
    print("\n=== Video Test Complete ===")

if __name__ == "__main__":
    test_video_playback()

