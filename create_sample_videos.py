#!/usr/bin/env python3
"""
Create sample RGB and thermal videos from existing images for testing the video processing pipeline.
"""

import cv2
import os
import numpy as np
from pathlib import Path

def create_video_from_images(image_dir, output_path, fps=10, duration=5):
    """Create a video from images in a directory."""
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        image_files.extend(Path(image_dir).glob(f'*{ext}'))
    
    if not image_files:
        print(f"No images found in {image_dir}")
        return False
    
    # Sort files by name
    image_files = sorted(image_files)
    
    # Read first image to get dimensions
    first_img = cv2.imread(str(image_files[0]))
    if first_img is None:
        print(f"Could not read first image: {image_files[0]}")
        return False
    
    height, width = first_img.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Calculate frames needed
    total_frames = fps * duration
    
    # Repeat images to fill duration
    for i in range(total_frames):
        img_idx = i % len(image_files)
        img = cv2.imread(str(image_files[img_idx]))
        if img is not None:
            out.write(img)
    
    out.release()
    print(f"Created video: {output_path} ({total_frames} frames, {fps} FPS)")
    return True

def main():
    """Create sample videos from RGB and thermal directories."""
    rgb_dir = "data/rgb"
    thermal_dir = "data/thermal"
    output_dir = "sample_videos"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create RGB video from images
    rgb_video = os.path.join(output_dir, "rgb_sample.mp4")
    if create_video_from_images(rgb_dir, rgb_video, fps=10, duration=10):
        print(f"[OK] RGB video created: {rgb_video}")
    else:
        print("[ERROR] Failed to create RGB video")
    
    # Check if thermal directory has videos or images
    thermal_files = list(Path(thermal_dir).glob('*'))
    video_files = [f for f in thermal_files if f.suffix.lower() in ['.mp4', '.avi', '.mov']]
    image_files = [f for f in thermal_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
    
    if video_files:
        # Copy first thermal video
        thermal_video = os.path.join(output_dir, "thermal_sample.mp4")
        import shutil
        shutil.copy2(str(video_files[0]), thermal_video)
        print(f"[OK] Thermal video copied: {thermal_video}")
    elif image_files:
        # Create thermal video from images
        thermal_video = os.path.join(output_dir, "thermal_sample.mp4")
        if create_video_from_images(thermal_dir, thermal_video, fps=10, duration=10):
            print(f"[OK] Thermal video created: {thermal_video}")
        else:
            print("[ERROR] Failed to create thermal video")
    else:
        print("[ERROR] No thermal videos or images found")
        thermal_video = None
    
    print(f"\nSample videos available in: {output_dir}")
    print("You can now use these videos in the Streamlit UI:")
    print(f"- RGB video: {rgb_video}")
    if thermal_video:
        print(f"- Thermal video: {thermal_video}")
    else:
        print("- Use existing thermal videos from data/thermal/ directory")

if __name__ == "__main__":
    main()
