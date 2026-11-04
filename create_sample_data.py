"""
Script to create sample RGB + thermal data for testing the training pipeline.
"""

import os
import numpy as np
import cv2
from PIL import Image

def create_sample_data():
    """Create sample RGB and thermal images with labels."""
    
    # Create directories
    os.makedirs("data/rgb", exist_ok=True)
    os.makedirs("data/thermal", exist_ok=True)
    os.makedirs("data/labels", exist_ok=True)
    
    # Create 10 sample image pairs
    for i in range(10):
        # Create RGB image (640x640x3)
        rgb_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Add some colored rectangles to simulate objects
        cv2.rectangle(rgb_image, (100, 100), (200, 200), (255, 0, 0), -1)  # Blue rectangle
        cv2.rectangle(rgb_image, (300, 300), (400, 400), (0, 255, 0), -1)  # Green rectangle
        
        # Create thermal image (640x640)
        thermal_image = np.random.randint(0, 255, (640, 640), dtype=np.uint8)
        
        # Add thermal signatures
        cv2.rectangle(thermal_image, (100, 100), (200, 200), 200, -1)  # Hot object
        cv2.rectangle(thermal_image, (300, 300), (400, 400), 150, -1)  # Warm object
        
        # Save images
        rgb_path = f"data/rgb/sample_{i:03d}.jpg"
        thermal_path = f"data/thermal/sample_{i:03d}.jpg"
        
        cv2.imwrite(rgb_path, rgb_image)
        cv2.imwrite(thermal_path, thermal_image)
        
        # Create YOLO format labels
        # Object 1: class 0 (person), center at (150, 150), size 100x100
        # Object 2: class 1 (car), center at (350, 350), size 100x100
        label_content = f"0 0.234375 0.234375 0.15625 0.15625\n1 0.546875 0.546875 0.15625 0.15625\n"
        
        label_path = f"data/labels/sample_{i:03d}.txt"
        with open(label_path, 'w') as f:
            f.write(label_content)
        
        print(f"Created sample_{i:03d}: RGB, thermal, and labels")
    
    print(f"\nSample data created successfully!")
    print(f"RGB images: data/rgb/")
    print(f"Thermal images: data/thermal/")
    print(f"Labels: data/labels/")

if __name__ == "__main__":
    create_sample_data()

