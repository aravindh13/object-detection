"""
Separate RGB and Thermal inference classes for independent analysis.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Optional
from inference import RGBTInference, NMSProcessor
from models.two_stream_yolo import create_model

class RGBOnlyInference:
    """RGB-only inference pipeline."""
    
    def __init__(self, 
                 model_path: str,
                 num_classes: int = 80,
                 input_size: int = 640,
                 conf_threshold: float = 0.5,
                 nms_threshold: float = 0.5,
                 device: str = 'auto'):
        """Initialize RGB-only inference."""
        self.num_classes = num_classes
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load model
        self.model = create_model(num_classes=num_classes, input_size=input_size)
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Direct state dict format
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize NMS processor
        self.nms_processor = NMSProcessor(conf_threshold, nms_threshold)
        
        # COCO class names
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
    
    def preprocess_rgb(self, rgb_image: np.ndarray) -> torch.Tensor:
        """Preprocess RGB image for inference."""
        # Resize image
        rgb_resized = cv2.resize(rgb_image, (self.input_size, self.input_size))
        
        # Normalize to [0, 1]
        rgb_normalized = rgb_resized.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        rgb_tensor = torch.from_numpy(rgb_normalized).permute(2, 0, 1).unsqueeze(0)
        rgb_tensor = rgb_tensor.to(self.device)
        
        return rgb_tensor
    
    def predict(self, rgb_image: np.ndarray) -> List[Dict]:
        """Run inference on RGB image only."""
        # Preprocess RGB image
        rgb_tensor = self.preprocess_rgb(rgb_image)
        
        # Create dummy thermal tensor (zeros) for model compatibility
        thermal_tensor = torch.zeros(1, 1, self.input_size, self.input_size).to(self.device)
        
        # Run inference
        with torch.no_grad():
            predictions = self.model(rgb_tensor, thermal_tensor)
        
        # Process detections
        detections = self.nms_processor.process_detections(
            predictions, self.input_size, self.num_classes
        )
        
        return detections[0]  # Return detections for first image

class ThermalOnlyInference:
    """Thermal-only inference pipeline."""
    
    def __init__(self, 
                 model_path: str,
                 num_classes: int = 80,
                 input_size: int = 640,
                 conf_threshold: float = 0.5,
                 nms_threshold: float = 0.5,
                 device: str = 'auto'):
        """Initialize thermal-only inference."""
        self.num_classes = num_classes
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load model
        self.model = create_model(num_classes=num_classes, input_size=input_size)
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Direct state dict format
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize NMS processor
        self.nms_processor = NMSProcessor(conf_threshold, nms_threshold)
        
        # COCO class names
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
    
    def preprocess_thermal(self, thermal_image: np.ndarray) -> torch.Tensor:
        """Preprocess thermal image for inference."""
        # Ensure thermal image is grayscale
        if thermal_image.ndim == 3:
            thermal_gray = cv2.cvtColor(thermal_image, cv2.COLOR_BGR2GRAY)
        else:
            thermal_gray = thermal_image
        
        # Resize image
        thermal_resized = cv2.resize(thermal_gray, (self.input_size, self.input_size))
        
        # Normalize to [0, 1]
        thermal_normalized = thermal_resized.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        thermal_tensor = torch.from_numpy(thermal_normalized).unsqueeze(0).unsqueeze(0)
        thermal_tensor = thermal_tensor.to(self.device)
        
        return thermal_tensor
    
    def predict(self, thermal_image: np.ndarray) -> List[Dict]:
        """Run inference on thermal image only."""
        # Preprocess thermal image
        thermal_tensor = self.preprocess_thermal(thermal_image)
        
        # Create dummy RGB tensor (zeros) for model compatibility
        rgb_tensor = torch.zeros(1, 3, self.input_size, self.input_size).to(self.device)
        
        # Run inference
        with torch.no_grad():
            predictions = self.model(rgb_tensor, thermal_tensor)
        
        # Process detections
        detections = self.nms_processor.process_detections(
            predictions, self.input_size, self.num_classes
        )
        
        return detections[0]  # Return detections for first image

def visualize_detections(frame: np.ndarray, detections: List[Dict], class_names: List[str], 
                       stream_type: str = "RGB", max_detections: int = 20) -> np.ndarray:
    """Visualize detections on frame with improved clarity."""
    vis_frame = frame.copy()
    h, w = vis_frame.shape[:2]
    
    # Limit number of detections to avoid clutter
    detections = detections[:max_detections]
    
    # Sort by confidence to show best detections first
    detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
    
    for i, det in enumerate(detections):
        x, y, w_norm, h_norm = det["bbox"]
        conf = det["confidence"]
        cid = det["class_id"]
        
        # Skip very small detections
        if w_norm < 0.01 or h_norm < 0.01:
            continue
        
        # Convert normalized coordinates to pixel coordinates
        xc, yc, ww, hh = x * w, y * h, w_norm * w, h_norm * h
        x1, y1 = int(max(0, xc - ww / 2)), int(max(0, yc - hh / 2))
        x2, y2 = int(min(w - 1, x1 + ww)), int(min(h - 1, y1 + hh))
        
        # Skip if bounding box is too small
        if (x2 - x1) < 10 or (y2 - y1) < 10:
            continue
        
        # Draw bounding box with thicker lines
        color = (0, 255, 0) if stream_type == "RGB" else (0, 0, 255)  # Green for RGB, Red for Thermal
        thickness = max(2, int(min(w, h) / 200))  # Adaptive thickness
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label with background
        name = class_names[cid] if cid < len(class_names) else f"Class {cid}"
        label = f"{name} {conf:.2f}"
        
        # Calculate text size
        font_scale = max(0.4, min(w, h) / 1000)
        font_thickness = max(1, int(min(w, h) / 500))
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        
        # Draw background rectangle for text
        cv2.rectangle(vis_frame, (x1, y1 - text_h - 5), (x1 + text_w, y1), color, -1)
        
        # Draw text
        cv2.putText(vis_frame, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
    
    return vis_frame
