"""
Inference script for two-stream RGB + thermal object detection.
Supports NMS, visualization, and batch processing.
"""

import os
import argparse
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

from models.two_stream_yolo import create_model
from losses.yolo_loss import bbox_iou


class NMSProcessor:
    """Non-Maximum Suppression processor for multi-scale detections."""
    
    def __init__(self, conf_threshold: float = 0.5, nms_threshold: float = 0.5):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
    
    def process_detections(self, 
                          predictions: List[torch.Tensor],
                          input_size: int = 640,
                          num_classes: int = 80,
                          num_anchors: int = 3) -> List[Dict]:
        """
        Process raw predictions and apply NMS.
        
        Args:
            predictions: List of predictions for each scale [P3, P4, P5]
            input_size: Input image size
            num_classes: Number of classes
            num_anchors: Number of anchors per scale
            
        Returns:
            List of detections for each image in batch
        """
        batch_size = predictions[0].size(0)
        device = predictions[0].device
        
        all_detections = []
        
        for batch_idx in range(batch_size):
            detections = []
            
            for scale_idx, pred in enumerate(predictions):
                # Get grid size and stride
                grid_size = pred.size(-1)
                stride = input_size // grid_size
                
                # Reshape prediction
                pred = pred[batch_idx].view(num_anchors, -1, grid_size, grid_size)
                
                # Extract components
                pred_obj = torch.sigmoid(pred[:, 0, :, :])  # Objectness
                pred_cls = torch.sigmoid(pred[:, 1:1+num_classes, :, :])  # Classification
                pred_box = pred[:, 1+num_classes:5+num_classes, :, :]  # Box regression
                pred_temp = pred[:, 5+num_classes, :, :]  # Temperature
                
                # Find detections above confidence threshold
                conf_mask = pred_obj > self.conf_threshold
                
                if conf_mask.any():
                    # Get coordinates of high-confidence detections
                    conf_indices = torch.where(conf_mask)
                    
                    for i in range(len(conf_indices[0])):
                        anchor_idx = conf_indices[0][i].item()
                        grid_y = conf_indices[1][i].item()
                        grid_x = conf_indices[2][i].item()
                        
                        # Get confidence and class
                        confidence = pred_obj[anchor_idx, grid_y, grid_x].item()
                        class_scores = pred_cls[anchor_idx, :, grid_y, grid_x]
                        class_id = torch.argmax(class_scores).item()
                        class_conf = class_scores[class_id].item()
                        
                        # Get box coordinates
                        box_x = (grid_x + pred_box[anchor_idx, 0, grid_y, grid_x].item()) * stride
                        box_y = (grid_y + pred_box[anchor_idx, 1, grid_y, grid_x].item()) * stride
                        box_w = pred_box[anchor_idx, 2, grid_y, grid_x].item() * stride
                        box_h = pred_box[anchor_idx, 3, grid_y, grid_x].item() * stride
                        
                        # Get temperature
                        temperature = pred_temp[anchor_idx, grid_y, grid_x].item()
                        
                        # Convert to normalized coordinates
                        x_center = box_x / input_size
                        y_center = box_y / input_size
                        width = box_w / input_size
                        height = box_h / input_size
                        
                        detections.append({
                            'bbox': [x_center, y_center, width, height],
                            'confidence': confidence * class_conf,
                            'class_id': class_id,
                            'temperature': temperature
                        })
            
            # Apply NMS
            if detections:
                detections = self._apply_nms(detections)
            
            all_detections.append(detections)
        
        return all_detections
    
    def _apply_nms(self, detections: List[Dict]) -> List[Dict]:
        """Apply Non-Maximum Suppression to detections."""
        if not detections:
            return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        # Convert to tensors for NMS
        boxes = torch.tensor([det['bbox'] for det in detections])
        scores = torch.tensor([det['confidence'] for det in detections])
        
        # Apply NMS
        keep_indices = self._nms(boxes, scores, self.nms_threshold)
        
        return [detections[i] for i in keep_indices]
    
    def _nms(self, boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> List[int]:
        """Non-Maximum Suppression implementation."""
        if len(boxes) == 0:
            return []
        
        # Convert to xyxy format
        x_center, y_center, width, height = boxes.unbind(-1)
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)
        
        # Sort by scores
        _, indices = scores.sort(descending=True)
        keep = []
        
        while len(indices) > 0:
            # Pick the box with highest score
            current = indices[0]
            keep.append(current.item())
            
            if len(indices) == 1:
                break
            
            # Calculate IoU with remaining boxes
            current_box = boxes_xyxy[current:current+1]
            remaining_boxes = boxes_xyxy[indices[1:]]
            
            ious = bbox_iou(current_box, remaining_boxes, xywh=False)
            
            # Keep boxes with IoU below threshold
            keep_mask = ious < iou_threshold
            indices = indices[1:][keep_mask]
        
        return keep


class RGBTInference:
    """RGB + Thermal inference pipeline."""
    
    def __init__(self, 
                 model_path: str,
                 num_classes: int = 80,
                 input_size: int = 640,
                 conf_threshold: float = 0.5,
                 nms_threshold: float = 0.5,
                 device: str = 'auto'):
        """
        Initialize inference pipeline.
        
        Args:
            model_path: Path to trained model checkpoint
            num_classes: Number of object classes
            input_size: Input image size
            conf_threshold: Confidence threshold for detections
            nms_threshold: NMS threshold
            device: Device to use for inference
        """
        self.num_classes = num_classes
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        
        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Initialize NMS processor
        self.nms_processor = NMSProcessor(conf_threshold, nms_threshold)
        
        # Class names (COCO classes)
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
    
    def _load_model(self, model_path: str):
        """Load trained model from checkpoint."""
        # Create model
        model = create_model(
            num_classes=self.num_classes,
            input_size=self.input_size,
            pretrained=False
        )
        
        # Load checkpoint
        if os.path.exists(model_path):
            try:
                # Try loading with weights_only=True first (safer)
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            except Exception:
                # If that fails, load with weights_only=False (less secure but works with custom classes)
                print("Warning: Loading checkpoint with weights_only=False. Only use trusted checkpoints.")
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"Loaded model from {model_path}")
        else:
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        
        return model.to(self.device)
    
    def preprocess_images(self, rgb_image: np.ndarray, thermal_image: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess RGB and thermal images for inference.
        
        Args:
            rgb_image: RGB image [H, W, 3]
            thermal_image: Thermal image [H, W] or [H, W, 1]
            
        Returns:
            Preprocessed RGB and thermal tensors
        """
        # Resize images
        rgb_resized = cv2.resize(rgb_image, (self.input_size, self.input_size))
        thermal_resized = cv2.resize(thermal_image, (self.input_size, self.input_size))
        
        # Normalize RGB image
        rgb_normalized = rgb_resized.astype(np.float32) / 255.0
        rgb_normalized = (rgb_normalized - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        # Normalize thermal image
        thermal_normalized = thermal_resized.astype(np.float32) / 255.0
        thermal_normalized = (thermal_normalized - 0.5) / 0.5
        
        # Convert to tensors
        rgb_tensor = torch.from_numpy(rgb_normalized).permute(2, 0, 1).unsqueeze(0).float()
        thermal_tensor = torch.from_numpy(thermal_normalized).unsqueeze(0).unsqueeze(0).float()
        
        return rgb_tensor.to(self.device), thermal_tensor.to(self.device)
    
    def predict(self, rgb_image: np.ndarray, thermal_image: np.ndarray) -> List[Dict]:
        """
        Run inference on RGB + thermal image pair.
        
        Args:
            rgb_image: RGB image [H, W, 3]
            thermal_image: Thermal image [H, W] or [H, W, 1]
            
        Returns:
            List of detections
        """
        # Preprocess images
        rgb_tensor, thermal_tensor = self.preprocess_images(rgb_image, thermal_image)
        
        # Run inference
        with torch.no_grad():
            predictions = self.model(rgb_tensor, thermal_tensor)
        
        # Process detections
        detections = self.nms_processor.process_detections(
            predictions, self.input_size, self.num_classes
        )
        
        return detections[0]  # Return detections for first (and only) image
    
    def visualize_detections(self, 
                            rgb_image: np.ndarray, 
                            thermal_image: np.ndarray, 
                            detections: List[Dict],
                            save_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize detections on RGB and thermal images.
        
        Args:
            rgb_image: Original RGB image
            thermal_image: Original thermal image
            detections: List of detections
            save_path: Optional path to save visualization
            
        Returns:
            Visualization image
        """
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # RGB image with detections
        axes[0].imshow(rgb_image)
        axes[0].set_title('RGB Image')
        axes[0].axis('off')
        
        # Thermal image with detections
        axes[1].imshow(thermal_image, cmap='hot')
        axes[1].set_title('Thermal Image')
        axes[1].axis('off')
        
        # Combined visualization
        axes[2].imshow(rgb_image)
        axes[2].set_title('Detections')
        axes[2].axis('off')
        
        # Draw detections
        for det in detections:
            bbox = det['bbox']
            confidence = det['confidence']
            class_id = det['class_id']
            temperature = det['temperature']
            
            # Convert normalized coordinates to pixel coordinates
            x_center = bbox[0] * rgb_image.shape[1]
            y_center = bbox[1] * rgb_image.shape[0]
            width = bbox[2] * rgb_image.shape[1]
            height = bbox[3] * rgb_image.shape[0]
            
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            
            # Create rectangle
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            axes[2].add_patch(rect)
            
            # Add label
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f'Class {class_id}'
            label = f'{class_name}: {confidence:.2f}\nTemp: {temperature:.1f}°C'
            
            axes[2].text(
                x1, y1 - 5, label,
                fontsize=8, color='red', weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
            )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        # Convert to numpy array
        fig.canvas.draw()
        vis_array = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        vis_array = vis_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        vis_array = vis_array[:, :, 1:]  # Remove alpha channel
        
        plt.close(fig)
        
        return vis_array


def main():
    parser = argparse.ArgumentParser(description='RGB + Thermal Object Detection Inference')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--rgb_image', type=str, required=True, help='Path to RGB image')
    parser.add_argument('--thermal_image', type=str, required=True, help='Path to thermal image')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--conf_threshold', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--nms_threshold', type=float, default=0.5, help='NMS threshold')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--num_classes', type=int, default=80, help='Number of classes')
    parser.add_argument('--input_size', type=int, default=640, help='Input image size')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize inference pipeline
    inference = RGBTInference(
        model_path=args.model_path,
        num_classes=args.num_classes,
        input_size=args.input_size,
        conf_threshold=args.conf_threshold,
        nms_threshold=args.nms_threshold,
        device=args.device
    )
    
    # Load images
    rgb_image = cv2.imread(args.rgb_image)
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    
    thermal_image = cv2.imread(args.thermal_image, cv2.IMREAD_GRAYSCALE)
    
    if rgb_image is None:
        raise ValueError(f"Could not load RGB image: {args.rgb_image}")
    if thermal_image is None:
        raise ValueError(f"Could not load thermal image: {args.thermal_image}")
    
    print(f"Loaded RGB image: {rgb_image.shape}")
    print(f"Loaded thermal image: {thermal_image.shape}")
    
    # Run inference
    print("Running inference...")
    start_time = time.time()
    
    detections = inference.predict(rgb_image, thermal_image)
    
    inference_time = time.time() - start_time
    print(f"Inference completed in {inference_time:.3f} seconds")
    print(f"Found {len(detections)} detections")
    
    # Print detections
    for i, det in enumerate(detections):
        class_name = inference.class_names[det['class_id']] if det['class_id'] < len(inference.class_names) else f'Class {det["class_id"]}'
        print(f"Detection {i+1}: {class_name} (conf: {det['confidence']:.3f}, temp: {det['temperature']:.1f}°C)")
    
    # Visualize results
    output_path = os.path.join(args.output_dir, 'detection_result.png')
    inference.visualize_detections(rgb_image, thermal_image, detections, output_path)
    
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()

