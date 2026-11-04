"""
Custom dataset for RGB + thermal image pairs with object detection annotations.
Supports COCO-style and YOLO-style label formats.
"""

import os
import json
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, List, Tuple, Optional, Union
import random


class RGBTThermalDataset(Dataset):
    """
    Dataset for RGB + thermal image pairs with object detection annotations.
    
    Expected directory structure:
    
    data/
    ├── rgb/
    │   ├── image1.jpg
    │   └── image2.jpg
    ├── thermal/
    │   ├── image1.jpg
    │   └── image2.jpg
    └── labels/
        ├── image1.txt (YOLO format) or image1.json (COCO format)
        └── image2.txt
    """
    
    def __init__(self,
                 data_dir: str,
                 rgb_dir: str = "rgb",
                 thermal_dir: str = "thermal", 
                 labels_dir: str = "labels",
                 label_format: str = "yolo",  # "yolo" or "coco"
                 input_size: int = 640,
                 augment: bool = True,
                 class_names: Optional[List[str]] = None):
        """
        Args:
            data_dir: Root directory containing rgb, thermal, and labels folders
            rgb_dir: Subdirectory name for RGB images
            thermal_dir: Subdirectory name for thermal images
            labels_dir: Subdirectory name for labels
            label_format: "yolo" or "coco"
            input_size: Target input size for images
            augment: Whether to apply data augmentation
            class_names: List of class names (required for COCO format)
        """
        self.data_dir = data_dir
        self.rgb_dir = os.path.join(data_dir, rgb_dir)
        self.thermal_dir = os.path.join(data_dir, thermal_dir)
        self.labels_dir = os.path.join(data_dir, labels_dir)
        self.label_format = label_format
        self.input_size = input_size
        self.augment = augment
        self.class_names = class_names or []
        
        # Get list of image files
        self.image_files = self._get_image_files()
        
        # Setup augmentations
        self._setup_augmentations()
        
        # Load class mapping for COCO format
        if label_format == "coco" and class_names:
            self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        else:
            self.class_to_idx = None
    
    def _get_image_files(self) -> List[str]:
        """Get list of image files (without extension)."""
        rgb_files = set()
        thermal_files = set()
        label_files = set()
        
        # Get RGB files
        if os.path.exists(self.rgb_dir):
            for f in os.listdir(self.rgb_dir):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    rgb_files.add(os.path.splitext(f)[0])
        
        # Get thermal files
        if os.path.exists(self.thermal_dir):
            for f in os.listdir(self.thermal_dir):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    thermal_files.add(os.path.splitext(f)[0])
        
        # Get label files
        if os.path.exists(self.labels_dir):
            for f in os.listdir(self.labels_dir):
                if f.lower().endswith(('.txt', '.json')):
                    label_files.add(os.path.splitext(f)[0])
        
        # Find common files
        common_files = rgb_files.intersection(thermal_files).intersection(label_files)
        return sorted(list(common_files))
    
    def _setup_augmentations(self):
        """Setup data augmentation pipeline."""
        if self.augment:
            # RGB-specific augmentations
            self.rgb_transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomRotate90(p=0.3),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=0.5
                ),
                A.RandomCrop(
                    height=self.input_size,
                    width=self.input_size,
                    p=0.8
                ),
                A.Resize(self.input_size, self.input_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ], bbox_params=A.BboxParams(
                format='yolo',
                min_visibility=0.1,
                label_fields=['class_labels']
            ))
            
            # Thermal-specific augmentations
            self.thermal_transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomRotate90(p=0.3),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.RandomCrop(
                    height=self.input_size,
                    width=self.input_size,
                    p=0.8
                ),
                A.Resize(self.input_size, self.input_size),
                A.Normalize(
                    mean=[0.5],
                    std=[0.5]
                ),
                ToTensorV2()
            ], bbox_params=A.BboxParams(
                format='yolo',
                min_visibility=0.1,
                label_fields=['class_labels']
            ))
        else:
            # No augmentation
            self.rgb_transform = A.Compose([
                A.Resize(self.input_size, self.input_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ], bbox_params=A.BboxParams(
                format='yolo',
                min_visibility=0.1,
                label_fields=['class_labels']
            ))
            
            self.thermal_transform = A.Compose([
                A.Resize(self.input_size, self.input_size),
                A.Normalize(
                    mean=[0.5],
                    std=[0.5]
                ),
                ToTensorV2()
            ], bbox_params=A.BboxParams(
                format='yolo',
                min_visibility=0.1,
                label_fields=['class_labels']
            ))
    
    def _load_yolo_labels(self, label_path: str) -> Tuple[List[List[float]], List[int]]:
        """Load YOLO format labels."""
        boxes = []
        class_ids = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        boxes.append([x_center, y_center, width, height])
                        class_ids.append(class_id)
        
        return boxes, class_ids
    
    def _load_coco_labels(self, label_path: str) -> Tuple[List[List[float]], List[int]]:
        """Load COCO format labels."""
        boxes = []
        class_ids = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                data = json.load(f)
            
            for annotation in data.get('annotations', []):
                bbox = annotation['bbox']  # [x, y, w, h]
                category_id = annotation['category_id']
                
                # Convert to YOLO format (normalized)
                img_width = data['images'][0]['width']
                img_height = data['images'][0]['height']
                
                x_center = (bbox[0] + bbox[2] / 2) / img_width
                y_center = (bbox[1] + bbox[3] / 2) / img_height
                width = bbox[2] / img_width
                height = bbox[3] / img_height
                
                boxes.append([x_center, y_center, width, height])
                class_ids.append(category_id)
        
        return boxes, class_ids
    
    def _load_image_pair(self, base_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load RGB and thermal image pair."""
        # Try different extensions
        extensions = ['.jpg', '.jpeg', '.png']
        
        rgb_image = None
        thermal_image = None
        
        for ext in extensions:
            rgb_path = os.path.join(self.rgb_dir, base_name + ext)
            thermal_path = os.path.join(self.thermal_dir, base_name + ext)
            
            if os.path.exists(rgb_path) and os.path.exists(thermal_path):
                rgb_image = cv2.imread(rgb_path)
                rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
                
                thermal_image = cv2.imread(thermal_path, cv2.IMREAD_GRAYSCALE)
                break
        
        if rgb_image is None or thermal_image is None:
            raise FileNotFoundError(f"Could not find image pair for {base_name}")
        
        return rgb_image, thermal_image
    
    def _align_thermal_to_rgb(self, rgb_image: np.ndarray, thermal_image: np.ndarray) -> np.ndarray:
        """
        Align thermal image to RGB using homography transformation.
        This is a simplified version - in practice, you'd use camera calibration.
        """
        # Simple resize to match RGB dimensions
        thermal_aligned = cv2.resize(thermal_image, (rgb_image.shape[1], rgb_image.shape[0]))
        
        # Convert to 3-channel for consistency
        thermal_aligned = np.stack([thermal_aligned] * 3, axis=-1)
        
        return thermal_aligned
    
    def _add_thermal_noise(self, thermal_image: np.ndarray) -> np.ndarray:
        """Add realistic thermal noise."""
        # Add Gaussian noise
        noise = np.random.normal(0, 5, thermal_image.shape).astype(np.uint8)
        thermal_noisy = cv2.add(thermal_image, noise)
        
        # Add thermal contrast variation
        contrast_factor = random.uniform(0.8, 1.2)
        thermal_noisy = cv2.convertScaleAbs(thermal_noisy, alpha=contrast_factor)
        
        return thermal_noisy
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Returns:
            Dictionary containing:
            - rgb_image: RGB image tensor [3, H, W]
            - thermal_image: Thermal image tensor [1, H, W] 
            - boxes: Bounding boxes tensor [N, 4] (normalized)
            - class_ids: Class IDs tensor [N]
            - image_id: Image identifier
        """
        base_name = self.image_files[idx]
        
        # Load images
        rgb_image, thermal_image = self._load_image_pair(base_name)
        
        # Align thermal to RGB
        thermal_image = self._align_thermal_to_rgb(rgb_image, thermal_image)
        
        # Add thermal noise if augmenting
        if self.augment:
            thermal_image = self._add_thermal_noise(thermal_image)
        
        # Load labels
        if self.label_format == "yolo":
            label_path = os.path.join(self.labels_dir, base_name + ".txt")
            boxes, class_ids = self._load_yolo_labels(label_path)
        else:  # coco
            label_path = os.path.join(self.labels_dir, base_name + ".json")
            boxes, class_ids = self._load_coco_labels(label_path)
        
        # Apply augmentations separately for RGB and thermal
        if len(boxes) > 0:
            # Apply RGB transformations
            rgb_transformed = self.rgb_transform(
                image=rgb_image,
                bboxes=boxes,
                class_labels=class_ids
            )
            
            # Apply thermal transformations (convert to 3-channel for consistency)
            thermal_3ch = np.stack([thermal_image[:, :, 0]] * 3, axis=2)
            thermal_transformed = self.thermal_transform(
                image=thermal_3ch,
                bboxes=boxes,
                class_labels=class_ids
            )
        else:
            # Apply RGB transformations
            rgb_transformed = self.rgb_transform(image=rgb_image)
            
            # Apply thermal transformations
            thermal_3ch = np.stack([thermal_image[:, :, 0]] * 3, axis=2)
            thermal_transformed = self.thermal_transform(image=thermal_3ch)
        
        # Extract RGB and thermal tensors
        rgb_tensor = rgb_transformed['image']
        thermal_tensor = thermal_transformed['image'][:1]  # Take only first channel
        
        # Convert boxes to tensor
        if len(boxes) > 0:
            boxes_tensor = torch.tensor(rgb_transformed['bboxes'], dtype=torch.float32)
            class_ids_tensor = torch.tensor(rgb_transformed['class_labels'], dtype=torch.long)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            class_ids_tensor = torch.zeros((0,), dtype=torch.long)
        
        return {
            'rgb_image': rgb_tensor,
            'thermal_image': thermal_tensor,
            'boxes': boxes_tensor,
            'class_ids': class_ids_tensor,
            'image_id': base_name
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for DataLoader.
    Handles variable number of objects per image.
    """
    rgb_images = torch.stack([item['rgb_image'] for item in batch])
    thermal_images = torch.stack([item['thermal_image'] for item in batch])
    
    # Pad boxes and class_ids to same length
    max_objects = max(len(item['boxes']) for item in batch)
    
    boxes_list = []
    class_ids_list = []
    
    for item in batch:
        boxes = item['boxes']
        class_ids = item['class_ids']
        
        if len(boxes) < max_objects:
            # Pad with zeros
            pad_boxes = torch.zeros(max_objects - len(boxes), 4)
            pad_class_ids = torch.zeros(max_objects - len(class_ids), dtype=torch.long)
            
            boxes = torch.cat([boxes, pad_boxes], dim=0)
            class_ids = torch.cat([class_ids, pad_class_ids], dim=0)
        
        boxes_list.append(boxes)
        class_ids_list.append(class_ids)
    
    boxes_batch = torch.stack(boxes_list)
    class_ids_batch = torch.stack(class_ids_list)
    
    return {
        'rgb_images': rgb_images,
        'thermal_images': thermal_images,
        'boxes': boxes_batch,
        'class_ids': class_ids_batch,
        'image_ids': [item['image_id'] for item in batch]
    }


if __name__ == "__main__":
    # Test dataset
    dataset = RGBTThermalDataset(
        data_dir="data",
        label_format="yolo",
        input_size=640,
        augment=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"RGB image shape: {sample['rgb_image'].shape}")
        print(f"Thermal image shape: {sample['thermal_image'].shape}")
        print(f"Number of boxes: {len(sample['boxes'])}")
        print(f"Class IDs: {sample['class_ids']}")

