"""
YOLO-style loss functions for two-stream object detection.
Implements BCE loss for objectness/classification, GIoU loss for bounding boxes, and MSE loss for temperature.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict
import math


def bbox_iou(box1: torch.Tensor, box2: torch.Tensor, xywh: bool = True, GIoU: bool = False, DIoU: bool = False, CIoU: bool = False, eps: float = 1e-7) -> torch.Tensor:
    """
    Calculate IoU between two sets of bounding boxes.
    
    Args:
        box1: First set of boxes [N, 4]
        box2: Second set of boxes [M, 4]
        xywh: If True, boxes are in (x_center, y_center, width, height) format
        GIoU: If True, calculate GIoU
        DIoU: If True, calculate DIoU
        CIoU: If True, calculate CIoU
        eps: Small value to avoid division by zero
        
    Returns:
        IoU values [N, M]
    """
    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[..., 0] - box1[..., 2] / 2, box1[..., 0] + box1[..., 2] / 2
        b1_y1, b1_y2 = box1[..., 1] - box1[..., 3] / 2, box1[..., 1] + box1[..., 3] / 2
        b2_x1, b2_x2 = box2[..., 0] - box2[..., 2] / 2, box2[..., 0] + box2[..., 2] / 2
        b2_y1, b2_y2 = box2[..., 1] - box2[..., 3] / 2, box2[..., 1] + box2[..., 3] / 2
    else:  # x1, y1, x2, y2
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU


class YOLOLoss(nn.Module):
    """
    YOLO-style loss function for two-stream object detection.
    
    Loss components:
    1. Objectness loss (BCE)
    2. Classification loss (BCE) 
    3. Bounding box regression loss (GIoU)
    4. Temperature regression loss (MSE)
    """
    
    def __init__(self,
                 num_classes: int = 80,
                 num_anchors: int = 3,
                 input_size: int = 640,
                 anchor_sizes: List[List[float]] = None,
                 lambda_obj: float = 1.0,
                 lambda_cls: float = 1.0,
                 lambda_box: float = 1.0,
                 lambda_temp: float = 0.1,
                 obj_pos_weight: float = 1.0,
                 cls_pos_weight: float = 1.0):
        """
        Args:
            num_classes: Number of object classes
            num_anchors: Number of anchors per grid cell
            input_size: Input image size
            anchor_sizes: List of anchor sizes for each scale
            lambda_obj: Weight for objectness loss
            lambda_cls: Weight for classification loss
            lambda_box: Weight for bounding box loss
            lambda_temp: Weight for temperature loss
            obj_pos_weight: Positive weight for objectness loss
            cls_pos_weight: Positive weight for classification loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.input_size = input_size
        
        # Default anchor sizes for different scales
        if anchor_sizes is None:
            self.anchor_sizes = [
                [[10, 13], [16, 30], [33, 23]],      # P3 (1/8 scale)
                [[30, 61], [62, 45], [59, 119]],     # P4 (1/16 scale)
                [[116, 90], [156, 198], [373, 326]]  # P5 (1/32 scale)
            ]
        else:
            self.anchor_sizes = anchor_sizes
        
        # Loss weights
        self.lambda_obj = lambda_obj
        self.lambda_cls = lambda_cls
        self.lambda_box = lambda_box
        self.lambda_temp = lambda_temp
        
        # BCE loss with positive weights
        self.obj_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(obj_pos_weight))
        self.cls_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cls_pos_weight))
        
        # Temperature loss (MSE)
        self.temp_loss_fn = nn.MSELoss()
        
    def build_targets(self, 
                     predictions: List[torch.Tensor],
                     targets: Dict[str, torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Build targets for each scale.
        
        Args:
            predictions: List of predictions for each scale [P3, P4, P5]
            targets: Dictionary containing 'boxes', 'class_ids', 'image_ids'
            
        Returns:
            Tuple of (obj_targets, cls_targets, box_targets, temp_targets) for each scale
        """
        batch_size = predictions[0].size(0)
        device = predictions[0].device
        
        # Initialize targets for each scale
        obj_targets = []
        cls_targets = []
        box_targets = []
        temp_targets = []
        
        for scale_idx, pred in enumerate(predictions):
            # Get grid size for this scale
            grid_size = pred.size(-1)  # Assuming square grids
            stride = self.input_size // grid_size
            
            # Initialize target tensors
            obj_target = torch.zeros(batch_size, self.num_anchors, grid_size, grid_size, device=device)
            cls_target = torch.zeros(batch_size, self.num_anchors, self.num_classes, grid_size, grid_size, device=device)
            box_target = torch.zeros(batch_size, self.num_anchors, 4, grid_size, grid_size, device=device)
            temp_target = torch.zeros(batch_size, self.num_anchors, grid_size, grid_size, device=device)
            
            # Process each image in the batch
            for batch_idx in range(batch_size):
                boxes = targets['boxes'][batch_idx]
                class_ids = targets['class_ids'][batch_idx]
                
                # Filter out padded boxes (zero boxes)
                valid_mask = (boxes.sum(dim=1) > 0)
                if not valid_mask.any():
                    continue
                    
                valid_boxes = boxes[valid_mask]
                valid_class_ids = class_ids[valid_mask]
                
                # Convert boxes to grid coordinates
                grid_boxes = valid_boxes * grid_size
                
                for box_idx, (box, class_id) in enumerate(zip(grid_boxes, valid_class_ids)):
                    x_center, y_center, width, height = box
                    
                    # Find best anchor
                    best_anchor_idx = self._find_best_anchor(width, height, scale_idx)
                    
                    # Get grid cell
                    grid_x = int(x_center)
                    grid_y = int(y_center)
                    
                    # Ensure grid coordinates are within bounds
                    if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
                        # Set objectness target
                        obj_target[batch_idx, best_anchor_idx, grid_y, grid_x] = 1.0
                        
                        # Set classification target
                        cls_target[batch_idx, best_anchor_idx, class_id, grid_y, grid_x] = 1.0
                        
                        # Set box target (relative to grid cell)
                        box_target[batch_idx, best_anchor_idx, 0, grid_y, grid_x] = x_center - grid_x
                        box_target[batch_idx, best_anchor_idx, 1, grid_y, grid_x] = y_center - grid_y
                        box_target[batch_idx, best_anchor_idx, 2, grid_y, grid_x] = width / stride
                        box_target[batch_idx, best_anchor_idx, 3, grid_y, grid_x] = height / stride
                        
                        # Set temperature target (simulated thermal signature)
                        temp_target[batch_idx, best_anchor_idx, grid_y, grid_x] = self._simulate_temperature(class_id)
            
            obj_targets.append(obj_target)
            cls_targets.append(cls_target)
            box_targets.append(box_target)
            temp_targets.append(temp_target)
        
        return obj_targets, cls_targets, box_targets, temp_targets
    
    def _find_best_anchor(self, width: float, height: float, scale_idx: int) -> int:
        """Find the best matching anchor for a given box size."""
        anchors = self.anchor_sizes[scale_idx]
        best_iou = 0
        best_idx = 0
        
        for i, (anchor_w, anchor_h) in enumerate(anchors):
            # Calculate IoU between box and anchor
            box_area = width * height
            anchor_area = anchor_w * anchor_h
            
            # Simple IoU calculation
            intersection = min(width, anchor_w) * min(height, anchor_h)
            union = box_area + anchor_area - intersection
            
            iou = intersection / union if union > 0 else 0
            
            if iou > best_iou:
                best_iou = iou
                best_idx = i
        
        return best_idx
    
    def _simulate_temperature(self, class_id: int) -> float:
        """Simulate temperature based on object class."""
        # Simple temperature simulation based on object type
        temp_ranges = {
            0: (20, 25),    # person
            1: (15, 20),    # bicycle
            2: (10, 15),    # car
            3: (5, 10),     # motorcycle
            4: (15, 20),    # airplane
            5: (10, 15),    # bus
            6: (10, 15),    # train
            7: (5, 10),     # truck
            8: (20, 30),    # boat
            9: (15, 25),    # traffic light
        }
        
        if class_id in temp_ranges:
            min_temp, max_temp = temp_ranges[class_id]
            return (min_temp + max_temp) / 2
        else:
            return 20.0  # Default temperature
    
    def forward(self, 
                predictions: List[torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute YOLO loss.
        
        Args:
            predictions: List of predictions for each scale [P3, P4, P5]
            targets: Dictionary containing ground truth data
            
        Returns:
            Dictionary containing individual losses and total loss
        """
        # Build targets
        obj_targets, cls_targets, box_targets, temp_targets = self.build_targets(predictions, targets)
        
        total_obj_loss = 0
        total_cls_loss = 0
        total_box_loss = 0
        total_temp_loss = 0
        
        for scale_idx, pred in enumerate(predictions):
            batch_size, channels, height, width = pred.shape
            
            # Reshape predictions
            pred = pred.view(batch_size, self.num_anchors, -1, height, width)
            
            # Split predictions
            pred_obj = pred[:, :, 0, :, :]  # Objectness
            pred_cls = pred[:, :, 1:1+self.num_classes, :, :]  # Classification
            pred_box = pred[:, :, 1+self.num_classes:5+self.num_classes, :, :]  # Box regression
            pred_temp = pred[:, :, 5+self.num_classes, :, :]  # Temperature
            
            # Objectness loss
            obj_loss = self.obj_loss_fn(pred_obj, obj_targets[scale_idx])
            total_obj_loss += obj_loss
            
            # Classification loss (only for positive samples)
            pos_mask = obj_targets[scale_idx] > 0
            if pos_mask.any():
                # Flatten tensors for loss computation
                pred_cls_flat = pred_cls.reshape(-1, self.num_classes)
                cls_target_flat = cls_targets[scale_idx].reshape(-1, self.num_classes)
                pos_mask_flat = pos_mask.reshape(-1)
                
                # Only compute loss for positive samples
                if pos_mask_flat.any():
                    cls_loss = self.cls_loss_fn(pred_cls_flat[pos_mask_flat], cls_target_flat[pos_mask_flat])
                    total_cls_loss += cls_loss
            
            # Box regression loss (GIoU)
            if pos_mask.any():
                # Flatten tensors for loss computation
                pred_box_flat = pred_box.reshape(-1, 4)
                box_target_flat = box_targets[scale_idx].reshape(-1, 4)
                pos_mask_flat = pos_mask.reshape(-1)
                
                if pos_mask_flat.any():
                    pred_boxes = pred_box_flat[pos_mask_flat]
                    target_boxes = box_target_flat[pos_mask_flat]
                    
                    # Convert to xyxy format for GIoU calculation
                    pred_xyxy = self._xywh_to_xyxy(pred_boxes)
                    target_xyxy = self._xywh_to_xyxy(target_boxes)
                    
                    giou = bbox_iou(pred_xyxy, target_xyxy, xywh=False, GIoU=True)
                    box_loss = (1 - giou).mean()
                    total_box_loss += box_loss
            
            # Temperature loss
            if pos_mask.any():
                pred_temp_flat = pred_temp.reshape(-1)
                temp_target_flat = temp_targets[scale_idx].reshape(-1)
                pos_mask_flat = pos_mask.reshape(-1)
                
                if pos_mask_flat.any():
                    temp_loss = self.temp_loss_fn(pred_temp_flat[pos_mask_flat], temp_target_flat[pos_mask_flat])
                    total_temp_loss += temp_loss
        
        # Weighted total loss
        total_loss = (self.lambda_obj * total_obj_loss + 
                     self.lambda_cls * total_cls_loss + 
                     self.lambda_box * total_box_loss + 
                     self.lambda_temp * total_temp_loss)
        
        return {
            'total_loss': total_loss,
            'obj_loss': total_obj_loss,
            'cls_loss': total_cls_loss,
            'box_loss': total_box_loss,
            'temp_loss': total_temp_loss
        }
    
    def _xywh_to_xyxy(self, boxes: torch.Tensor) -> torch.Tensor:
        """Convert boxes from (x_center, y_center, width, height) to (x1, y1, x2, y2)."""
        x_center, y_center, width, height = boxes.unbind(-1)
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        return torch.stack([x1, y1, x2, y2], dim=-1)


if __name__ == "__main__":
    # Test loss function
    loss_fn = YOLOLoss(num_classes=80, num_anchors=3, input_size=640)
    
    # Create dummy predictions for 3 scales
    predictions = [
        torch.randn(2, 3 * (5 + 80 + 1), 80, 80),  # P3
        torch.randn(2, 3 * (5 + 80 + 1), 40, 40),  # P4
        torch.randn(2, 3 * (5 + 80 + 1), 20, 20)   # P5
    ]
    
    # Create dummy targets
    targets = {
        'boxes': torch.tensor([[[0.5, 0.5, 0.2, 0.3], [0.3, 0.7, 0.1, 0.2]]], dtype=torch.float32),
        'class_ids': torch.tensor([[0, 1]], dtype=torch.long),
        'image_ids': ['img1', 'img2']
    }
    
    # Compute loss
    loss_dict = loss_fn(predictions, targets)
    
    print("Loss components:")
    for key, value in loss_dict.items():
        print(f"{key}: {value.item():.4f}")

