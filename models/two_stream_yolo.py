"""
Two-stream object detection model for RGB + thermal images.
Combines RGB and thermal features using MobileNetV3 backbones and YOLO-style detection head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small
from typing import Dict, List, Tuple, Optional


class FeatureFusionModule(nn.Module):
    """
    Fuses multi-scale features from RGB and thermal streams.
    """
    def __init__(self, rgb_channels: int, thermal_channels: int, out_channels: int):
        super().__init__()
        self.rgb_conv = nn.Conv2d(rgb_channels, out_channels, 1, bias=False)
        self.thermal_conv = nn.Conv2d(thermal_channels, out_channels, 1, bias=False)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, rgb_feat: torch.Tensor, thermal_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb_feat: RGB features [B, rgb_channels, H, W]
            thermal_feat: Thermal features [B, thermal_channels, H, W]
        Returns:
            Fused features [B, out_channels, H, W]
        """
        rgb_proj = self.rgb_conv(rgb_feat)
        thermal_proj = self.thermal_conv(thermal_feat)
        
        # Concatenate and fuse
        fused = torch.cat([rgb_proj, thermal_proj], dim=1)
        return self.fusion_conv(fused)


class DetectionHead(nn.Module):
    """
    YOLO-style detection head that predicts boxes, objectness, class, and temperature.
    """
    def __init__(self, in_channels: int, num_classes: int, num_anchors: int = 3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Each anchor predicts: [x, y, w, h, obj, class1, ..., classN, temp]
        out_channels = num_anchors * (5 + num_classes + 1)  # +1 for temperature
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * 2, out_channels, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [B, in_channels, H, W]
        Returns:
            Detection predictions [B, num_anchors * (5 + num_classes + 1), H, W]
        """
        return self.conv(x)


class TwoStreamDetector(nn.Module):
    """
    Two-stream object detector for RGB + thermal images.
    
    Architecture:
    1. Separate MobileNetV3 backbones for RGB and thermal
    2. Feature fusion at multiple scales (P3, P4, P5)
    3. YOLO-style detection heads for each scale
    """
    
    def __init__(self, 
                 num_classes: int = 80,
                 input_size: int = 640,
                 pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        
        # RGB backbone (MobileNetV3-small)
        rgb_backbone = mobilenet_v3_small(pretrained=pretrained)
        self.rgb_backbone = nn.Sequential(*list(rgb_backbone.features.children()))
        
        # Thermal backbone (MobileNetV3-small, no pretrained weights)
        thermal_backbone = mobilenet_v3_small(pretrained=False)
        # Modify first layer to accept single channel input
        thermal_features = list(thermal_backbone.features.children())
        thermal_features[0] = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.thermal_backbone = nn.Sequential(*thermal_features)
        
        # Feature extraction layers for multi-scale features
        # MobileNetV3-small actual outputs: [24, 40, 576] channels
        self.rgb_feat_extractors = nn.ModuleList([
            nn.Conv2d(24, 64, 1),   # P3 (1/8 scale)
            nn.Conv2d(40, 128, 1),  # P4 (1/16 scale) 
            nn.Conv2d(576, 256, 1)  # P5 (1/32 scale)
        ])
        
        self.thermal_feat_extractors = nn.ModuleList([
            nn.Conv2d(24, 64, 1),   # P3
            nn.Conv2d(40, 128, 1),  # P4
            nn.Conv2d(576, 256, 1)  # P5
        ])
        
        # Feature fusion modules
        self.fusion_modules = nn.ModuleList([
            FeatureFusionModule(64, 64, 64),    # P3
            FeatureFusionModule(128, 128, 128), # P4
            FeatureFusionModule(256, 256, 256)   # P5
        ])
        
        # Detection heads
        self.detection_heads = nn.ModuleList([
            DetectionHead(64, num_classes),   # P3
            DetectionHead(128, num_classes),  # P4
            DetectionHead(256, num_classes)   # P5
        ])
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights for custom layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, rgb_images: torch.Tensor, thermal_images: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through the two-stream detector.
        
        Args:
            rgb_images: RGB images [B, 3, H, W]
            thermal_images: Thermal images [B, 1, H, W]
            
        Returns:
            List of detection predictions for each scale [P3, P4, P5]
        """
        batch_size = rgb_images.size(0)
        
        # Extract RGB features
        rgb_feats = []
        x_rgb = rgb_images
        for i, layer in enumerate(self.rgb_backbone):
            x_rgb = layer(x_rgb)
            # Extract features at specific layers for multi-scale detection
            if i in [2, 4, 12]:  # P3, P4, P5 layers
                rgb_feats.append(x_rgb)
        
        # Extract thermal features
        thermal_feats = []
        x_thermal = thermal_images
        for i, layer in enumerate(self.thermal_backbone):
            x_thermal = layer(x_thermal)
            # Extract features at same layers as RGB
            if i in [2, 4, 12]:  # P3, P4, P5 layers
                thermal_feats.append(x_thermal)
        
        # Process multi-scale features
        detections = []
        for i, (rgb_feat, thermal_feat) in enumerate(zip(rgb_feats, thermal_feats)):
            # Project features to common dimension
            rgb_proj = self.rgb_feat_extractors[i](rgb_feat)
            thermal_proj = self.thermal_feat_extractors[i](thermal_feat)
            
            # Fuse features
            fused_feat = self.fusion_modules[i](rgb_proj, thermal_proj)
            
            # Generate detections
            det = self.detection_heads[i](fused_feat)
            detections.append(det)
        
        return detections
    
    def get_model_size(self) -> Dict[str, int]:
        """Get model size information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }


def create_model(num_classes: int = 80, 
                input_size: int = 640, 
                pretrained: bool = True) -> TwoStreamDetector:
    """
    Create a TwoStreamDetector model.
    
    Args:
        num_classes: Number of object classes
        input_size: Input image size (assumed square)
        pretrained: Whether to use pretrained RGB backbone
        
    Returns:
        TwoStreamDetector model
    """
    return TwoStreamDetector(num_classes=num_classes, 
                           input_size=input_size, 
                           pretrained=pretrained)


if __name__ == "__main__":
    # Test model creation and forward pass
    model = create_model(num_classes=80, input_size=640)
    
    # Create dummy inputs
    rgb_input = torch.randn(2, 3, 640, 640)
    thermal_input = torch.randn(2, 1, 640, 640)
    
    # Forward pass
    with torch.no_grad():
        detections = model(rgb_input, thermal_input)
    
    print(f"Model created successfully!")
    print(f"Model size: {model.get_model_size()}")
    print(f"Number of detection scales: {len(detections)}")
    for i, det in enumerate(detections):
        print(f"Scale P{i+3} output shape: {det.shape}")

