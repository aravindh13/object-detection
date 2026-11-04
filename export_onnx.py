"""
ONNX export script for two-stream RGB + thermal object detection model.
Exports PyTorch model to ONNX format for deployment.
"""

import os
import argparse
import torch
import torch.onnx
import numpy as np
from typing import Tuple

from models.two_stream_yolo import create_model


class TwoStreamDetectorONNX(torch.nn.Module):
    """
    ONNX-compatible wrapper for TwoStreamDetector.
    Combines RGB and thermal inputs into a single tensor for ONNX export.
    """
    
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        
    def forward(self, combined_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with combined RGB + thermal input.
        
        Args:
            combined_input: Combined input tensor [B, 4, H, W] where first 3 channels are RGB, last channel is thermal
            
        Returns:
            Tuple of detection outputs for each scale [P3, P4, P5]
        """
        # Split combined input into RGB and thermal
        rgb_input = combined_input[:, :3, :, :]  # First 3 channels
        thermal_input = combined_input[:, 3:4, :, :]  # Last channel
        
        # Run model
        predictions = self.model(rgb_input, thermal_input)
        
        return predictions[0], predictions[1], predictions[2]


def export_to_onnx(model_path: str,
                   output_path: str,
                   input_size: int = 640,
                   num_classes: int = 80,
                   batch_size: int = 1,
                   opset_version: int = 11,
                   device: str = 'cpu') -> str:
    """
    Export PyTorch model to ONNX format.
    
    Args:
        model_path: Path to PyTorch model checkpoint
        output_path: Path to save ONNX model
        input_size: Input image size
        num_classes: Number of classes
        batch_size: Batch size for export
        opset_version: ONNX opset version
        device: Device to use for export
        
    Returns:
        Path to exported ONNX model
    """
    print(f"Exporting model to ONNX format...")
    print(f"Model path: {model_path}")
    print(f"Output path: {output_path}")
    print(f"Input size: {input_size}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
    
    # Create model
    model = create_model(
        num_classes=num_classes,
        input_size=input_size,
        pretrained=False
    )
    
    # Load checkpoint
    if os.path.exists(model_path):
        try:
            # Try loading with weights_only=True first (safer)
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        except Exception:
            # If that fails, load with weights_only=False (less secure but works with custom classes)
            print("Warning: Loading checkpoint with weights_only=False. Only use trusted checkpoints.")
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded model from {model_path}")
    else:
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    # Wrap model for ONNX export
    onnx_model = TwoStreamDetectorONNX(model)
    onnx_model.eval()
    
    # Create dummy input (combined RGB + thermal)
    dummy_input = torch.randn(batch_size, 4, input_size, input_size)
    
    # Export to ONNX
    print("Exporting to ONNX...")
    torch.onnx.export(
        onnx_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['combined_input'],
        output_names=['p3_output', 'p4_output', 'p5_output'],
        dynamic_axes={
            'combined_input': {0: 'batch_size'},
            'p3_output': {0: 'batch_size'},
            'p4_output': {0: 'batch_size'},
            'p5_output': {0: 'batch_size'}
        },
        verbose=False
    )
    
    print(f"ONNX model exported successfully to {output_path}")
    
    # Verify ONNX model
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verification passed!")
    except ImportError:
        print("ONNX not installed, skipping verification")
    except Exception as e:
        print(f"ONNX model verification failed: {e}")
    
    return output_path


def test_onnx_model(onnx_path: str, 
                   pytorch_model_path: str,
                   input_size: int = 640,
                   num_classes: int = 80,
                   batch_size: int = 1) -> bool:
    """
    Test ONNX model by comparing outputs with PyTorch model.
    
    Args:
        onnx_path: Path to ONNX model
        pytorch_model_path: Path to PyTorch model
        input_size: Input image size
        num_classes: Number of classes
        batch_size: Batch size for testing
        
    Returns:
        True if outputs match, False otherwise
    """
    try:
        import onnxruntime as ort
        
        print("Testing ONNX model...")
        
        # Create PyTorch model
        pytorch_model = create_model(
            num_classes=num_classes,
            input_size=input_size,
            pretrained=False
        )
        
        # Load PyTorch checkpoint
        try:
            # Try loading with weights_only=True first (safer)
            checkpoint = torch.load(pytorch_model_path, map_location='cpu', weights_only=True)
        except Exception:
            # If that fails, load with weights_only=False (less secure but works with custom classes)
            print("Warning: Loading checkpoint with weights_only=False. Only use trusted checkpoints.")
            checkpoint = torch.load(pytorch_model_path, map_location='cpu', weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            pytorch_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            pytorch_model.load_state_dict(checkpoint)
        
        pytorch_model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(batch_size, 4, input_size, input_size)
        
        # Get PyTorch outputs
        with torch.no_grad():
            rgb_input = dummy_input[:, :3, :, :]
            thermal_input = dummy_input[:, 3:4, :, :]
            pytorch_outputs = pytorch_model(rgb_input, thermal_input)
        
        # Get ONNX outputs
        ort_session = ort.InferenceSession(onnx_path)
        onnx_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
        onnx_outputs = ort_session.run(None, onnx_inputs)
        
        # Compare outputs
        tolerance = 1e-3
        all_match = True
        
        for i, (pytorch_out, onnx_out) in enumerate(zip(pytorch_outputs, onnx_outputs)):
            pytorch_np = pytorch_out.numpy()
            onnx_np = np.array(onnx_out)
            
            # Check shapes
            if pytorch_np.shape != onnx_np.shape:
                print(f"Shape mismatch for output {i}: PyTorch {pytorch_np.shape} vs ONNX {onnx_np.shape}")
                all_match = False
                continue
            
            # Check values
            max_diff = np.max(np.abs(pytorch_np - onnx_np))
            if max_diff > tolerance:
                print(f"Value mismatch for output {i}: max difference = {max_diff}")
                all_match = False
            else:
                print(f"Output {i} matches (max diff: {max_diff:.6f})")
        
        if all_match:
            print("ONNX model test passed!")
        else:
            print("ONNX model test failed!")
        
        return all_match
        
    except ImportError:
        print("ONNXRuntime not installed, skipping ONNX test")
        return True
    except Exception as e:
        print(f"ONNX model test failed with error: {e}")
        return False


def optimize_onnx_model(onnx_path: str, optimized_path: str) -> str:
    """
    Optimize ONNX model for better performance.
    
    Args:
        onnx_path: Path to input ONNX model
        optimized_path: Path to save optimized ONNX model
        
    Returns:
        Path to optimized ONNX model
    """
    try:
        from onnxruntime.tools import optimizer
        from onnxruntime.tools.onnx_model_utils import make_input_shape_fixed
        
        print("Optimizing ONNX model...")
        
        # Load and optimize model
        optimized_model = optimizer.optimize_model(onnx_path)
        
        # Save optimized model
        optimized_model.save_model_to_file(optimized_path)
        
        print(f"Optimized ONNX model saved to {optimized_path}")
        
        return optimized_path
        
    except ImportError:
        print("ONNXRuntime tools not available, skipping optimization")
        return onnx_path
    except Exception as e:
        print(f"ONNX optimization failed: {e}")
        return onnx_path


def main():
    parser = argparse.ArgumentParser(description='Export Two-Stream RGB+Thermal Model to ONNX')
    parser.add_argument('--model_path', type=str, required=True, help='Path to PyTorch model checkpoint')
    parser.add_argument('--output_path', type=str, default='model.onnx', help='Path to save ONNX model')
    parser.add_argument('--input_size', type=int, default=640, help='Input image size')
    parser.add_argument('--num_classes', type=int, default=80, help='Number of classes')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for export')
    parser.add_argument('--opset_version', type=int, default=11, help='ONNX opset version')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for export')
    parser.add_argument('--test', action='store_true', help='Test ONNX model after export')
    parser.add_argument('--optimize', action='store_true', help='Optimize ONNX model')
    
    args = parser.parse_args()
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output_path)
    if output_dir:  # Only create directory if there's a path
        os.makedirs(output_dir, exist_ok=True)
    
    # Export to ONNX
    onnx_path = export_to_onnx(
        model_path=args.model_path,
        output_path=args.output_path,
        input_size=args.input_size,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        opset_version=args.opset_version,
        device=args.device
    )
    
    # Test ONNX model
    if args.test:
        test_onnx_model(
            onnx_path=onnx_path,
            pytorch_model_path=args.model_path,
            input_size=args.input_size,
            num_classes=args.num_classes,
            batch_size=args.batch_size
        )
    
    # Optimize ONNX model
    if args.optimize:
        optimized_path = args.output_path.replace('.onnx', '_optimized.onnx')
        optimize_onnx_model(onnx_path, optimized_path)
    
    print("Export completed successfully!")


if __name__ == "__main__":
    main()

