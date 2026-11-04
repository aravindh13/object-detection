"""
TensorRT export script for two-stream RGB + thermal object detection model.
Converts ONNX model to TensorRT engine with FP16 and INT8 quantization support.
"""

import os
import argparse
import numpy as np
import cv2
from typing import List, Optional, Dict
import logging

# TensorRT imports
try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
    # Try to import PyCUDA, but don't fail if it's not available
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        PYCUDA_AVAILABLE = True
    except ImportError:
        PYCUDA_AVAILABLE = False
        print("Warning: PyCUDA not available. Some features may be limited.")
except ImportError:
    TENSORRT_AVAILABLE = False
    PYCUDA_AVAILABLE = False
    print("TensorRT not available. Please install TensorRT.")


class TensorRTLogger(trt.ILogger):
    """TensorRT logger for debugging."""
    
    def __init__(self):
        trt.ILogger.__init__(self)
        self.logger = logging.getLogger('TensorRT')
        self.logger.setLevel(logging.INFO)
        
    def log(self, severity, msg):
        if severity == trt.Logger.Severity.ERROR:
            self.logger.error(msg)
        elif severity == trt.Logger.Severity.WARNING:
            self.logger.warning(msg)
        elif severity == trt.Logger.Severity.INFO:
            self.logger.info(msg)
        elif severity == trt.Logger.Severity.VERBOSE:
            self.logger.debug(msg)


class CalibrationDataLoader:
    """Data loader for INT8 calibration."""
    
    def __init__(self, calibration_dir: str, batch_size: int = 1, input_size: int = 640):
        self.calibration_dir = calibration_dir
        self.batch_size = batch_size
        self.input_size = input_size
        
        # Get list of calibration images
        self.image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            self.image_files.extend([f for f in os.listdir(calibration_dir) if f.lower().endswith(ext)])
        
        self.image_files = sorted(self.image_files)
        print(f"Found {len(self.image_files)} calibration images")
        
    def __len__(self):
        return len(self.image_files) // self.batch_size
    
    def __getitem__(self, idx):
        batch_images = []
        
        for i in range(self.batch_size):
            img_idx = idx * self.batch_size + i
            if img_idx >= len(self.image_files):
                break
                
            img_path = os.path.join(self.calibration_dir, self.image_files[img_idx])
            
            # Load and preprocess image
            image = cv2.imread(img_path)
            if image is None:
                continue
                
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize
            image = cv2.resize(image, (self.input_size, self.input_size))
            
            # Normalize
            image = image.astype(np.float32) / 255.0
            image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            
            # Convert to CHW format
            image = np.transpose(image, (2, 0, 1))
            
            batch_images.append(image)
        
        # Pad batch if necessary
        while len(batch_images) < self.batch_size:
            batch_images.append(np.zeros((3, self.input_size, self.input_size), dtype=np.float32))
        
        return np.array(batch_images, dtype=np.float32)


class INT8Calibrator(trt.IInt8EntropyCalibrator2):
    """INT8 calibration class for TensorRT."""
    
    def __init__(self, calibration_data_loader: CalibrationDataLoader, cache_file: str):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.data_loader = calibration_data_loader
        self.cache_file = cache_file
        self.current_index = 0
        
        # Allocate GPU memory
        self.device_input = cuda.mem_alloc(self.data_loader.batch_size * 3 * self.data_loader.input_size * self.data_loader.input_size * 4)
        
    def get_batch_size(self):
        return self.data_loader.batch_size
    
    def get_batch(self, names):
        if self.current_index < len(self.data_loader):
            batch_data = self.data_loader[self.current_index]
            
            # Copy data to GPU
            cuda.memcpy_htod(self.device_input, batch_data)
            self.current_index += 1
            
            return [self.device_input]
        else:
            return None
    
    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None
    
    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


def build_tensorrt_engine(onnx_path: str,
                          engine_path: str,
                          precision: str = 'fp16',
                          calibration_data_dir: Optional[str] = None,
                          batch_size: int = 1,
                          input_size: int = 640,
                          max_workspace_size: int = 1 << 30) -> str:
    """
    Build TensorRT engine from ONNX model.
    
    Args:
        onnx_path: Path to ONNX model
        engine_path: Path to save TensorRT engine
        precision: Precision mode ('fp32', 'fp16', 'int8')
        calibration_data_dir: Directory containing calibration images for INT8
        batch_size: Batch size for engine
        input_size: Input image size
        max_workspace_size: Maximum workspace size in bytes
        
    Returns:
        Path to TensorRT engine
    """
    if not TENSORRT_AVAILABLE:
        raise ImportError("TensorRT not available. Please install TensorRT.")
    
    print(f"Building TensorRT engine...")
    print(f"ONNX path: {onnx_path}")
    print(f"Engine path: {engine_path}")
    print(f"Precision: {precision}")
    print(f"Batch size: {batch_size}")
    print(f"Input size: {input_size}")
    
    # Create TensorRT logger
    logger = TensorRTLogger()
    
    # Create builder
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    
    # Set workspace size (handle both old and new TensorRT APIs)
    try:
        # Newer TensorRT (8.6+)
        if hasattr(trt, 'MemoryPoolType') and hasattr(config, 'set_memory_pool_limit'):
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)
        else:
            # Older TensorRT
            config.max_workspace_size = max_workspace_size
    except Exception:
        # Fallback for environments where attribute detection is unreliable
        try:
            config.max_workspace_size = max_workspace_size
        except Exception:
            if hasattr(trt, 'MemoryPoolType') and hasattr(config, 'set_memory_pool_limit'):
                config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)
    
    # Set precision
    if precision == 'fp16':
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("FP16 precision enabled")
        else:
            print("FP16 not supported on this platform, using FP32")
    elif precision == 'int8':
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            print("INT8 precision enabled")
            
            # Set up INT8 calibration
            if calibration_data_dir and PYCUDA_AVAILABLE:
                calibration_loader = CalibrationDataLoader(calibration_data_dir, batch_size, input_size)
                calibrator = INT8Calibrator(calibration_loader, f"{engine_path}.cache")
                config.int8_calibrator = calibrator
                print(f"INT8 calibration enabled with {len(calibration_loader)} images")
            elif calibration_data_dir and not PYCUDA_AVAILABLE:
                print("Warning: PyCUDA not available, cannot perform INT8 calibration. Using FP32.")
                precision = 'fp32'
            else:
                print("Warning: No calibration data provided for INT8, using FP32")
                precision = 'fp32'
        else:
            print("INT8 not supported on this platform, using FP32")
            precision = 'fp32'
    
    # Parse ONNX model
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("Failed to parse ONNX model")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise RuntimeError("ONNX parsing failed")
    
    print("ONNX model parsed successfully")
    
    # Define optimization profile for explicit/dynamic shapes
    try:
        has_explicit_batch = (network.has_implicit_batch_dimension is False)
    except Exception:
        has_explicit_batch = True
    
    if has_explicit_batch:
        profile = builder.create_optimization_profile()
        for i in range(network.num_inputs):
            tensor = network.get_input(i)
            name = tensor.name
            dims = list(tensor.shape)
            # Build min/opt/max shapes by replacing dynamic dims (-1)
            def resolve_dim(index, value):
                if value != -1:
                    return value
                # Batch dim assumed at index 0
                if index == 0:
                    return None  # placeholder handled per-shape below
                # Channel dim (fallback to 4 if unknown)
                if index == 1:
                    return 4
                # Spatial dims fallback to input_size
                return input_size
            resolved = [resolve_dim(idx, d) for idx, d in enumerate(dims)]
            # Fill batch for min/opt/max
            min_batch = 1
            opt_batch = max(1, batch_size)
            max_batch = max(opt_batch, 4)
            def build_shape(batch_val):
                shape = []
                for idx, d in enumerate(resolved):
                    if idx == 0:
                        shape.append(batch_val)
                    else:
                        shape.append(d)
                return tuple(shape)
            min_shape = build_shape(min_batch)
            opt_shape = build_shape(opt_batch)
            max_shape = build_shape(max_batch)
            profile.set_shape(name, min=min_shape, opt=opt_shape, max=max_shape)
        config.add_optimization_profile(profile)
        print("Optimization profile added for dynamic/explicit batch network")
    
    # Build engine
    print("Building TensorRT engine...")
    engine = None
    serialized_engine = None
    
    if hasattr(builder, 'build_engine'):
        engine = builder.build_engine(network, config)
        if engine is None:
            raise RuntimeError("Failed to build TensorRT engine")
        serialized_engine = engine.serialize()
    else:
        # TensorRT 8.5+/8.6+ path
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError("Failed to build serialized TensorRT engine")
    
    # Save engine
    with open(engine_path, 'wb') as f:
        # serialized_engine may be bytes or IHostMemory
        if hasattr(serialized_engine, 'tobytes'):
            f.write(serialized_engine.tobytes())
        else:
            f.write(serialized_engine)
    
    print(f"TensorRT engine saved to {engine_path}")
    
    # Print engine info (best-effort)
    print(f"Engine built with {precision} precision")
    try:
        if engine is None:
            # Load to inspect
            logger = TensorRTLogger()
            runtime = trt.Runtime(logger)
            with open(engine_path, 'rb') as f:
                engine = runtime.deserialize_cuda_engine(f.read())
        if engine is not None and hasattr(engine, 'max_batch_size'):
            print(f"Max batch size: {engine.max_batch_size}")
    except Exception:
        pass
    
    return engine_path


def test_tensorrt_engine(engine_path: str,
                        batch_size: int = 1,
                        input_size: int = 640) -> bool:
    """
    Test TensorRT engine with dummy data.
    
    Args:
        engine_path: Path to TensorRT engine
        batch_size: Batch size for testing
        input_size: Input image size
        
    Returns:
        True if test passes, False otherwise
    """
    if not TENSORRT_AVAILABLE:
        print("TensorRT not available, skipping test")
        return True
    
    try:
        print("Testing TensorRT engine...")
        
        # Load engine
        logger = TensorRTLogger()
        runtime = trt.Runtime(logger)
        
        with open(engine_path, 'rb') as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        
        if engine is None:
            print("Failed to load TensorRT engine")
            return False
        
        # Create execution context
        context = engine.create_execution_context()
        
        # Allocate GPU memory
        inputs = []
        outputs = []
        bindings = []
        
        for i in range(engine.num_bindings):
            size = trt.volume(engine.get_binding_shape(i)) * batch_size
            dtype = trt.nptype(engine.get_binding_dtype(i))
            
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            bindings.append(int(device_mem))
            
            if engine.binding_is_input(i):
                inputs.append({'host': host_mem, 'device': device_mem})
            else:
                outputs.append({'host': host_mem, 'device': device_mem})
        
        # Create dummy input data
        dummy_input = np.random.randn(batch_size, 4, input_size, input_size).astype(np.float32)
        
        # Copy input to GPU
        np.copyto(inputs[0]['host'], dummy_input.ravel())
        cuda.memcpy_htod(inputs[0]['device'], inputs[0]['host'])
        
        # Run inference
        context.execute(batch_size, bindings)
        
        # Copy outputs back to CPU
        for output in outputs:
            cuda.memcpy_dtoh(output['host'], output['device'])
        
        print("TensorRT engine test passed!")
        return True
        
    except Exception as e:
        print(f"TensorRT engine test failed: {e}")
        return False


def benchmark_tensorrt_engine(engine_path: str,
                            batch_size: int = 1,
                            input_size: int = 640,
                            num_iterations: int = 100) -> Dict[str, float]:
    """
    Benchmark TensorRT engine performance.
    
    Args:
        engine_path: Path to TensorRT engine
        batch_size: Batch size for benchmarking
        input_size: Input image size
        num_iterations: Number of iterations for benchmarking
        
    Returns:
        Dictionary containing performance metrics
    """
    if not TENSORRT_AVAILABLE:
        print("TensorRT not available, skipping benchmark")
        return {}
    
    try:
        print(f"Benchmarking TensorRT engine ({num_iterations} iterations)...")
        
        # Load engine
        logger = TensorRTLogger()
        runtime = trt.Runtime(logger)
        
        with open(engine_path, 'rb') as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        
        # Create execution context
        context = engine.create_execution_context()
        
        # Allocate GPU memory
        inputs = []
        outputs = []
        bindings = []
        
        for i in range(engine.num_bindings):
            size = trt.volume(engine.get_binding_shape(i)) * batch_size
            dtype = trt.nptype(engine.get_binding_dtype(i))
            
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            bindings.append(int(device_mem))
            
            if engine.binding_is_input(i):
                inputs.append({'host': host_mem, 'device': device_mem})
            else:
                outputs.append({'host': host_mem, 'device': device_mem})
        
        # Warm up
        dummy_input = np.random.randn(batch_size, 4, input_size, input_size).astype(np.float32)
        np.copyto(inputs[0]['host'], dummy_input.ravel())
        cuda.memcpy_htod(inputs[0]['device'], inputs[0]['host'])
        
        for _ in range(10):
            context.execute(batch_size, bindings)
        
        # Benchmark
        cuda.Context.synchronize()
        start_time = cuda.Event()
        end_time = cuda.Event()
        
        start_time.record()
        
        for _ in range(num_iterations):
            context.execute(batch_size, bindings)
        
        end_time.record()
        cuda.Context.synchronize()
        
        # Calculate metrics
        elapsed_time = start_time.time_since(end_time) / 1000.0  # Convert to seconds
        avg_time = elapsed_time / num_iterations
        fps = batch_size / avg_time
        
        metrics = {
            'avg_inference_time_ms': avg_time * 1000,
            'fps': fps,
            'total_time_s': elapsed_time
        }
        
        print(f"Average inference time: {metrics['avg_inference_time_ms']:.2f} ms")
        print(f"FPS: {metrics['fps']:.2f}")
        
        return metrics
        
    except Exception as e:
        print(f"Benchmarking failed: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(description='Export ONNX Model to TensorRT Engine')
    parser.add_argument('--onnx_path', type=str, required=True, help='Path to ONNX model')
    parser.add_argument('--engine_path', type=str, default='model.trt', help='Path to save TensorRT engine')
    parser.add_argument('--precision', type=str, default='fp16', choices=['fp32', 'fp16', 'int8'], help='Precision mode')
    parser.add_argument('--calibration_dir', type=str, help='Directory containing calibration images for INT8')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for engine')
    parser.add_argument('--input_size', type=int, default=640, help='Input image size')
    parser.add_argument('--max_workspace_size', type=int, default=1024, help='Max workspace size in MB')
    parser.add_argument('--test', action='store_true', help='Test TensorRT engine after build')
    parser.add_argument('--benchmark', action='store_true', help='Benchmark TensorRT engine')
    parser.add_argument('--num_iterations', type=int, default=100, help='Number of iterations for benchmarking')
    
    args = parser.parse_args()
    
    if not TENSORRT_AVAILABLE:
        print("TensorRT not available. Please install TensorRT.")
        return
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.engine_path)
    if output_dir:  # Only create directory if there's a path
        os.makedirs(output_dir, exist_ok=True)
    
    # Build TensorRT engine
    engine_path = build_tensorrt_engine(
        onnx_path=args.onnx_path,
        engine_path=args.engine_path,
        precision=args.precision,
        calibration_data_dir=args.calibration_dir,
        batch_size=args.batch_size,
        input_size=args.input_size,
        max_workspace_size=args.max_workspace_size << 20  # Convert MB to bytes
    )
    
    # Test engine
    if args.test:
        test_tensorrt_engine(
            engine_path=engine_path,
            batch_size=args.batch_size,
            input_size=args.input_size
        )
    
    # Benchmark engine
    if args.benchmark:
        benchmark_tensorrt_engine(
            engine_path=engine_path,
            batch_size=args.batch_size,
            input_size=args.input_size,
            num_iterations=args.num_iterations
        )
    
    print("TensorRT export completed successfully!")


if __name__ == "__main__":
    main()

