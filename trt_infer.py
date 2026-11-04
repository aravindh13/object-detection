import os
import cv2
import numpy as np
import tensorrt as trt
from cuda import cudart

ENGINE_PATH = "model.trt"
RGB_PATH = os.path.join("data", "rgb", "sample_000.jpg")
THERMAL_PATH = os.path.join("data", "thermal", "sample_000.jpg")
INPUT_NAME = "combined_input"
INPUT_SIZE = 640


def preprocess(rgb_path: str, thermal_path: str, input_size: int) -> np.ndarray:
    rgb = cv2.imread(rgb_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    thr = cv2.imread(thermal_path, cv2.IMREAD_GRAYSCALE)
    if rgb is None or thr is None:
        raise RuntimeError("Failed to load input images")
    rgb = cv2.resize(rgb, (input_size, input_size))
    thr = cv2.resize(thr, (input_size, input_size))
    rgb = rgb.astype(np.float32) / 255.0
    rgb = (rgb - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
    thr = thr.astype(np.float32) / 255.0
    thr = (thr - 0.5) / 0.5
    thr = thr[:, :, None]
    combined = np.concatenate([rgb, thr], axis=2)  # HWC, 4 channels
    combined = np.transpose(combined, (2, 0, 1))  # CHW
    combined = combined[None, ...]  # NCHW
    return combined.astype(np.float32)


def main():
    logger = trt.Logger(trt.Logger.INFO)
    runtime = trt.Runtime(logger)
    with open(ENGINE_PATH, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError("Failed to load TensorRT engine")

    context = engine.create_execution_context()

    # Build tensor list compatible with TRT 10 API if available
    use_new_api = hasattr(engine, "get_tensor_name")

    # Set input shape
    input_shape = (1, 4, INPUT_SIZE, INPUT_SIZE)
    if use_new_api:
        context.set_input_shape(INPUT_NAME, input_shape)
    else:
        # Older API uses implicit shapes via bindings; ensure binding 0 matches
        pass

    # Prepare input/output buffers
    host_input = preprocess(RGB_PATH, THERMAL_PATH, INPUT_SIZE)

    if use_new_api:
        input_name = INPUT_NAME
        output_names = []
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                output_names.append(name)
    else:
        input_name = engine.get_binding_name(0)
        output_names = [engine.get_binding_name(i) for i in range(engine.num_bindings) if not engine.binding_is_input(i)]

    # Allocate device buffers
    _, d_input = cudart.cudaMalloc(host_input.nbytes)
    outputs_host = []
    outputs_dev = []
    for name in output_names:
        if use_new_api:
            out_shape = context.get_tensor_shape(name)
        else:
            out_shape = engine.get_binding_shape(engine.get_binding_index(name))
        out_size = int(np.prod(out_shape))
        host_out = np.empty(out_size, dtype=np.float32)
        outputs_host.append((name, host_out, out_shape))
        _, d_out = cudart.cudaMalloc(host_out.nbytes)
        outputs_dev.append((name, d_out))

    # Copy input to device
    cudart.cudaMemcpy(d_input, host_input.ctypes.data, host_input.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    # Bind tensors and run
    if use_new_api:
        context.set_tensor_address(input_name, int(d_input))
        for name, d_out in outputs_dev:
            context.set_tensor_address(name, int(d_out))
        ok = context.execute_async_v3(0)
        if not ok:
            raise RuntimeError("TensorRT execution failed")
    else:
        bindings = [None] * engine.num_bindings
        bindings[engine.get_binding_index(input_name)] = int(d_input)
        for name, d_out in outputs_dev:
            bindings[engine.get_binding_index(name)] = int(d_out)
        ok = context.execute_v2(bindings)
        if not ok:
            raise RuntimeError("TensorRT execution failed")

    # Copy outputs back
    for (name, host_out, out_shape), (_, d_out) in zip(outputs_host, outputs_dev):
        cudart.cudaMemcpy(host_out.ctypes.data, d_out, host_out.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
        print(f"{name} shape={tuple(out_shape)} first_vals={host_out[:5]}")

    # Free device memory
    for _, d_out in outputs_dev:
        cudart.cudaFree(d_out)
    cudart.cudaFree(d_input)


if __name__ == "__main__":
    main()
