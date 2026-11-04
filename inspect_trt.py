import tensorrt as trt

def main():
    logger = trt.Logger(trt.Logger.INFO)
    runtime = trt.Runtime(logger)
    engine_path = "model.trt"
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        print("Failed to load engine")
        return
    print(f"Engine name: {engine.name}")
    num_bindings = engine.num_io_tensors if hasattr(engine, 'num_io_tensors') else engine.num_bindings
    print(f"Num bindings: {num_bindings}")
    for i in range(num_bindings):
        if hasattr(engine, 'get_tensor_name'):
            name = engine.get_tensor_name(i)
            is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            dtype = engine.get_tensor_dtype(name)
            shape = engine.get_tensor_shape(name)
        else:
            name = engine.get_binding_name(i)
            is_input = engine.binding_is_input(i)
            dtype = engine.get_binding_dtype(i)
            shape = engine.get_binding_shape(i)
        io = "INPUT" if is_input else "OUTPUT"
        print(f"[{i}] {io} name='{name}' dtype={dtype} shape={tuple(shape)}")

if __name__ == "__main__":
    main()


