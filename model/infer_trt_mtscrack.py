import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

BATCH = 1


def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

logger = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(logger, "")
engine = load_engine("mts_neural.engine")
print(engine.has_implicit_batch_dimension)
context = engine.create_execution_context()

# ------------------------------------------------
# Batch = 8 input / output shape
# 必须与 build engine 时的 batch 一致 or 在 profile 范围内

input_shape = (BATCH, 3, 256, 256)
output_shape = (BATCH, 1, 256, 256)

input_size = int(np.prod(input_shape)  * np.dtype(np.float32).itemsize)
output_size = int(np.prod(output_shape) * np.dtype(np.float32).itemsize)

# ------------------------------------------------
# allocate once
d_input = cuda.mem_alloc(input_size)
d_output = cuda.mem_alloc(output_size)

bindings = [int(d_input), int(d_output)]
stream = cuda.Stream()

# host buffers (optional but cleaner)
h_output = np.empty(output_shape, dtype=np.float32)


def infer(input_np):
    """
    input_np shape:
        (8, 1, 256, 256) float32
    """
    assert input_np.shape == input_shape
    assert input_np.dtype == np.float32

    cuda.memcpy_htod_async(d_input, input_np, stream)

    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    #print(h_output.shape)
    #h_output.reshape((1, 256, 64, 64))
    return h_output
