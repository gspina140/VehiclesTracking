import tensorrt as trt
import numpy as np
#import pycuda.driver as cuda
from cuda import cuda, cudart
from tensorrt import BuilderFlag

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network,logger)
onnx_file_path = '../models/average_model.onnx'
with open(onnx_file_path, 'rb') as model:
    parser.parse(model.read())
#succ = parser.parse_from_file(onnx_file_path)
    
config = builder.create_builder_config()
config.max_workspace_size = 1 << 30
config.set_flag(BuilderFlag.FP16) #abilitate fp16 precision
serialized_engine= builder.build_serialized_network(network, config)
with open("../models/average_model.engine", "wb") as f:
    f.write(serialized_engine)
