# This utility converts a fixed-batch ONNX face-recognition model into a
# batch-dynamic variant by replacing the leading input and output
# dimensions with a symbolic batch parameter. The model weights and graph
# structure remain unchanged; only the tensor-shape metadata is edited so
# that the exported model can accept variable batch sizes during
# inference.

import onnx
m = onnx.load("w600k_r50.onnx")
m.graph.input[0].type.tensor_type.shape.dim[0].dim_param  = "batch"
m.graph.output[0].type.tensor_type.shape.dim[0].dim_param = "batch"
onnx.save(m, "w600k_r50_dynamic.onnx")
