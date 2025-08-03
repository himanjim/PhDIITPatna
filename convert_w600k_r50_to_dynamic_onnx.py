import onnx
m = onnx.load("w600k_r50.onnx")
m.graph.input[0].type.tensor_type.shape.dim[0].dim_param  = "batch"
m.graph.output[0].type.tensor_type.shape.dim[0].dim_param = "batch"
onnx.save(m, "w600k_r50_dynamic.onnx")