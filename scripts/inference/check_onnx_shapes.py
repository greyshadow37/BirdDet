import os
import onnx
import sys

# Locate the best_weights directory relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
weights_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "..", "best_weights"))

files = [
    os.path.join(weights_dir, "yolov8_best.onnx"),
    os.path.join(weights_dir, "yolov9_best.onnx"),
    os.path.join(weights_dir, "yolov10_best.onnx"),
    os.path.join(weights_dir, "yolov11_best.onnx"),
    os.path.join(weights_dir, "effdet_rezipped.onnx"),
    os.path.join(weights_dir, "nanodet_model_best.onnx")
]

for f in files:
    try:
        model = onnx.load(f)
        for i in model.graph.input:
            shape = [d.dim_value for d in i.type.tensor_type.shape.dim]
            print(f"{f}: {i.name} -> {shape}")
    except Exception as e:
        print(f"Error loading {f}: {e}")
