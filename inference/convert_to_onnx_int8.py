import os
import sys

def fix_onnx_compatibility(onnx_path):
    """Post-processes an ONNX model to force compatibility with older runtimes (like Pi)."""
    print(f"Post-processing {onnx_path} for Raspberry Pi compatibility...")
    try:
        import onnx
        model = onnx.load(onnx_path)
        modified = False
        
        # Force IR version to 9
        if model.ir_version > 9:
            print(f"  Downgrading IR version from {model.ir_version} to 9")
            model.ir_version = 9
            modified = True
            
        # Force Opset version to 17 or lower
        for opset in model.opset_import:
            if (opset.domain == "" or opset.domain == "ai.onnx") and opset.version > 17:
                print(f"  Downgrading Opset version from {opset.version} to 17")
                opset.version = 17
                modified = True
                
        if modified:
            onnx.save(model, onnx_path)
            print(f"Successfully saved compatibility changes to {onnx_path}")
        else:
            print("  Already compatible.")
    except Exception as e:
        print(f"Warning: Could not post-process {onnx_path} for compatibility: {e}")

def quantize_onnx_model(onnx_path, output_path):
    print(f"\n=== Quantizing ONNX model to INT8: {onnx_path} ===")
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        # Run dynamic quantization
        quantize_dynamic(
            model_input=onnx_path,
            model_output=output_path,
            weight_type=QuantType.QUInt8
        )
        print(f"Successfully quantized model saved to: {output_path}")
        fix_onnx_compatibility(output_path)
    except ImportError:
        print("Error: 'onnxruntime' or 'onnx' library is not installed. Please run: pip install onnxruntime-gpu onnxruntime")
    except Exception as e:
        print(f"Error during ONNX quantization: {e}")

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weights_dir = os.path.join(project_root, "best_weights")
    onnx_dir = os.path.join(weights_dir, "onnx")
    onnx_int8_dir = os.path.join(weights_dir, "onnx_int8")
    
    os.makedirs(onnx_int8_dir, exist_ok=True)
    
    # Models to convert
    models = [
        "yolo26.onnx",
        "yolov10.onnx",
        "yolov11.onnx",
        "yolov12.onnx",
        "RT-DETR.onnx",
        "nanodet_model_best.onnx"
    ]
    
    for m in models:
        path = os.path.join(onnx_dir, m)
        if os.path.exists(path):
            out_path = os.path.join(onnx_int8_dir, m.replace(".onnx", "_int8.onnx"))
            quantize_onnx_model(path, out_path)
            
            # If nanodet, copy the .data file if it exists (though dynamic quantization of the .onnx file handles constants in-memory, ORT will check it if it expects it)
            if m == "nanodet_model_best.onnx":
                data_file = path + ".data"
                if os.path.exists(data_file):
                    shutil_dest = out_path + ".data"
                    import shutil
                    shutil.copy(data_file, shutil_dest)
        else:
            print(f"ONNX model not found: {path}")

if __name__ == "__main__":
    main()
