import os
import shutil
import subprocess
import sys

def export_yolo_to_ncnn(model_path, output_dir):
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    print(f"\n=== Converting YOLO/RT-DETR model to NCNN: {model_name} ===")
    
    success = False
    try:
        from ultralytics import YOLO, RTDETR
        if "RT-DETR" in model_name:
            model = RTDETR(model_path)
        else:
            model = YOLO(model_path)
            
        exported_path = model.export(format='ncnn')
        
        target_path = os.path.join(output_dir, f"{model_name}_ncnn")
        if os.path.exists(target_path):
            shutil.rmtree(target_path)
        shutil.move(exported_path, target_path)
        print(f"Successfully converted and saved to: {target_path}")
        success = True
    except Exception as e:
        print(f"Standard export failed for {model_name}: {e}")
        
    # Fallback to direct ONNX to NCNN conversion via pnnx for RT-DETR
    if not success and "RT-DETR" in model_name:
        print(f"Attempting fallback conversion for RT-DETR using its ONNX model...")
        try:
            onnx_dir = os.path.join(os.path.dirname(os.path.dirname(model_path)), "onnx")
            rtdetr_onnx = os.path.join(onnx_dir, "RT-DETR.onnx")
            if os.path.exists(rtdetr_onnx):
                target_path = os.path.join(output_dir, f"{model_name}_ncnn")
                os.makedirs(target_path, exist_ok=True)
                
                # Copy ONNX file to target directory to execute pnnx in place
                shutil.copy(rtdetr_onnx, os.path.join(target_path, "model.onnx"))
                
                print("Running pnnx directly on RT-DETR.onnx...")
                cmd = ["pnnx", "model.onnx"]
                subprocess.run(cmd, cwd=target_path, check=True)
                
                # Clean up copied ONNX file
                if os.path.exists(os.path.join(target_path, "model.onnx")):
                    os.remove(os.path.join(target_path, "model.onnx"))
                print(f"Successfully converted RT-DETR via fallback pnnx and saved to: {target_path}")
            else:
                print(f"Fallback failed: RT-DETR ONNX file not found at {rtdetr_onnx}")
        except Exception as fe:
            print(f"Fallback conversion failed for RT-DETR: {fe}")

def export_nanodet_to_ncnn(onnx_path, output_dir):
    print(f"\n=== Converting NanoDet model to NCNN: {onnx_path} ===")
    try:
        import onnx
        # Load and save as self-contained ONNX to resolve external data issues for pnnx
        print("Loading ONNX model and saving as self-contained...")
        model = onnx.load(onnx_path)
        self_contained_onnx = onnx_path.replace(".onnx", "_self_contained.onnx")
        onnx.save(model, self_contained_onnx)
        
        # Prepare output names
        model_name = "nanodet_model_best"
        target_dir = os.path.join(output_dir, f"{model_name}_ncnn")
        os.makedirs(target_dir, exist_ok=True)
        
        # Run pnnx conversion
        print("Running pnnx conversion...")
        cmd = [
            "pnnx",
            self_contained_onnx
        ]
        
        # Run in target directory so pnnx outputs are placed there
        subprocess.run(cmd, cwd=target_dir, check=True)
        
        # Clean up temporary self-contained ONNX
        if os.path.exists(self_contained_onnx):
            os.remove(self_contained_onnx)
            
        print(f"Successfully converted NanoDet and saved to: {target_dir}")
    except FileNotFoundError:
        print("Error: 'pnnx' executable not found in PATH. Please install NCNN/PNNX.")
    except Exception as e:
        print(f"Error exporting NanoDet model to NCNN: {e}")

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    weights_dir = os.path.join(project_root, "best_weights")
    pt_dir = os.path.join(weights_dir, "pt_or_pth")
    onnx_dir = os.path.join(weights_dir, "onnx")
    ncnn_dir = os.path.join(weights_dir, "ncnn")
    
    os.makedirs(ncnn_dir, exist_ok=True)
    
    # YOLO & RT-DETR models in pt_or_pth
    yolo_models = ["yolo26.pt", "yolov10.pt", "yolov11.pt", "yolov12.pt", "RT-DETR.pt"]
    for ym in yolo_models:
        path = os.path.join(pt_dir, ym)
        if os.path.exists(path):
            export_yolo_to_ncnn(path, ncnn_dir)
        else:
            print(f"Model file not found: {path}")
            
    # NanoDet model in onnx
    nd_onnx = os.path.join(onnx_dir, "nanodet_model_best.onnx")
    if os.path.exists(nd_onnx):
        export_nanodet_to_ncnn(nd_onnx, ncnn_dir)
    else:
        print(f"NanoDet ONNX model not found: {nd_onnx}")

if __name__ == "__main__":
    main()
