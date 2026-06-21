import os
import shutil
import subprocess
import sys

# Mock cv2 GUI attributes if running headless to avoid Ultralytics export errors
try:
    import cv2
    if not hasattr(cv2, 'imshow'):
        cv2.imshow = lambda *args, **kwargs: None
    if not hasattr(cv2, 'waitKey'):
        cv2.waitKey = lambda *args, **kwargs: 0
    if not hasattr(cv2, 'destroyAllWindows'):
        cv2.destroyAllWindows = lambda *args, **kwargs: None
except ImportError:
    pass

def export_yolo_to_tflite(model_path, output_dir):
    print(f"\n=== Converting YOLO/RT-DETR model to TFLite: {model_path} ===")
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        # Exporting to TFLite format
        exported_path = model.export(format='tflite')
        
        # Move the exported folder/file to the target output directory
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        target_path = os.path.join(output_dir, f"{model_name}_tflite")
        if os.path.exists(target_path):
            shutil.rmtree(target_path, ignore_errors=True)
            if os.path.exists(target_path):
                os.remove(target_path)
                
        shutil.move(exported_path, target_path)
        print(f"Successfully converted and saved to: {target_path}")
    except Exception as e:
        print(f"Error exporting YOLO model {model_path} to TFLite: {e}")

def export_nanodet_to_tflite(onnx_path, output_dir):
    print(f"\n=== Converting NanoDet model to TFLite: {onnx_path} ===")
    try:
        import onnx
        # Load and save as self-contained ONNX to resolve external data issues
        model = onnx.load(onnx_path)
        self_contained_onnx = onnx_path.replace(".onnx", "_self_contained.onnx")
        onnx.save(model, self_contained_onnx)
        
        model_name = "nanodet_model_best"
        target_dir = os.path.join(output_dir, f"{model_name}_tflite")
        os.makedirs(target_dir, exist_ok=True)
        
        # Check if onnx2tf is installed
        try:
            import onnx2tf
        except ImportError:
            print("Required library 'onnx2tf' is not installed. Attempting to install...")
            subprocess.run([sys.executable, "-m", "pip", "install", "onnx2tf", "tensorflow"], check=True)
            
        print("Running onnx2tf conversion...")
        cmd = [
            "onnx2tf",
            "-i", self_contained_onnx,
            "-o", target_dir
        ]
        subprocess.run(cmd, check=True)
        
        # Clean up temporary self-contained ONNX
        if os.path.exists(self_contained_onnx):
            os.remove(self_contained_onnx)
            
        print(f"Successfully converted NanoDet to TFLite and saved to: {target_dir}")
    except Exception as e:
        print(f"Error exporting NanoDet model to TFLite: {e}")
        print("Make sure you have 'onnx2tf' and 'tensorflow' installed: pip install onnx2tf tensorflow")

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    weights_dir = os.path.join(project_root, "best_weights")
    pt_dir = os.path.join(weights_dir, "pt_or_pth")
    onnx_dir = os.path.join(weights_dir, "onnx")
    tflite_dir = os.path.join(weights_dir, "tflite")
    
    os.makedirs(tflite_dir, exist_ok=True)
    
    # YOLO & RT-DETR models in pt_or_pth
    yolo_models = ["yolo26.pt", "yolov10.pt", "yolov11.pt", "yolov12.pt", "RT-DETR.pt"]
    for ym in yolo_models:
        path = os.path.join(pt_dir, ym)
        if os.path.exists(path):
            export_yolo_to_tflite(path, tflite_dir)
        else:
            print(f"Model file not found: {path}")
            
    # NanoDet model in onnx
    nd_onnx = os.path.join(onnx_dir, "nanodet_model_best.onnx")
    if os.path.exists(nd_onnx):
        export_nanodet_to_tflite(nd_onnx, tflite_dir)
    else:
        print(f"NanoDet ONNX model not found: {nd_onnx}")

if __name__ == "__main__":
    main()
