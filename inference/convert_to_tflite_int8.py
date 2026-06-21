import os
import shutil
import subprocess
import sys
import numpy as np

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

def export_yolo_to_tflite_int8(model_path, output_dir):
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    print(f"\n=== Converting YOLO/RT-DETR model to INT8 TFLite: {model_name} ===")
    try:
        from ultralytics import YOLO, RTDETR
        if "RT-DETR" in model_name:
            model = RTDETR(model_path)
        else:
            model = YOLO(model_path)
            
        # Exporting to TFLite format with INT8 quantization enabled
        # Ultralytics automatically handles representative calibration datasets
        exported_path = model.export(format='tflite', int8=True)
        
        target_path = os.path.join(output_dir, f"{model_name}_int8.tflite")
        if os.path.exists(target_path):
            os.remove(target_path)
            
        source_tflite = None
        if os.path.isfile(exported_path):
            source_tflite = exported_path
        elif os.path.isdir(exported_path):
            # Ultralytics saves the int8 model inside the exported folder, usually named 'model-int8.tflite'
            source_tflite = os.path.join(exported_path, "model-int8.tflite")
            if not os.path.exists(source_tflite):
                # Try other common names
                source_tflite = os.path.join(exported_path, f"{model_name}-int8.tflite")
            if not os.path.exists(source_tflite):
                # List files in exported_path to find any .tflite
                files = [f for f in os.listdir(exported_path) if f.endswith(".tflite")]
                if files:
                    source_tflite = os.path.join(exported_path, files[0])
                    
        if source_tflite and os.path.exists(source_tflite):
            shutil.copy(source_tflite, target_path)
            print(f"Successfully converted and saved to: {target_path}")
            # Clean up the exported path or directory
            if os.path.isdir(exported_path):
                shutil.rmtree(exported_path, ignore_errors=True)
            elif os.path.isfile(exported_path):
                parent_dir = os.path.dirname(exported_path)
                if parent_dir.endswith("_saved_model") or parent_dir.endswith("_web_model"):
                    shutil.rmtree(parent_dir, ignore_errors=True)
                else:
                    try:
                        os.remove(exported_path)
                    except Exception:
                        pass
        else:
            print(f"Error: Could not locate TFLite model at {exported_path}")
    except Exception as e:
        print(f"Error exporting YOLO model {model_name} to INT8 TFLite: {e}")

def export_nanodet_to_tflite_int8(onnx_path, output_dir, pi_images_dir):
    print(f"\n=== Converting NanoDet model to INT8 TFLite: {onnx_path} ===")
    try:
        import onnx
        import tensorflow as tf
        
        model = onnx.load(onnx_path)
        self_contained_onnx = onnx_path.replace(".onnx", "_self_contained.onnx")
        onnx.save(model, self_contained_onnx)
        
        temp_tf_dir = onnx_path.replace(".onnx", "_temp_saved_model")
        os.makedirs(temp_tf_dir, exist_ok=True)
        
        # Use onnx2tf to generate the TF saved_model first
        print("Generating intermediate TensorFlow SavedModel via onnx2tf...")
        cmd = [
            "onnx2tf",
            "-i", self_contained_onnx,
            "-o", temp_tf_dir
        ]
        subprocess.run(cmd, check=True)
        
        # Load and convert using TFLiteConverter with representative dataset
        print("Quantizing SavedModel to INT8 TFLite using calibration images...")
        converter = tf.lite.TFLiteConverter.from_saved_model(temp_tf_dir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Define representative dataset generator
        def representative_data_gen():
            image_files = [os.path.join(pi_images_dir, f) for f in os.listdir(pi_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for img_path in image_files[:100]:  # up to 100 images
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (416, 416))  # NanoDet input size is 416x416
                    img = img.astype(np.float32) / 255.0  # Normalize to 0-1
                    # HWC to CHW if the model is NCHW (ONNX models are typically NCHW, tf converted might be NHWC or NCHW depending on conversion)
                    # onnx2tf usually outputs NHWC by default for TensorFlow compatibility, but let's check
                    # We can yield the input matching the model input shape
                    img = np.expand_dims(img, axis=0)
                    yield [img]
                    
        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        tflite_model = converter.convert()
        
        target_path = os.path.join(output_dir, "nanodet_model_best_int8.tflite")
        with open(target_path, "wb") as f:
            f.write(tflite_model)
            
        # Clean up temp files
        if os.path.exists(self_contained_onnx):
            os.remove(self_contained_onnx)
        shutil.rmtree(temp_tf_dir, ignore_errors=True)
        
        print(f"Successfully converted NanoDet to INT8 TFLite and saved to: {target_path}")
    except Exception as e:
        print(f"Error exporting NanoDet model to INT8 TFLite: {e}")

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    weights_dir = os.path.join(project_root, "best_weights")
    pt_dir = os.path.join(weights_dir, "pt_or_pth")
    onnx_dir = os.path.join(weights_dir, "onnx")
    tflite_int8_dir = os.path.join(weights_dir, "tflite_int8")
    pi_images_dir = os.path.join(project_root, "pi_images")
    
    os.makedirs(tflite_int8_dir, exist_ok=True)
    
    # YOLO & RT-DETR models in pt_or_pth
    yolo_models = ["yolo26.pt", "yolov10.pt", "yolov11.pt", "yolov12.pt", "RT-DETR.pt"]
    for ym in yolo_models:
        path = os.path.join(pt_dir, ym)
        if os.path.exists(path):
            export_yolo_to_tflite_int8(path, tflite_int8_dir)
        else:
            print(f"Model file not found: {path}")
            
    # NanoDet model in onnx
    nd_onnx = os.path.join(onnx_dir, "nanodet_model_best.onnx")
    if os.path.exists(nd_onnx):
        export_nanodet_to_tflite_int8(nd_onnx, tflite_int8_dir, pi_images_dir)
    else:
        print(f"NanoDet ONNX model not found: {nd_onnx}")

if __name__ == "__main__":
    main()
