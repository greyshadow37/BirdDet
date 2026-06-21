import os
import sys

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    weights_dir = os.path.join(project_root, "best_weights")
    onnx_dir = os.path.join(weights_dir, "onnx")
    ncnn_dir = os.path.join(weights_dir, "ncnn")
    pi_images_dir = os.path.join(project_root, "pi_images")
    
    os.makedirs(ncnn_dir, exist_ok=True)
    
    # 1. Create a list of calibration images
    images_list_path = os.path.join(weights_dir, "ncnn_calibration_list.txt")
    print(f"Creating calibration images list at: {images_list_path}")
    image_files = [f for f in os.listdir(pi_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    with open(images_list_path, "w") as f:
        for img in image_files:
            # We write relative path from the weights directory
            f.write(f"../pi_images/{img}\n")
            
    # 2. Write a shell script for running quantization on the Raspberry Pi
    # Since ncnn2table and ncnnoptimize are typically compiled on the deployment Pi
    sh_script_path = os.path.join(weights_dir, "quantize_ncnn_on_pi.sh")
    print(f"Writing Pi quantization helper script to: {sh_script_path}")
    
    sh_content = """#!/bin/bash
# Helper script to run NCNN INT8 Quantization on Raspberry Pi
# Ensure you run this from the 'best_weights' directory on the Pi.

echo "================================================="
echo "Starting NCNN INT8 Quantization on Pi..."
echo "================================================="

# Create directories
mkdir -p ncnn_int8

# Models to convert
models=("yolo26" "yolov10" "yolov11" "yolov12" "RT-DETR" "nanodet_model_best")

for model in "${models[@]}"; do
    echo ""
    echo "--- Quantizing $model ---"
    
    # Set paths
    ONNX_FILE="onnx/$model.onnx"
    PARAM_FILE="ncnn/${model}_ncnn/model.ncnn.param"
    BIN_FILE="ncnn/${model}_ncnn/model.ncnn.bin"
    TABLE_FILE="ncnn_int8/$model.table"
    OUT_PARAM="ncnn_int8/${model}_int8.param"
    OUT_BIN="ncnn_int8/${model}_int8.bin"
    
    if [ ! -f "$PARAM_FILE" ]; then
        # Handle different subfolder layouts if any
        PARAM_FILE="ncnn/${model}_ncnn/model.param"
        BIN_FILE="ncnn/${model}_ncnn/model.bin"
    fi

    # Step 1: Generate calibration table using ncnn2table
    # Note: adjust input size as per model requirements (YOLO/RT-DETR: 512x512/320x320, NanoDet: 416x416)
    if [ "$model" == "nanodet_model_best" ]; then
        img_size="416,416"
        mean_vals="103.53,116.28,123.675"
        norm_vals="0.017429,0.017507,0.017125"
    elif [ "$model" == "RT-DETR" ]; then
        img_size="320,320"
        mean_vals="103.53,116.28,123.675"
        norm_vals="0.017429,0.017507,0.017125"
    else
        img_size="512,512"
        mean_vals="103.53,116.28,123.675"
        norm_vals="0.017429,0.017507,0.017125"
    fi
    
    echo "Generating calibration table: $TABLE_FILE"
    ncnn2table "$PARAM_FILE" "$BIN_FILE" ncnn_calibration_list.txt "$TABLE_FILE" \
        size="$img_size" mean="$mean_vals" norm="$norm_vals" shape=3,${img_size//,/x} pixel=BGR thread=4
        
    # Step 2: Optimize and Quantize model to INT8
    echo "Creating quantized INT8 model..."
    ncnnoptimize "$PARAM_FILE" "$BIN_FILE" "$OUT_PARAM" "$OUT_BIN" 0 "$TABLE_FILE"
    
    echo "Successfully quantized $model!"
done

echo ""
echo "================================================="
echo "NCNN INT8 Quantization Complete!"
echo "================================================="
"""
    with open(sh_script_path, "w", newline='\n') as f:
        f.write(sh_content)
        
    print("\nNCNN INT8 calibration list and Pi helper script generated successfully.")

if __name__ == "__main__":
    main()
