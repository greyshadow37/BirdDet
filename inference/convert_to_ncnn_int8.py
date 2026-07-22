import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.out_path, exist_ok=True)

    # 1. Create a list of calibration images
    images_list_path = os.path.join(args.out_path, 'ncnn_calibration_list.txt')
    print(f"Creating calibration images list at: {images_list_path}")
    
    # Locate pi_images relative to the project root
    project_root = r"D:\Projects\Mini-Projects\Mini-Project-1"
    pi_images_dir = os.path.join(project_root, "pi_images")
    
    if os.path.exists(pi_images_dir):
        image_files = [f for f in os.listdir(pi_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        with open(images_list_path, "w") as f_out:
            for img in image_files:
                f_out.write(f"../../../pi_images/{img}\n")
    else:
        print(f"Warning: pi_images directory not found at {pi_images_dir}")

    # 2. Write a shell script for running quantization on the Raspberry Pi
    sh_script_path = os.path.join(args.out_path, 'quantize_ncnn_on_pi.sh')
    print(f"Writing Pi quantization helper script to: {sh_script_path}")

    sh_content = """#!/bin/bash
# Helper script to run NCNN INT8 Quantization on Raspberry Pi

echo "================================================="
echo "Starting NCNN INT8 Quantization on Pi..."
echo "================================================="

# Generate calibration table using ncnn2table
ncnn2table model.ncnn.param model.ncnn.bin ncnn_calibration_list.txt model.table \\
    mean="[103.53,116.28,123.675]" norm="[0.017429,0.017507,0.017125]" shape="[416,416,3]" pixel=BGR thread=4

# Optimize and Quantize model to INT8
ncnnoptimize model.ncnn.param model.ncnn.bin model_int8.param model_int8.bin 0 model.table

echo "================================================="
echo "NCNN INT8 Quantization Complete!"
echo "================================================="
"""
    with open(sh_script_path, "w", newline='\n') as f_sh:
        f_sh.write(sh_content)

if __name__ == "__main__":
    main()
