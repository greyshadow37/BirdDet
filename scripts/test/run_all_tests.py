import os
import subprocess
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Test runner for BirdDet models.")
    parser.add_argument('--model-type', type=str, default='all', choices=['all', 'nanodet', 'yolo', 'rtdetr'], help='Filter by model type to test')
    args = parser.parse_args()

    bird_det_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    train_dir = os.path.join(bird_det_dir, "results", "train")
    
    # Paths to files
    yolo_test_script = os.path.join(bird_det_dir, "scripts", "test", "test_yolo.py")
    rtdetr_test_script = os.path.join(bird_det_dir, "scripts", "test", "test_rtdetr.py")
    nanodet_test_script = os.path.join(bird_det_dir, "scripts", "test", "test_nanodet.py")
    
    yolo_data_yaml = os.path.join(bird_det_dir, "data", "yolo", "data.yaml")
    nanodet_config_yml = os.path.join(bird_det_dir, "results", "train", "NanoDet", "logs-2025-12-18-01-29-46", "train_cfg.yml")

    # We will find all best/last weights
    # Structure of found weights: (model_type, weight_path, output_dir, config_or_data_path)
    models_to_test = []

    for root, dirs, files in os.walk(train_dir):
        for file in files:
            full_path = os.path.join(root, file)
            
            # Check if this is a weight file
            is_weight = False
            model_type = None
            
            # NanoDet weights
            if "NanoDet" in root:
                if file in ["model_best.ckpt", "model_last.ckpt"]:
                    is_weight = True
                    model_type = "nanodet"
            # RT-DETR weights
            elif "RT-DETR" in root:
                if file in ["best.pt", "last.pt"]:
                    is_weight = True
                    model_type = "rtdetr"
            # YOLO weights
            elif "YOLO" in root:
                if file in ["best.pt", "last.pt"]:
                    is_weight = True
                    model_type = "yolo"
            
            if is_weight:
                if args.model_type != 'all' and model_type != args.model_type:
                    continue
                # Compute a clean unique output directory name based on the weight's location relative to results/train
                rel_path = os.path.relpath(root, train_dir)
                # Output dir will be BirdDet/results/test/<rel_path>/<filename_without_ext>
                weight_name = os.path.splitext(file)[0]
                output_dir = os.path.join(bird_det_dir, "results", "test", rel_path, weight_name)
                
                if model_type == "nanodet":
                    models_to_test.append((model_type, full_path, output_dir, nanodet_config_yml))
                elif model_type == "rtdetr":
                    models_to_test.append((model_type, full_path, output_dir, yolo_data_yaml))
                elif model_type == "yolo":
                    models_to_test.append((model_type, full_path, output_dir, yolo_data_yaml))

    print(f"Found {len(models_to_test)} models to test.")
    
    for idx, (model_type, weight_path, out_dir, cfg_or_data) in enumerate(models_to_test, 1):
        print(f"\n[{idx}/{len(models_to_test)}] Testing {model_type.upper()}:")
        print(f"  Weights: {weight_path}")
        print(f"  Output:  {out_dir}")
        
        # Build command
        if model_type == "nanodet":
            cmd = [
                sys.executable, nanodet_test_script,
                "--model-path", weight_path,
                "--config-path", cfg_or_data,
                "--output-dir", out_dir
            ]
        elif model_type == "rtdetr":
            cmd = [
                sys.executable, rtdetr_test_script,
                "--model-path", weight_path,
                "--data-path", cfg_or_data,
                "--output-dir", out_dir
            ]
        elif model_type == "yolo":
            cmd = [
                sys.executable, yolo_test_script,
                "--model-path", weight_path,
                "--data-path", cfg_or_data,
                "--output-dir", out_dir
            ]
            
        print(f"  Running: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            print("  Status: Success")
        except subprocess.CalledProcessError as e:
            print(f"  Status: Failed with exit code {e.returncode}")

if __name__ == "__main__":
    main()
