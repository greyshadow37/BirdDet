import argparse
import os
import yaml
from ultralytics import RTDETR

def test(model, data_yaml_path, imgsz, output_dir, images_dir=None):
    if images_dir:
        test_images_dir = os.path.abspath(images_dir)
    else:
        # Load dataset configuration
        with open(data_yaml_path, 'r') as f:
            data_cfg = yaml.safe_load(f)
        
        # Resolve absolute path to test images
        base_path = data_cfg.get('path', '')
        test_path = data_cfg.get('test', '')
        
        if not os.path.isabs(base_path):
            base_path = os.path.abspath(os.path.join(os.path.dirname(data_yaml_path), base_path))
            
        test_images_dir = os.path.join(base_path, test_path)
        if not os.path.exists(test_images_dir):
            # Fallback to local relative path check
            test_images_dir = os.path.abspath(os.path.join(os.path.dirname(data_yaml_path), 'images', 'test'))
        
    print(f"Running predictions on test images from: {test_images_dir}")
    
    # Run predictions and save annotated images with bounding boxes
    results = model.predict(
        source=test_images_dir,
        imgsz=imgsz,
        save=True,
        project=output_dir,
        name="predictions",
        exist_ok=True,
        conf=0.25
    )
    
    predictions_output_dir = os.path.abspath(os.path.join(output_dir, "predictions"))
    print(f"Testing complete. Bounding box predictions saved to: {predictions_output_dir}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test RT-DETR model by generating bounding boxes.")
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model weights')
    parser.add_argument('--data-path', type=str, required=True, help='Path to dataset YAML file')
    parser.add_argument('--img-size', type=int, default=512, help='Image size (default: 512)')
    parser.add_argument('--output-dir', type=str, default="test_rtdetr_results", help='Directory for test results')
    parser.add_argument('--images-dir', type=str, default=None, help='Path to test images folder')

    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model weights not found: {args.model_path}")
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Dataset YAML not found: {args.data_path}")

    model = RTDETR(args.model_path)

    test(
        model=model,
        data_yaml_path=args.data_path,
        imgsz=args.img_size,
        output_dir=args.output_dir,
        images_dir=args.images_dir
    )
