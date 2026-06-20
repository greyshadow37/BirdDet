from ultralytics import RTDETR
import argparse
import os

def test(model, data, imgsz, batch_size, output_dir):
    results = model.val(
        data=data,
        imgsz=imgsz,
        batch=batch_size,
        save_json=True,
        project=output_dir,
        name="output",
        exist_ok=False,
        split='test'
    )
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test RT-DETR model after training.")
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model weights')
    parser.add_argument('--data-path', type=str, required=True, help='Path to dataset YAML file')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for testing (default: 4)')
    parser.add_argument('--img-size', type=int, default=512, help='Image size (default: 512)')
    parser.add_argument('--output-dir', type=str, default="test_rtdetr_results", help='Directory for test results')

    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model weights not found: {args.model_path}")
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Dataset YAML not found: {args.data_path}")

    model = RTDETR(args.model_path)

    test(
        model=model,
        data=args.data_path,
        imgsz=args.img_size,
        batch_size=args.batch_size,
        output_dir=args.output_dir
    )
