import argparse
import os
import sys
import time
import glob
import yaml
from pathlib import Path
from ultralytics import YOLO

# Add current scripts directory to path to import metrics_utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics_utils import (
    load_yolo_ground_truths,
    load_coco_ground_truths,
    evaluate_single_image,
    compute_dataset_summary_and_classes,
    export_metrics_to_excel
)


def run_yolo_evaluation(model_path, data_yaml_path=None, imgsz=512, conf_thresh=0.25, iou_thresh=0.5,
                        output_dir="test_yolo_results", images_dir=None, labels_dir=None, ann_path=None,
                        excel_filename="yolo_test_evaluation.xlsx"):
    """
    Run custom image-by-image evaluation for YOLO model and export comprehensive metrics to Excel.
    Does NOT use model.val() and does NOT save bounding box overlay images.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n=======================================================")
    print(f"           YOLO Model Evaluation (Per-Image)           ")
    print(f"=======================================================")
    print(f"Model Path:      {model_path}")
    print(f"Confidence Thresh: {conf_thresh}")
    print(f"IoU Threshold:     {iou_thresh}")
    print(f"Image Size:        {imgsz}")

    # Parse dataset YAML if provided
    class_names = {}
    test_images_dir = None

    if data_yaml_path and os.path.exists(data_yaml_path):
        with open(data_yaml_path, 'r', encoding='utf-8') as f:
            data_cfg = yaml.safe_load(f)
        
        # Load class names
        names = data_cfg.get('names', {})
        if isinstance(names, list):
            class_names = {i: name for i, name in enumerate(names)}
        elif isinstance(names, dict):
            class_names = {int(k): str(v) for k, v in names.items()}
            
        # Resolve test images path
        base_path = data_cfg.get('path', '')
        test_path = data_cfg.get('test', '')
        
        if not os.path.isabs(base_path):
            base_path = os.path.abspath(os.path.join(os.path.dirname(data_yaml_path), base_path))
            
        if test_path:
            test_images_dir = os.path.join(base_path, test_path)
            if not os.path.exists(test_images_dir):
                test_images_dir = os.path.abspath(os.path.join(os.path.dirname(data_yaml_path), 'images', 'test'))
    
    if images_dir:
        test_images_dir = os.path.abspath(images_dir)

    if not test_images_dir or not os.path.exists(test_images_dir):
        raise FileNotFoundError(f"Test images directory could not be resolved or found: {test_images_dir}")

    # Gather test image files
    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG', '*.bmp')
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(test_images_dir, ext)))
    image_files = sorted(list(set(image_files)))

    if not image_files:
        raise FileNotFoundError(f"No test images found in directory: {test_images_dir}")

    print(f"Found {len(image_files)} test images in: {test_images_dir}")

    # Load Ground Truth annotations
    gts_by_image = {}
    image_shapes = {}

    if ann_path and os.path.exists(ann_path):
        print(f"Loading ground truth from COCO JSON: {ann_path}")
        gts_by_image, image_shapes, coco_cats = load_coco_ground_truths(ann_path, images_dir=test_images_dir)
        if not class_names:
            class_names = coco_cats
    else:
        print(f"Loading ground truth from YOLO format labels...")
        gts_by_image, image_shapes = load_yolo_ground_truths(test_images_dir, labels_dir=labels_dir, image_files=image_files)

    # Load YOLO Model
    print(f"Loading YOLO model weights from: {model_path}")
    model = YOLO(model_path)
    if hasattr(model, 'names') and model.names and not class_names:
        class_names = model.names

    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Starting inference and per-image evaluation...")

    per_image_records = []
    all_preds_by_image = {}

    start_eval_time = time.time()

    for idx, img_path in enumerate(image_files, 1):
        img_name = os.path.basename(img_path)
        img_gts = gts_by_image.get(img_name, [])

        # Run inference (save=False ensures NO bounding box images are saved)
        t0 = time.perf_counter()
        results = model.predict(
            source=img_path,
            imgsz=imgsz,
            conf=conf_thresh,
            save=False,
            verbose=False
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Extract predictions
        preds = []
        if results and len(results) > 0:
            res = results[0]
            boxes = res.boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clss = boxes.cls.cpu().numpy().astype(int)

                for box, conf, cls_id in zip(xyxy, confs, clss):
                    preds.append({
                        'class_id': int(cls_id),
                        'bbox': [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                        'score': float(conf)
                    })

        all_preds_by_image[img_name] = preds

        # Get image shape
        img_w, img_h = image_shapes.get(img_name, (0, 0))
        if (img_w == 0 or img_h == 0) and results and len(results) > 0:
            orig_shape = results[0].orig_shape # (h, w)
            img_h, img_w = orig_shape[0], orig_shape[1]

        # Evaluate this single image
        record = evaluate_single_image(
            image_path=img_path,
            width=img_w,
            height=img_h,
            gts=img_gts,
            preds=preds,
            class_names=class_names,
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
            latency_ms=latency_ms
        )
        per_image_records.append(record)

        if idx % 25 == 0 or idx == len(image_files):
            print(f"[{idx}/{len(image_files)}] Processed: {img_name} -> GT: {record['GT Count']}, Preds: {record['Pred Count']}, TP: {record['TP']}, FP: {record['FP']}, FN: {record['FN']}, F1: {record['F1 Score']:.2f}")

    total_eval_time = time.time() - start_eval_time
    print(f"\nCompleted inference and evaluation in {total_eval_time:.2f} seconds.")

    # Compute dataset summary and per-class metrics
    summary_data, per_class_records = compute_dataset_summary_and_classes(
        per_image_records=per_image_records,
        all_gts_by_image=gts_by_image,
        all_preds_by_image=all_preds_by_image,
        class_names=class_names,
        conf_thresh=conf_thresh,
        iou_thresh=iou_thresh
    )

    # Export to Excel & CSV
    excel_path = os.path.join(output_dir, excel_filename)
    excel_out, csv_out = export_metrics_to_excel(
        per_image_records=per_image_records,
        summary_data=summary_data,
        per_class_records=per_class_records,
        excel_filepath=excel_path
    )

    print(f"\n=======================================================")
    print(f"                EVALUATION SUMMARY RESULTS              ")
    print(f"=======================================================")
    for row in summary_data:
        print(f"  {row['Metric']:<38}: {row['Value']}")
    print(f"=======================================================")
    print(f"[SUCCESS] Multi-sheet Excel metrics saved to: {excel_out}")
    print(f"[SUCCESS] Companion CSV per-image metrics saved to: {csv_out}")
    print(f"=======================================================\n")

    return {
        "per_image": per_image_records,
        "summary": summary_data,
        "per_class": per_class_records,
        "excel_path": excel_out,
        "csv_path": csv_out
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLO model per-image and output metrics to Excel.")
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model weights (.pt)')
    parser.add_argument('--data-path', type=str, default=None, help='Path to dataset YAML file')
    parser.add_argument('--img-size', type=int, default=512, help='Image size (default: 512)')
    parser.add_argument('--conf-thresh', type=float, default=0.25, help='Confidence threshold for detections (default: 0.25)')
    parser.add_argument('--iou-thresh', type=float, default=0.5, help='IoU threshold for evaluation matching (default: 0.5)')
    parser.add_argument('--output-dir', type=str, default="test_yolo_results", help='Directory for output results')
    parser.add_argument('--images-dir', type=str, default=None, help='Path to test images folder')
    parser.add_argument('--labels-dir', type=str, default=None, help='Path to test labels folder (YOLO format)')
    parser.add_argument('--ann-path', type=str, default=None, help='Path to COCO format annotations JSON (optional)')
    parser.add_argument('--excel-filename', type=str, default="yolo_test_evaluation.xlsx", help='Name of output Excel file')

    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model weights not found: {args.model_path}")
    if args.data_path and not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Dataset YAML not found: {args.data_path}")

    run_yolo_evaluation(
        model_path=args.model_path,
        data_yaml_path=args.data_path,
        imgsz=args.img_size,
        conf_thresh=args.conf_thresh,
        iou_thresh=args.iou_thresh,
        output_dir=args.output_dir,
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        ann_path=args.ann_path,
        excel_filename=args.excel_filename
    )
