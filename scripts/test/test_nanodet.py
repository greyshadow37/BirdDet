import argparse
import os
import sys
import time
import glob
import cv2
import torch
from pathlib import Path

# Add current scripts directory to path to import metrics_utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics_utils import (
    load_coco_ground_truths,
    load_yolo_ground_truths,
    evaluate_single_image,
    compute_dataset_summary_and_classes,
    export_metrics_to_excel
)


def run_nanodet_evaluation(config_path, model_path, output_dir="test_nanodet_results",
                           images_dir=None, ann_path=None, conf_thresh=0.25, iou_thresh=0.5,
                           excel_filename="nanodet_test_evaluation.xlsx"):
    """
    Run custom image-by-image evaluation for NanoDet model and export comprehensive metrics to Excel.
    Does NOT save bounding box overlay images.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Add nanodet repository to python path if present
    nanodet_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'nanodet'))
    if nanodet_repo not in sys.path and os.path.exists(nanodet_repo):
        sys.path.insert(0, nanodet_repo)

    try:
        from nanodet.util import cfg, load_config
        from nanodet.data.transform import Pipeline
        from nanodet.data.batch_process import stack_batch_img
        from nanodet.data.collate import naive_collate
        from nanodet.model.arch import build_model
    except ImportError as e:
        print(f"Error importing nanodet modules: {e}")
        return None

    print(f"\n=======================================================")
    print(f"          NanoDet Model Evaluation (Per-Image)         ")
    print(f"=======================================================")
    print(f"Model Path:        {model_path}")
    print(f"Config Path:       {config_path}")
    print(f"Confidence Thresh: {conf_thresh}")
    print(f"IoU Threshold:     {iou_thresh}")

    print("Loading NanoDet config...")
    load_config(cfg, config_path)

    # Resolve annotations and image paths
    bird_det_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    resolved_ann_path = ann_path
    if not resolved_ann_path:
        # Check cfg test or val annotation path
        if hasattr(cfg.data, 'test') and hasattr(cfg.data.test, 'ann_path') and cfg.data.test.ann_path:
            resolved_ann_path = cfg.data.test.ann_path
        elif hasattr(cfg.data, 'val') and hasattr(cfg.data.val, 'ann_path') and cfg.data.val.ann_path:
            resolved_ann_path = cfg.data.val.ann_path
        else:
            resolved_ann_path = os.path.join(bird_det_dir, 'data', 'coco', 'annotations', 'instances_test2017.json')

    if not os.path.isabs(resolved_ann_path) and not os.path.exists(resolved_ann_path):
        resolved_ann_path = os.path.abspath(os.path.join(bird_det_dir, resolved_ann_path))

    resolved_images_dir = images_dir
    if not resolved_images_dir:
        if hasattr(cfg.data, 'test') and hasattr(cfg.data.test, 'img_path') and cfg.data.test.img_path:
            resolved_images_dir = cfg.data.test.img_path
        elif hasattr(cfg.data, 'val') and hasattr(cfg.data.val, 'img_path') and cfg.data.val.img_path:
            resolved_images_dir = cfg.data.val.img_path
        else:
            resolved_images_dir = os.path.join(bird_det_dir, 'data', 'coco', 'images', 'test2017')

    if not os.path.isabs(resolved_images_dir) and not os.path.exists(resolved_images_dir):
        resolved_images_dir = os.path.abspath(os.path.join(bird_det_dir, resolved_images_dir))

    # Resolve class names
    class_names = getattr(cfg, 'class_names', None)
    if not class_names:
        class_names = ['Asian Green Bee-eater', 'Indian Pitta', 'Gray Wagtail', 'Cattle Egret', 'Ruddy Shelduck']

    # Load Ground Truth annotations
    gts_by_image = {}
    image_shapes = {}

    if resolved_ann_path and os.path.exists(resolved_ann_path):
        print(f"Loading ground truth from COCO JSON: {resolved_ann_path}")
        gts_by_image, image_shapes, coco_cats = load_coco_ground_truths(resolved_ann_path, images_dir=resolved_images_dir)
        if coco_cats and not getattr(cfg, 'class_names', None):
            class_names = coco_cats
    else:
        print(f"COCO annotation file not found at {resolved_ann_path}. Attempting YOLO format labels fallback...")
        if resolved_images_dir and os.path.exists(resolved_images_dir):
            gts_by_image, image_shapes = load_yolo_ground_truths(resolved_images_dir)

    print("Building NanoDet model...")
    model = build_model(cfg.model)

    print(f"Loading weights from {model_path}...")
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)

    if any(k.startswith('model.') for k in state_dict.keys()):
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()

    val_data_cfg = getattr(cfg.data, 'test', getattr(cfg.data, 'val', None))
    pipeline = Pipeline(val_data_cfg.pipeline, val_data_cfg.keep_ratio)

    # Gather test images
    if not resolved_images_dir or not os.path.exists(resolved_images_dir):
        raise FileNotFoundError(f"Test images directory not found: {resolved_images_dir}")

    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG', '*.bmp')
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(resolved_images_dir, ext)))
    image_paths = sorted(list(set(image_paths)))

    if not image_paths:
        raise FileNotFoundError(f"No test images found in {resolved_images_dir}!")

    print(f"Found {len(image_paths)} images in {resolved_images_dir}.")
    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Starting inference and per-image evaluation on {device}...")

    per_image_records = []
    all_preds_by_image = {}

    start_eval_time = time.time()

    with torch.no_grad():
        for idx, img_path in enumerate(image_paths, 1):
            img_name = os.path.basename(img_path)
            raw_img = cv2.imread(img_path)
            if raw_img is None:
                continue

            height, width = raw_img.shape[:2]
            img_info = {
                "id": idx,
                "file_name": img_name,
                "height": height,
                "width": width
            }

            meta = dict(img_info=img_info, raw_img=raw_img, img=raw_img)
            input_size = getattr(val_data_cfg, 'input_size', [512, 512])
            meta = pipeline(None, meta, input_size)
            meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1)).to(device)
            meta = naive_collate([meta])
            meta["img"] = stack_batch_img(meta["img"], divisible=32)

            t0 = time.perf_counter()
            preds_raw = model.inference(meta)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            # Convert nanodet raw preds to standard format
            # preds_raw[0] is typically a dict: {class_id: np.ndarray([x1, y1, x2, y2, score])}
            formatted_preds = []
            if preds_raw and len(preds_raw) > 0:
                first_pred = preds_raw[0]
                if isinstance(first_pred, dict):
                    for cid, det_arr in first_pred.items():
                        if det_arr is not None and len(det_arr) > 0:
                            for det in det_arr:
                                score = float(det[4])
                                if score >= conf_thresh:
                                    formatted_preds.append({
                                        'class_id': int(cid),
                                        'bbox': [float(det[0]), float(det[1]), float(det[2]), float(det[3])],
                                        'score': score
                                    })
                elif isinstance(first_pred, (list, tuple)):
                    for cid, det_arr in enumerate(first_pred):
                        if det_arr is not None and len(det_arr) > 0:
                            for det in det_arr:
                                score = float(det[4])
                                if score >= conf_thresh:
                                    formatted_preds.append({
                                        'class_id': int(cid),
                                        'bbox': [float(det[0]), float(det[1]), float(det[2]), float(det[3])],
                                        'score': score
                                    })

            all_preds_by_image[img_name] = formatted_preds
            img_gts = gts_by_image.get(img_name, [])

            # Evaluate this single image
            record = evaluate_single_image(
                image_path=img_path,
                width=width,
                height=height,
                gts=img_gts,
                preds=formatted_preds,
                class_names=class_names,
                conf_thresh=conf_thresh,
                iou_thresh=iou_thresh,
                latency_ms=latency_ms
            )
            per_image_records.append(record)

            if idx % 25 == 0 or idx == len(image_paths):
                print(f"[{idx}/{len(image_paths)}] Processed: {img_name} -> GT: {record['GT Count']}, Preds: {record['Pred Count']}, TP: {record['TP']}, FP: {record['FP']}, FN: {record['FN']}, F1: {record['F1 Score']:.2f}")

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
    parser = argparse.ArgumentParser(description="Evaluate NanoDet model per-image and output metrics to Excel.")
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model weights (.ckpt)')
    parser.add_argument('--config-path', type=str, required=True, help='Path to nanodet config YAML file')
    parser.add_argument('--output-dir', type=str, default="test_nanodet_results", help='Directory for test results')
    parser.add_argument('--images-dir', type=str, default=None, help='Path to test images folder')
    parser.add_argument('--ann-path', type=str, default=None, help='Path to COCO format annotations JSON')
    parser.add_argument('--conf-thresh', type=float, default=0.25, help='Confidence threshold for detections (default: 0.25)')
    parser.add_argument('--iou-thresh', type=float, default=0.5, help='IoU threshold for evaluation matching (default: 0.5)')
    parser.add_argument('--excel-filename', type=str, default="nanodet_test_evaluation.xlsx", help='Name of output Excel file')

    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model weights not found: {args.model_path}")
    if not os.path.exists(args.config_path):
        raise FileNotFoundError(f"Config YAML not found: {args.config_path}")

    run_nanodet_evaluation(
        config_path=args.config_path,
        model_path=args.model_path,
        output_dir=args.output_dir,
        images_dir=args.images_dir,
        ann_path=args.ann_path,
        conf_thresh=args.conf_thresh,
        iou_thresh=args.iou_thresh,
        excel_filename=args.excel_filename
    )
