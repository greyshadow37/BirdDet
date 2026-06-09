'''Purpose:
- Runs inference on a folder of images (or a single image/video) using a pre-trained YOLO model and saves images with detections.
Key Functionality:
- Loads a YOLO model and processes a directory of images, a single image, or a video file.
- Saves each resulting plotted image with detections in a specified output directory.'''

from ultralytics import YOLO
import os
import argparse
from PIL import Image
import numpy as np
import sys


def parse_args():
    p = argparse.ArgumentParser(description='Run YOLO inference on a video/file and save frames with detections')
    p.add_argument('--model', type=str, required=True, help='Path to YOLO model (.pt)')
    p.add_argument('--source', type=str, required=True, help='Source video/image/directory to run inference on')
    p.add_argument('--output-dir', type=str, required=True, help='Directory to save result images')
    p.add_argument('--imgsz', type=int, default=640, help='Inference image size')
    p.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    return p.parse_args()


def save_frame_from_result(result, save_path: str):
    """Save plotted result to disk. Uses PIL fallback if necessary."""
    try:
        img = result.plot()
        if isinstance(img, np.ndarray):
            im = Image.fromarray(img)
            im.save(save_path)
        else:
            try:
                result.save(filename=save_path)
            except Exception:
                raise RuntimeError('Unable to save plotted image')
    except Exception as e:
        try:
            result.save(save_path)
        except Exception as e2:
            print(f"Failed to save frame to {save_path}: {e2}")
            return False
    return True


if __name__ == "__main__":
    args = parse_args()

    # Validate source: if it's a directory, ensure it contains supported image files
    def dir_has_images(path: str) -> bool:
        """Return True if directory contains at least one supported image file (recursively)."""
        if not os.path.isdir(path):
            return False
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp', '.dng', '.pfm', '.mpo', '.heic'}
        for root, _, files in os.walk(path):
            for f in files:
                if os.path.splitext(f)[1].lower() in exts:
                    return True
        return False

    # If user passed a directory, check it contains images and fail early with a helpful message.
    src = args.source
    if os.path.isdir(src):
        if not dir_has_images(src):
            print(f"Error: No supported images found in {src}. Place images (jpg/png/etc.) in this folder or point --source to a folder that contains images.")
            sys.exit(1)

    model = YOLO(args.model)

    os.makedirs(args.output_dir, exist_ok=True)

    # Run inference. Ultralytics accepts a folder path as source; we've already validated directories.
    results = model(args.source, imgsz=args.imgsz, conf=args.conf)

    for i, result in enumerate(results):
        save_path = os.path.join(args.output_dir, f"result_{i:06d}.jpg")
        ok = save_frame_from_result(result, save_path)
        if ok:
            print(f"Saved: {save_path}")
        else:
            print(f"Failed to save: {save_path}")
