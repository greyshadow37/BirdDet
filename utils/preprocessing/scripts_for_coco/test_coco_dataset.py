import os
import json
import sys

def test_split(split):
    json_path = f"data/coco/annotations/instances_{split}2017.json"
    img_dir = f"data/coco/images/{split}2017"

    print(f"\n--- Testing Split: {split} ---")
    print(f"Annotation JSON: {json_path}")
    print(f"Image Directory: {img_dir}")

    if not os.path.exists(json_path):
        print(f"[ERROR] Annotation file not found: {json_path}")
        return False
    if not os.path.isdir(img_dir):
        print(f"[ERROR] Image directory not found: {img_dir}")
        return False

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to parse JSON: {e}")
        return False

    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    print(f"[OK] Parsed JSON successfully.")
    print(f"  Categories: {len(categories)} ({[c['name'] for c in categories]})")
    print(f"  Images in JSON: {len(images)}")
    print(f"  Annotations in JSON: {len(annotations)}")

    # Check images existence
    missing_files = []
    for img in images:
        fn = img.get("file_name")
        if not fn:
            print(f"[WARNING] Image entry missing 'file_name' field: {img}")
            continue
        
        # In standard COCO, file_name is relative to the split image directory
        full_path = os.path.join(img_dir, fn)
        if not os.path.exists(full_path):
            missing_files.append((fn, full_path))

    if missing_files:
        print(f"[ERROR] Mismatch: {len(missing_files)} / {len(images)} images not found on disk!")
        print("  First 5 missing paths:")
        for fn, fp in missing_files[:5]:
            print(f"    - Expected at: {fp}")
        return False
    else:
        print(f"[OK] Success: All {len(images)} images found at their standard path: {img_dir}/<file_name>")
        return True

def run_pycocotools_check():
    print("\n--- Testing pycocotools & YOLO Integration ---")
    try:
        from pycocotools.coco import COCO
        print("[OK] 'pycocotools' library is installed.")
        # Load validation annotations using COCO API
        coco = COCO("data/coco/annotations/instances_val2017.json")
        print("[OK] COCO API successfully loaded instances_val2017.json!")
    except ImportError:
        print("[INFO] 'pycocotools' not installed. You can install it using: pip install pycocotools")
    except Exception as e:
        print(f"[ERROR] Error loading with COCO API: {e}")

    try:
        from ultralytics import YOLO
        print("[OK] 'ultralytics' (YOLO) library is installed.")
    except ImportError:
        print("[INFO] 'ultralytics' not installed. You can install it using: pip install ultralytics")

def main():
    success = True
    for split in ("train", "val", "test"):
        if not test_split(split):
            success = False

    run_pycocotools_check()

    if success:
        print("\n[SUCCESS] Overall Status: COCO Dataset is fully standardized and ready for publication/training!")
        sys.exit(0)
    else:
        print("\n[ERROR] Overall Status: Mismatch or errors found in COCO dataset structure.")
        sys.exit(1)

if __name__ == "__main__":
    main()
