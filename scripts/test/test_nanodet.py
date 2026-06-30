import argparse
import os
import sys
import glob
import cv2
import torch

def test(config_path, model_path, output_dir, images_dir=None):
    # Add nanodet repository to python path
    nanodet_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'nanodet'))
    if nanodet_repo not in sys.path:
        sys.path.insert(0, nanodet_repo)

    try:
        from nanodet.util import cfg, load_config, Logger
        from nanodet.data.transform import Pipeline
        from nanodet.data.batch_process import stack_batch_img
        from nanodet.data.collate import naive_collate
        from nanodet.model.arch import build_model
    except ImportError as e:
        print(f"Error importing nanodet modules: {e}")
        return

    print("Loading config...")
    load_config(cfg, config_path)
    
    # Resolve directories
    if images_dir:
        test_images_dir = os.path.abspath(images_dir)
    else:
        bird_det_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        test_images_dir = os.path.join(bird_det_dir, 'data', 'coco', 'images', 'test2017')
        
    predictions_output_dir = os.path.join(output_dir, "predictions")
    os.makedirs(predictions_output_dir, exist_ok=True)
    
    print("Building model...")
    model = build_model(cfg.model)
    
    print(f"Loading weights from {model_path}...")
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    # Handle Lightning prefix
    if any(k.startswith('model.') for k in state_dict.keys()):
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        
    model.load_state_dict(state_dict, strict=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()
    
    # Define pipeline and class names
    pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)
    class_names = cfg.class_names
    if not class_names:
        class_names = ['Asian Green Bee-eater', 'Indian Pitta', 'Gray Wagtail', 'Cattle Egret', 'Ruddy Shelduck']
        
    print(f"Scanning test images in: {test_images_dir}")
    image_paths = glob.glob(os.path.join(test_images_dir, "*.jpg")) + glob.glob(os.path.join(test_images_dir, "*.png"))
    if not image_paths:
        print(f"No test images found in {test_images_dir}!")
        return
        
    print(f"Found {len(image_paths)} images. Running predictions...")
    
    with torch.no_grad():
        for i, img_path in enumerate(image_paths, 1):
            img_name = os.path.basename(img_path)
            raw_img = cv2.imread(img_path)
            if raw_img is None:
                continue
                
            height, width = raw_img.shape[:2]
            img_info = {
                "id": 0,
                "file_name": img_name,
                "height": height,
                "width": width
            }
            
            meta = dict(img_info=img_info, raw_img=raw_img, img=raw_img)
            meta = pipeline(None, meta, cfg.data.val.input_size)
            meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1)).to(device)
            meta = naive_collate([meta])
            meta["img"] = stack_batch_img(meta["img"], divisible=32)
            
            # Predict
            preds = model.inference(meta)
            
            # Draw result bounding boxes on image
            # Note: show=False to avoid headless environment display errors
            result_img = model.head.show_result(
                meta["raw_img"][0], 
                preds[0], 
                class_names, 
                score_thres=0.35, 
                show=False
            )
            
            # Save annotated image
            out_file = os.path.join(predictions_output_dir, img_name)
            cv2.imwrite(out_file, result_img)
            
            if i % 50 == 0 or i == len(image_paths):
                print(f"[{i}/{len(image_paths)}] Processed and saved: {img_name}")

    print(f"Testing complete. Bounding box predictions saved to: {predictions_output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test NanoDet model by generating bounding boxes.")
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model weights (.ckpt)')
    parser.add_argument('--config-path', type=str, required=True, help='Path to nanodet config YAML file')
    parser.add_argument('--output-dir', type=str, default="test_nanodet_results", help='Directory for test results')
    parser.add_argument('--images-dir', type=str, default=None, help='Path to test images folder')

    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model weights not found: {args.model_path}")
    if not os.path.exists(args.config_path):
        raise FileNotFoundError(f"Config YAML not found: {args.config_path}")

    test(
        config_path=args.config_path,
        model_path=args.model_path,
        output_dir=args.output_dir,
        images_dir=args.images_dir
    )
