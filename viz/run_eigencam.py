import os
import sys
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn

# Add nanodet directory to path to allow import
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'nanodet'))

from ultralytics import YOLO, RTDETR

def get_eigencam(activation_tensor):
    """
    Computes EigenCAM heatmap from activation tensor.
    activation_tensor: shape (1, C, H, W)
    """
    # Detach and convert to CPU tensor
    act = activation_tensor.squeeze(0).detach().cpu()
    c, h, w = act.shape
    
    # Reshape to (C, H*W)
    reshaped = act.reshape(c, -1)
    
    # Perform SVD on transposed matrix (H*W, C)
    # The first left singular vector corresponds to the 1st principal component
    U, S, V = torch.linalg.svd(reshaped.t(), full_matrices=False)
    projection = U[:, 0].reshape(h, w)
    
    # Align direction/sign with positive activation sum
    if projection.sum() < 0:
        projection = -projection
        
    # Normalize to [0, 1]
    proj_min = projection.min()
    proj_max = projection.max()
    projection = (projection - proj_min) / (proj_max - proj_min + 1e-8)
    return projection.numpy()

def overlay_heatmap(image_path, heatmap):
    """
    Loads original image and overlays normalized heatmap.
    """
    img = cv2.imread(image_path)
    h, w, _ = img.shape
    
    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_u8 = np.uint8(255 * heatmap_resized)
    
    # Apply JET colormap
    colormap = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    
    # Superimpose heatmap with transparency
    blended = cv2.addWeighted(img, 0.6, colormap, 0.4, 0)
    return blended

def load_nanodet_model(model_path, config_path):
    from nanodet.model.arch import build_model
    from nanodet.util.config import cfg, load_config
    
    load_config(cfg, config_path)
    model = build_model(cfg.model)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint.get('state_dict', checkpoint))
    model.eval()
    return model

def main():
    weights_dir = os.path.join(PROJECT_ROOT, "best_weights", "pt_or_pth")
    pi_images_dir = os.path.join(PROJECT_ROOT, "pi_images")
    output_dir = os.path.join(PROJECT_ROOT, "results", "eigencam")
    os.makedirs(output_dir, exist_ok=True)
    
    # List of images
    image_paths = glob.glob(os.path.join(pi_images_dir, "*.jpg")) + glob.glob(os.path.join(pi_images_dir, "*.png"))
    if not image_paths:
        print(f"No test images found in {pi_images_dir}!")
        return
        
    # Model configuration definitions
    models_to_run = [
        {"name": "yolo26", "path": os.path.join(weights_dir, "yolo26.pt"), "type": "yolo"},
        {"name": "yolov10", "path": os.path.join(weights_dir, "yolov10.pt"), "type": "yolo"},
        {"name": "yolov11", "path": os.path.join(weights_dir, "yolov11.pt"), "type": "yolo"},
        {"name": "yolov12", "path": os.path.join(weights_dir, "yolov12.pt"), "type": "yolo"},
        {"name": "RT-DETR", "path": os.path.join(weights_dir, "RT-DETR.pt"), "type": "rtdetr"},
        {"name": "nanodet", "path": os.path.join(weights_dir, "nanodet_model_best.pth"), "type": "nanodet", 
         "config": os.path.join(PROJECT_ROOT, "BirdDet", "results", "train", "NanoDet", "logs-2025-12-18-01-29-46", "train_cfg.yml")}
    ]
    
    for m_cfg in models_to_run:
        m_name = m_cfg["name"]
        m_path = m_cfg["path"]
        
        if not os.path.exists(m_path):
            print(f"Model file {m_path} not found. Skipping...")
            continue
            
        print(f"\n--- Running EigenCAM for model: {m_name} ---")
        
        # Load model and hook target head layer
        captured_features = []
        
        if m_cfg["type"] == "yolo":
            model = YOLO(m_path)
            pyt_model = model.model
            head_layer = pyt_model.model[-1]
            input_size = 512
        elif m_cfg["type"] == "rtdetr":
            model = RTDETR(m_path)
            pyt_model = model.model
            head_layer = pyt_model.model[-1]
            input_size = 320
        elif m_cfg["type"] == "nanodet":
            pyt_model = load_nanodet_model(m_path, m_cfg["config"])
            head_layer = pyt_model.head
            input_size = 416
            
        # Register hook
        def hook_fn(module, inp, out):
            # inp[0] is the list/tuple of feature maps
            captured_features.append(inp[0])
            
        hook_handle = head_layer.register_forward_hook(hook_fn)
        
        # Process each image
        for img_path in image_paths:
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            print(f"Processing image: {img_name}")
            
            # Load and preprocess
            orig_img = cv2.imread(img_path)
            if orig_img is None:
                continue
                
            h_orig, w_orig, _ = orig_img.shape
            
            # Prepare tensor
            img_resized = cv2.resize(orig_img, (input_size, input_size))
            img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0)  # Shape: (1, 3, input_size, input_size)
            
            # Forward pass
            captured_features.clear()
            with torch.no_grad():
                if m_cfg["type"] in ["yolo", "rtdetr"]:
                    pyt_model(img_tensor)
                else:
                    pyt_model(img_tensor)
                    
            if not captured_features:
                print(f"Warning: Failed to capture feature maps for {m_name}")
                continue
                
            # Extract feature maps
            fmaps = captured_features[0]
            
            # Generate EigenCAM for the first scale
            target_fmap = fmaps[0]
            heatmap = get_eigencam(target_fmap)
            
            # Overlay and save
            result_img = overlay_heatmap(img_path, heatmap)
            out_path = os.path.join(output_dir, f"{m_name}_{img_name}_eigencam.jpg")
            cv2.imwrite(out_path, result_img)
            print(f"Saved EigenCAM result to: {out_path}")
            
        # Remove hook
        hook_handle.remove()

if __name__ == "__main__":
    main()
