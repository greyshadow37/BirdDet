import os
import sys
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'nanodet'))

def get_eigencam(activation_tensor):
    act = activation_tensor.squeeze(0).detach().cpu()
    c, h, w = act.shape
    reshaped = act.reshape(c, -1)
    U, S, V = torch.linalg.svd(reshaped.t(), full_matrices=False)
    projection = U[:, 0].reshape(h, w)
    if projection.sum() < 0:
        projection = -projection
    proj_min = projection.min()
    proj_max = projection.max()
    projection = (projection - proj_min) / (proj_max - proj_min + 1e-8)
    return projection.numpy()

def overlay_heatmap(image_path, heatmap):
    img = cv2.imread(image_path)
    h, w, _ = img.shape
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_u8 = np.uint8(255 * heatmap_resized)
    colormap = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(img, 0.6, colormap, 0.4, 0)
    return blended

def load_nanodet_model(model_path, config_path):
    from nanodet.model.arch import build_model
    from nanodet.util.config import cfg, load_config
    load_config(cfg, config_path)
    model = build_model(cfg.model)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('state_dict', checkpoint)
    if any(k.startswith('model.') for k in state_dict.keys()):
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser(description='Run SVD-based EigenCAM visualizations.')
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--config-path', type=str, required=True)
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--images_dir', type=str, default=os.path.join(PROJECT_ROOT, 'BirdDet', 'data', 'yolo', 'images', 'test'))
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    image_paths = glob.glob(os.path.join(args.images_dir, '*.jpg')) + glob.glob(os.path.join(args.images_dir, '*.png'))
    if not image_paths:
        print(f'No test images found in {args.images_dir}!')
        return
        
    pyt_model = load_nanodet_model(args.model_path, args.config_path)
    head_layer = pyt_model.head
    input_size = 416
    captured_features = []
    
    def hook_fn(module, inp, out):
        captured_features.append(inp[0])
        
    hook_handle = head_layer.register_forward_hook(hook_fn)
    
    for img_path in image_paths:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        orig_img = cv2.imread(img_path)
        img_resized = cv2.resize(orig_img, (input_size, input_size))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)
        
        captured_features.clear()
        with torch.no_grad():
            pyt_model(img_tensor)
                
        if captured_features:
            heatmap = get_eigencam(captured_features[0][0])
            result_img = overlay_heatmap(img_path, heatmap)
            cv2.imwrite(os.path.join(args.output_dir, f'{img_name}_eigencam.jpg'), result_img)
            
    hook_handle.remove()

if __name__ == '__main__':
    main()
