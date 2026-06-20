import os
import argparse
import glob
import sys
import torch

def export_yolo(model_path):
    print(f"\n--- Exporting YOLO model: {model_path} ---")
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        model.export(format='onnx')
        print(f"Successfully exported {model_path} to ONNX")
    except ImportError:
        print("Error: 'ultralytics' library is not installed. Please run: pip install ultralytics")
    except Exception as e:
        print(f"Error exporting YOLO model {model_path}: {e}")

def export_nanodet(model_path, config_path=None, search_dir='.'):
    print(f"\n--- Exporting NanoDet model: {model_path} ---")
    try:
        # Add nanodet directory to sys.path if not there
        nanodet_dir = os.path.abspath(os.path.join(os.path.dirname(model_path), '..', 'nanodet'))
        if nanodet_dir not in sys.path:
            sys.path.insert(0, nanodet_dir)

        if not config_path:
            print("No config provided, searching in", search_dir, "...")
            yamls = glob.glob(os.path.join(search_dir, '**', '*.yml'), recursive=True) + \
                    glob.glob(os.path.join(search_dir, '**', '*.yaml'), recursive=True)
            for y in yamls:
                if 'nanodet' in os.path.basename(y).lower():
                    config_path = y
                    break
                    
        if config_path and os.path.exists(config_path):
            print(f"Using config: {config_path}")
            from nanodet.model.arch import build_model
            from nanodet.util.config import cfg, load_config
            
            load_config(cfg, config_path)
            model = build_model(cfg.model)
            
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint.get('state_dict', checkpoint))
            model.eval()
            
            output_path = model_path.replace('.pth', '.onnx')
            input_size = cfg.data.val.input_size
            dummy_input = torch.randn(1, 3, input_size[0], input_size[1])
            
            torch.onnx.export(
                model, dummy_input, output_path, 
                opset_version=11, 
                input_names=['data'], 
                output_names=['output']
            )
            print(f"Successfully exported NanoDet to {output_path}")
        else:
            print("Note: NanoDet requires a config file to rebuild the architecture for export.")
            print("Could not auto-find it. Please pass --nanodet_cfg <path>.")
    except ImportError:
        print("Error: 'nanodet' library is not installed.")
    except Exception as e:
        print(f"Error exporting NanoDet model {model_path}: {e}")

def export_effdet(model_path):
    print(f"\n--- Exporting EfficientDet model: {model_path} ---")
    try:
        # Import EfficientDet
        try:
            import effdet
            from effdet import create_model
        except ImportError:
            print("Error: 'effdet' library not installed. Please run: pip install effdet")
            return

        print("Attempting to load PyTorch checkpoint...")
        
        model = None
        # Try FastAI load_learner first
        if model_path.endswith('.pkl'):
            try:
                from fastai.learner import load_learner
                print("Detected .pkl, attempting to load as FastAI Learner...")
                learn = load_learner(model_path)
                model = learn.model
            except ImportError:
                print("fastai not installed, cannot load fastai learner.")
            except Exception as e:
                print(f"FastAI load failed: {e}")

        # Fallback to standard torch.load
        if model is None:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            
            if hasattr(checkpoint, 'model'): # FastAI Learner fallback if unpickled directly
                model = checkpoint.model
            elif isinstance(checkpoint, torch.nn.Module):
                model = checkpoint
            elif isinstance(checkpoint, dict):
                print("Found state_dict. Attempting to initialize model...")
                # EffDet usually requires the model name (e.g. tf_efficientdet_d0). 
                arch = checkpoint.get('config', {}).get('architecture', 'tf_efficientdet_d0')
                if 'config' not in checkpoint:
                    print("Architecture not found in config, defaulting to 'tf_efficientdet_d0'")
                
                model = create_model(arch, bench_task='', num_classes=5) # Bypass NMS for ONNX export
                model.load_state_dict(checkpoint.get('state_dict', checkpoint), strict=False)
            else:
                print("Unrecognized checkpoint format.")
                return

        model.eval()
        dummy_input = torch.randn(1, 3, 512, 512) # Default image size
        
        output_path = model_path.replace('.pkl', '.onnx').replace('.pth', '.onnx')
        if output_path == model_path:
            output_path += ".onnx"
            
        torch.onnx.export(model, dummy_input, output_path, opset_version=16)
        print(f"Successfully exported EfficientDet model to {output_path}")
            
    except Exception as e:
        print(f"Error exporting EfficientDet model {model_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Convert model best weights to ONNX format.")
    parser.add_argument('--weights_dir', type=str, default='best_weights', help='Base directory containing the weights')
    parser.add_argument('--nanodet_cfg', type=str, default=None, help='Path to nanodet config file')
    parser.add_argument('--skip_yolo', action='store_true', help='Skip YOLO models conversion')
    
    args = parser.parse_args()

    # List of models specified by the user
    yolo_models = [
        'yolov8_best.pt',
        'yolov9_best.pt',
        'yolov10_best.pt',
        'yolov11_best.pt'
    ]
    nanodet_model = 'nanodet_model_best.pth'
    effdet_model = 'effdet_rezipped.pth'

    # Convert YOLO models
    if not args.skip_yolo:
        for ym in yolo_models:
            path = os.path.join(args.weights_dir, ym)
            if os.path.exists(path):
                export_yolo(path)
            else:
                print(f"\nModel file not found: {path}")

    # Convert NanoDet model
    nd_path = os.path.join(args.weights_dir, nanodet_model)
    if os.path.exists(nd_path):
        export_nanodet(nd_path, args.nanodet_cfg)
    else:
        print(f"\nModel file not found: {nd_path}")

    # Convert EfficientDet model
    ed_path = os.path.join(args.weights_dir, effdet_model)
    if os.path.exists(ed_path):
        export_effdet(ed_path)
    else:
        print(f"\nModel file not found: {ed_path}")

if __name__ == '__main__':
    main()
