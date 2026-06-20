import argparse
import os
import sys
import torch

def test(config_path, model_path, output_dir):
    # Add nanodet repository to python path
    nanodet_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'nanodet'))
    if nanodet_repo not in sys.path:
        sys.path.insert(0, nanodet_repo)

    try:
        from nanodet.util import cfg, load_config, Logger
        from nanodet.data.collate import naive_collate
        from nanodet.data.dataset import build_dataset
        from nanodet.model.arch import build_model
        from nanodet.evaluator import build_evaluator
        from torch.utils.data import DataLoader
    except ImportError as e:
        print(f"Error importing nanodet modules: {e}")
        return None

    print("Loading config...")
    load_config(cfg, config_path)
    
    # Dynamically override dataset paths to match current environment structure
    cfg.defrost()
    bird_det_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    cfg.data.val.ann_path = os.path.join(bird_det_dir, 'data', 'coco', 'annotations', 'instances_val2017.json')
    cfg.data.val.img_path = os.path.join(bird_det_dir, 'data', 'coco', 'images', 'val2017')
    cfg.freeze()
    
    # Initialize logger
    logger = Logger(-1, output_dir, False)

    print("Building dataset...")
    val_dataset = build_dataset(cfg.data.val, "val")
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.device.batchsize_per_gpu,
        shuffle=False,
        num_workers=cfg.device.workers_per_gpu,
        pin_memory=True,
        collate_fn=naive_collate,
        drop_last=False,
    )

    print("Building model...")
    model = build_model(cfg.model)
    
    print(f"Loading weights from {model_path}...")
    checkpoint = torch.load(model_path, map_location='cpu')
    # PyTorch Lightning ckpt stores weights in 'state_dict'
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    # Handle cases where the state_dict keys have 'model.' prefix (from LightningModule)
    if any(k.startswith('model.') for k in state_dict.keys()):
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        
    model.load_state_dict(state_dict, strict=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()

    print("Building evaluator...")
    evaluator = build_evaluator(cfg.evaluator, val_dataset)

    print(f"Starting evaluation on {device}...")
    from nanodet.data.batch_process import stack_batch_img
    
    all_results = {}
    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            # Preprocess batch
            batch_imgs = batch["img"]
            if isinstance(batch_imgs, list):
                batch_imgs = [img.to(device) for img in batch_imgs]
                batch_img_tensor = stack_batch_img(batch_imgs, divisible=32)
                batch["img"] = batch_img_tensor
            else:
                batch["img"] = batch["img"].to(device)
            
            # Send other tensors to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Predict
            preds = model(batch["img"])
            
            # Post process
            dets = model.head.post_process(preds, batch)
            
            # Accumulate results
            all_results.update(dets)
            
            if i % 10 == 0:
                print(f"Evaluated {i}/{len(val_dataloader)} batches")

    print("Calculating final metrics...")
    results = evaluator.evaluate(all_results, output_dir)
    print("Evaluation completed.")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test NanoDet model after training.")
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model weights (.ckpt)')
    parser.add_argument('--config-path', type=str, required=True, help='Path to nanodet config YAML file')
    parser.add_argument('--output-dir', type=str, default="eval", help='Directory for test results')

    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model weights not found: {args.model_path}")
    if not os.path.exists(args.config_path):
        raise FileNotFoundError(f"Config YAML not found: {args.config_path}")

    os.makedirs(args.output_dir, exist_ok=True)

    test(
        config_path=args.config_path,
        model_path=args.model_path,
        output_dir=args.output_dir
    )
