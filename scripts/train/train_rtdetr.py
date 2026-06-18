from ultralytics import RTDETR
import argparse

def train(model, data, epochs, batch_size, imgsz, lr0, opt, output_dir, resume):
    model.train(
        data=data,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        lr0=lr0,
        half=True,
        workers=2,
        save=True,
        plots=True,
        optimizer=opt,
        patience=5,
        project=output_dir,
        name=f'{opt}_output',
        exist_ok=False,
        resume=resume
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RT-DETR model with default settings.")
    parser.add_argument('--model-path', type=str, default='rtdetr-l.pt', help='Path to the pre-trained model weights (default: rtdetr-l.pt)')
    parser.add_argument('--data-path', type=str, required=True, help='Path to the dataset YAML file')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs (default: 30)')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size (default: 4)')
    parser.add_argument('--img-size', type=int, default=512, help='Image size (default: 512)')
    parser.add_argument('--lr0', type=float, default=0.01, help='Initial learning rate (default: 0.01)')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='Optimizer to use (default: AdamW)')
    parser.add_argument('--output-dir', type=str, help='Output directory for results')
    parser.add_argument('--resume', action='store_true', help='Resume training from the last checkpoint')

    args = parser.parse_args()

    # Load model
    model = RTDETR(args.model_path)

    # Train model
    train(
        model=model,
        data=args.data_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.img_size,
        lr0=args.lr0,
        opt=args.optimizer,
        output_dir=args.output_dir,
        resume=args.resume
    )
