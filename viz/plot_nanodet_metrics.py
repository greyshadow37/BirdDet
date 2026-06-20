import os
import re
import matplotlib.pyplot as plt
from collections import defaultdict

def parse_nanodet_logs(log_path):
    train_losses = defaultdict(list)
    val_losses = defaultdict(list)
    val_metrics = defaultdict(list)
    
    # We will average train losses per epoch
    epoch_train_losses = defaultdict(lambda: defaultdict(list))
    
    val_epochs = []
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        # Parse train loss
        if "Train|Epoch" in line:
            epoch_match = re.search(r'Epoch(\d+)/', line)
            if epoch_match:
                epoch = int(epoch_match.group(1))
                
                loss_qfl = re.search(r'loss_qfl:([\d\.]+)', line)
                loss_bbox = re.search(r'loss_bbox:([\d\.]+)', line)
                loss_dfl = re.search(r'loss_dfl:([\d\.]+)', line)
                
                if loss_qfl: epoch_train_losses[epoch]['loss_qfl'].append(float(loss_qfl.group(1)))
                if loss_bbox: epoch_train_losses[epoch]['loss_bbox'].append(float(loss_bbox.group(1)))
                if loss_dfl: epoch_train_losses[epoch]['loss_dfl'].append(float(loss_dfl.group(1)))
                
        # Parse val loss
        elif "Val|Epoch" in line:
            epoch_match = re.search(r'Epoch(\d+)/', line)
            if epoch_match:
                epoch = int(epoch_match.group(1))
                if epoch not in val_epochs:
                    val_epochs.append(epoch)
                    
                loss_qfl = re.search(r'loss_qfl:([\d\.]+)', line)
                loss_bbox = re.search(r'loss_bbox:([\d\.]+)', line)
                loss_dfl = re.search(r'loss_dfl:([\d\.]+)', line)
                
                # Val loss might have multiple iterations per epoch, we just take the last or average
                if loss_qfl: val_losses['loss_qfl'].append(float(loss_qfl.group(1)))
                if loss_bbox: val_losses['loss_bbox'].append(float(loss_bbox.group(1)))
                if loss_dfl: val_losses['loss_dfl'].append(float(loss_dfl.group(1)))
                
        # Parse val metrics
        elif "Val_metrics:" in line:
            map_match = re.search(r"'mAP': ([\d\.]+)", line)
            ap50_match = re.search(r"'AP_50': ([\d\.]+)", line)
            if map_match: val_metrics['mAP'].append(float(map_match.group(1)))
            if ap50_match: val_metrics['AP_50'].append(float(ap50_match.group(1)))

    # Average train losses per epoch
    train_epochs = sorted(epoch_train_losses.keys())
    for ep in train_epochs:
        for k in ['loss_qfl', 'loss_bbox', 'loss_dfl']:
            train_losses[k].append(sum(epoch_train_losses[ep][k]) / len(epoch_train_losses[ep][k]))
            
    return train_epochs, train_losses, val_epochs, val_losses, val_metrics

def plot_yolo_style_metrics(log_path, output_path):
    train_epochs, train_losses, val_epochs, val_losses, val_metrics = parse_nanodet_logs(log_path)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('NanoDet Training Metrics (YOLO-style)', fontsize=16, fontweight='bold')
    
    # Define plots
    plots = [
        (0, 0, 'train/loss_qfl', train_epochs, train_losses['loss_qfl']),
        (0, 1, 'train/loss_bbox', train_epochs, train_losses['loss_bbox']),
        (0, 2, 'train/loss_dfl', train_epochs, train_losses['loss_dfl']),
        
        (1, 0, 'val/loss_qfl', val_epochs, val_losses['loss_qfl']),
        (1, 1, 'val/loss_bbox', val_epochs, val_losses['loss_bbox']),
        (1, 2, 'val/loss_dfl', val_epochs, val_losses['loss_dfl']),
        
        (0, 3, 'metrics/mAP50', val_epochs, val_metrics['AP_50']),
        (1, 3, 'metrics/mAP50-95', val_epochs, val_metrics['mAP'])
    ]
    
    for row, col, title, x, y in plots:
        ax = axes[row, col]
        if len(x) > 0 and len(y) > 0:
            # If val has fewer points, plot them as markers too
            if len(x) < 10:
                ax.plot(x, y[:len(x)], marker='o', linestyle='-', linewidth=2)
            else:
                ax.plot(x, y[:len(x)], linestyle='-', linewidth=2)
            ax.set_title(title, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_xlabel('Epoch')
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
            ax.set_title(title, fontweight='bold')
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Metrics plot saved to {output_path}")

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(project_root, "results", "train", "NanoDet")
    
    log_txt_path = None
    for item in os.listdir(log_dir):
        if item.startswith("logs-"):
            potential_log = os.path.join(log_dir, item, "logs.txt")
            if os.path.exists(potential_log):
                log_txt_path = potential_log
                break
                
    if log_txt_path:
        out_file = os.path.join(project_root, "viz", "nanodet_results_plot.png")
        plot_yolo_style_metrics(log_txt_path, out_file)
    else:
        print(f"Could not find logs.txt in {log_dir}")
