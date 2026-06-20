import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os

def calculate_iou(box1, box2):
    # box: [x, y, w, h]
    x1_1, y1_1, w1, h1 = box1
    x2_1 = x1_1 + w1
    y2_1 = y1_1 + h1
    
    x1_2, y1_2, w2, h2 = box2
    x2_2 = x1_2 + w2
    y2_2 = y1_2 + h2
    
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    
    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h
    
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0
    return inter_area / union_area

def load_data(ann_file, res_file):
    with open(ann_file, 'r') as f:
        coco = json.load(f)
    
    if os.path.exists(res_file):
        with open(res_file, 'r') as f:
            preds = json.load(f)
    else:
        preds = []
        
    categories = {cat['id']: cat['name'] for cat in coco.get('categories', [])}
    if not categories:
        categories = {0: 'Ruddy Shelduck', 1: 'Asian Green Bee-Eater', 2: 'Cattle Egret', 3: 'Gray Wagtail', 4: 'Indian Pitta'}
        
    gts_by_img = defaultdict(list)
    for ann in coco['annotations']:
        gts_by_img[ann['image_id']].append(ann)
        
    preds_by_img = defaultdict(list)
    for p in preds:
        preds_by_img[p['image_id']].append(p)
        
    return gts_by_img, preds_by_img, categories

def evaluate(gts_by_img, preds_by_img, categories, iou_thresh=0.5, conf_thresh=0.25):
    class_preds = defaultdict(list)
    class_total_gts = defaultdict(int)
    
    num_classes = max(categories.keys()) + 1
    conf_mat = np.zeros((num_classes + 1, num_classes + 1)) 
    bg_idx = num_classes
    
    for img_id, gts in gts_by_img.items():
        preds = preds_by_img.get(img_id, [])
        preds.sort(key=lambda x: x['score'], reverse=True)
        
        for gt in gts:
            class_total_gts[gt['category_id']] += 1
            
        matched_gts = set()
        for p in preds:
            best_iou = 0
            best_gt_idx = -1
            
            for i, gt in enumerate(gts):
                if gt['category_id'] == p['category_id']:
                    iou = calculate_iou(p['bbox'], gt['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = i
                        
            if best_iou >= iou_thresh and best_gt_idx not in matched_gts:
                class_preds[p['category_id']].append((p['score'], 1))
                matched_gts.add(best_gt_idx)
            else:
                class_preds[p['category_id']].append((p['score'], 0))
                
        cm_preds = [p for p in preds if p['score'] >= conf_thresh]
        matched_gts_cm = set()
        matched_preds_cm = set()
        
        iou_matrix = np.zeros((len(cm_preds), len(gts)))
        for i, p in enumerate(cm_preds):
            for j, gt in enumerate(gts):
                iou_matrix[i, j] = calculate_iou(p['bbox'], gt['bbox'])
                
        while True:
            if iou_matrix.size == 0 or np.max(iou_matrix) < iou_thresh:
                break
            i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            iou_matrix[i, :] = 0
            iou_matrix[:, j] = 0
            
            pred_cls = cm_preds[i]['category_id']
            gt_cls = gts[j]['category_id']
            conf_mat[gt_cls, pred_cls] += 1
            
            matched_preds_cm.add(i)
            matched_gts_cm.add(j)
            
        for j, gt in enumerate(gts):
            if j not in matched_gts_cm:
                conf_mat[gt['category_id'], bg_idx] += 1
                
        for i, p in enumerate(cm_preds):
            if i not in matched_preds_cm:
                conf_mat[bg_idx, p['category_id']] += 1
                
    return class_preds, class_total_gts, conf_mat

def plot_curves_and_cm(class_preds, class_total_gts, conf_mat, categories, split_name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    
    thresholds = np.linspace(0, 1, 1000)
    metrics = defaultdict(dict)
    
    for c in categories.keys():
        preds = class_preds.get(c, [])
        preds.sort(key=lambda x: x[0], reverse=True)
        total_gt = class_total_gts.get(c, 0)
        
        tps = np.cumsum([1 if x[1] == 1 else 0 for x in preds])
        fps = np.cumsum([1 if x[1] == 0 else 0 for x in preds])
        scores = np.array([x[0] for x in preds])
        
        precisions = tps / (tps + fps + 1e-16)
        recalls = tps / (total_gt + 1e-16)
        
        metrics[c]['precisions'] = precisions
        metrics[c]['recalls'] = recalls
        metrics[c]['scores'] = scores
        metrics[c]['total_gt'] = total_gt
        
    plt.figure(figsize=(10, 8))
    for c in categories.keys():
        if len(metrics[c]['recalls']) > 0:
            plt.plot(metrics[c]['recalls'], metrics[c]['precisions'], label=categories[c], linewidth=2)
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'{split_name} Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(out_dir, f'{split_name}_PR_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    mean_p = np.zeros_like(thresholds)
    mean_r = np.zeros_like(thresholds)
    mean_f1 = np.zeros_like(thresholds)
    valid_classes = 0
    
    plt.figure(figsize=(10, 8))
    for c in categories.keys():
        if len(metrics[c]['scores']) == 0: continue
        valid_classes += 1
        p_interp = np.zeros_like(thresholds)
        r_interp = np.zeros_like(thresholds)
        f1_interp = np.zeros_like(thresholds)
        
        for i, t in enumerate(thresholds):
            idx = np.searchsorted(-metrics[c]['scores'], -t)
            if idx > 0:
                p = metrics[c]['precisions'][idx-1]
                r = metrics[c]['recalls'][idx-1]
                p_interp[i] = p
                r_interp[i] = r
                f1_interp[i] = 2 * p * r / (p + r + 1e-16)
            else:
                p_interp[i] = 1.0 
                r_interp[i] = 0.0
                f1_interp[i] = 0.0
                
        mean_p += p_interp
        mean_r += r_interp
        mean_f1 += f1_interp
        
        plt.plot(thresholds, f1_interp, label=categories[c], alpha=0.5)
        
    if valid_classes > 0:
        mean_p /= valid_classes
        mean_r /= valid_classes
        mean_f1 /= valid_classes
        
    plt.plot(thresholds, mean_f1, label='All Classes', color='black', linewidth=3)
    plt.xlabel('Confidence', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title(f'{split_name} F1-Confidence Curve', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(out_dir, f'{split_name}_F1_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 8))
    plt.plot(thresholds, mean_p, label='All Classes', color='black', linewidth=3)
    plt.xlabel('Confidence', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'{split_name} Precision-Confidence Curve', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(out_dir, f'{split_name}_P_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 8))
    plt.plot(thresholds, mean_r, label='All Classes', color='black', linewidth=3)
    plt.xlabel('Confidence', fontsize=12)
    plt.ylabel('Recall', fontsize=12)
    plt.title(f'{split_name} Recall-Confidence Curve', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(out_dir, f'{split_name}_R_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    class_names = [categories.get(i, f'Class {i}') for i in range(len(categories))] + ['Background']
    
    # Plot Confusion Matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(conf_mat, interpolation='nearest', cmap='Blues')
    plt.title(f'{split_name} Confusion Matrix', fontsize=14, fontweight='bold')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = conf_mat.max() / 2.
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            plt.text(j, i, format(int(conf_mat[i, j]), 'd'),
                     ha="center", va="center",
                     color="white" if conf_mat[i, j] > thresh else "black")
                     
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{split_name}_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Normalized Confusion Matrix
    conf_mat_norm = conf_mat.astype('float') / (conf_mat.sum(axis=1)[:, np.newaxis] + 1e-16)
    plt.figure(figsize=(10, 8))
    plt.imshow(conf_mat_norm, interpolation='nearest', cmap='Blues')
    plt.title(f'{split_name} Normalized Confusion Matrix', fontsize=14, fontweight='bold')
    plt.colorbar()
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    
    thresh_norm = conf_mat_norm.max() / 2.
    for i in range(conf_mat_norm.shape[0]):
        for j in range(conf_mat_norm.shape[1]):
            plt.text(j, i, format(conf_mat_norm[i, j], '.2f'),
                     ha="center", va="center",
                     color="white" if conf_mat_norm[i, j] > thresh_norm else "black")
                     
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{split_name}_confusion_matrix_normalized.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved all plots for {split_name} in {out_dir}")

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    train_anns = os.path.join(project_root, "data", "coco", "annotations", "instances_train2017.json")
    train_res = os.path.join(project_root, "results", "train", "NanoDet", "results0.json")
    train_out = os.path.join(project_root, "viz", "nanodet_train_plots")
    
    val_anns = os.path.join(project_root, "data", "coco", "annotations", "instances_val2017.json")
    val_res = os.path.join(project_root, "results", "val", "nanodet", "results-1.json")
    val_out = os.path.join(project_root, "viz", "nanodet_val_plots")
    
    print("Evaluating Train Results...")
    train_gts, train_preds, cats = load_data(train_anns, train_res)
    class_preds, class_total_gts, conf_mat = evaluate(train_gts, train_preds, cats)
    plot_curves_and_cm(class_preds, class_total_gts, conf_mat, cats, "Train", train_out)
    
    print("Evaluating Validation Results...")
    val_gts, val_preds, cats = load_data(val_anns, val_res)
    class_preds, class_total_gts, conf_mat = evaluate(val_gts, val_preds, cats)
    plot_curves_and_cm(class_preds, class_total_gts, conf_mat, cats, "Validation", val_out)
    
    test_anns = os.path.join(project_root, "data", "coco", "annotations", "instances_test2017.json")
    test_res = os.path.join(project_root, "results", "test", "NanoDet", "model_best", "model_best", "results-1.json")
    test_out = os.path.join(project_root, "viz", "nanodet_test_plots")
    
    if os.path.exists(test_res):
        print("Evaluating Test Results...")
        test_gts, test_preds, cats = load_data(test_anns, test_res)
        class_preds, class_total_gts, conf_mat = evaluate(test_gts, test_preds, cats)
        plot_curves_and_cm(class_preds, class_total_gts, conf_mat, cats, "Test", test_out)
    else:
        print(f"Test results not found at {test_res}")

if __name__ == "__main__":
    main()
