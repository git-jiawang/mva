import os
import logging
import time
from pathlib import Path
import torch
from scipy.optimize import linear_sum_assignment
import numpy as np
import matplotlib.pyplot as plt

def create_logger(cfg, phase='train'):
    this_dir = Path(os.path.dirname(__file__))
    final_output_dir = (this_dir / '..' / cfg.EXP_DIR / cfg.DATASET).resolve()
    final_logs_dir = final_output_dir / 'logs'
    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)
    print('=> creating {}'.format(final_logs_dir))
    final_logs_dir.mkdir(parents=True, exist_ok=True)
    
    time_str = time.strftime('%Y-%m-%d_%H-%M')
    log_file = '{}_{}.log'.format(time_str, phase)
    final_log_file = final_logs_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger, str(final_output_dir)
        
def plot_loss_curves(logger, epoch_history, train_loss_history, val_loss_history, best_loss, best_loss_epoch, output_dir):
    try:
        plt.figure(figsize=(12, 8))
        plt.plot(epoch_history, train_loss_history, 'b-', linewidth=2, label='Training Loss')

        val_epochs = []
        val_losses = []
        for i, epoch in enumerate(epoch_history):
            if i < len(val_loss_history) and val_loss_history[i] is not None:
                val_epochs.append(epoch)
                val_losses.append(val_loss_history[i])
        
        if val_epochs:
            plt.plot(val_epochs, val_losses, 'r-', linewidth=2, label='Validation Loss')
            plt.plot(val_epochs, val_losses, 'ro', markersize=4)
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training and Validation Loss Curves', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        
        # 标记最佳loss
        if best_loss_epoch != -1:
            plt.plot(best_loss_epoch, best_loss, 'g*', markersize=12, label=f'Best Loss: {best_loss:.4f}')
            plt.annotate(f'Best: {best_loss:.4f}', 
                        xy=(best_loss_epoch, best_loss), 
                        xytext=(best_loss_epoch+5, best_loss+0.1),
                        fontsize=10,
                        arrowprops=dict(arrowstyle='->', color='green'))
        
        loss_curve_path = os.path.join(output_dir, 'loss_curve.png')
        plt.savefig(loss_curve_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved loss curve to {loss_curve_path}")
    except Exception as e:
        logger.warning(f"Failed to plot loss curve: {e}")
        
def plot_AP_curve(labels, scores, ap, output_dir):
    from sklearn.metrics import precision_recall_curve
    precision_vals, recall_vals, _ = precision_recall_curve(labels, scores)
    plt.figure(figsize=(10, 8))
    plt.plot(recall_vals, precision_vals, 'b-', linewidth=2, label=f'PR Curve (AP = {ap:.4f})')
    plt.fill_between(recall_vals, precision_vals, alpha=0.2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve (AP: {ap:.4f})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'pr_curve_test.png'), dpi=150)
    plt.close()

def get_dis_matrix(pred_pts, gt_pts, mode='l2'):
    pts1 = pred_pts.repeat_interleave(len(gt_pts), 0)
    pts2 = gt_pts.repeat(len(pred_pts), 1)
    if mode == 'l2':
        dis_matrix = torch.sqrt((pts1 - pts2).pow(2).sum(-1)).reshape(len(pred_pts), len(gt_pts))
    elif mode == 'l1':
        dis_matrix = (pts1 - pts2).abs().sum(-1).reshape(len(pred_pts), len(gt_pts))
    return dis_matrix

def matrix_normalization(matrix, min_max=False):
    min_v = matrix.min() if min_max else 0
    max_v = matrix.max()
    return (matrix - min_v) / (max_v - min_v + 1e-6)

def cross_view_matching_evaluation(pred_pts, gt_pts, labels1, labels2, reid_fea1=None, reid_fea2=None, thresh=0.4, alpha=0.1):
    if len(pred_pts) == 0:
        return {}
    dis_matrix = get_dis_matrix(pred_pts, gt_pts)
    norm_dis_matrix = matrix_normalization(dis_matrix)
    if reid_fea1 is not None:
        reid_dis_matrix = get_dis_matrix(reid_fea1, reid_fea2)
        reid_norm_dis_matrix = matrix_normalization(reid_dis_matrix)
        norm_dis_matrix = norm_dis_matrix * (1 - alpha) + reid_norm_dis_matrix * alpha
    
    norm_dis_matrix = norm_dis_matrix.cpu().detach().numpy()
    matches_x, matches_y = linear_sum_assignment(norm_dis_matrix) # 匈牙利匹配算法
    matches_x = list(matches_x)
    matches_y = list(matches_y)

    score_matrix = 1 - norm_dis_matrix
    # 按阈值过滤置信度低的行人匹配对
    for k in range(len(matches_x) - 1, -1, -1):
        if len(matches_x) == 1:
            break
        if score_matrix[matches_x[k], matches_y[k]] < thresh:
            del matches_x[k]
            del matches_y[k]
                      
    P_tot = len(matches_x)
    correct = sum(1 for k in range(P_tot) if labels1[matches_x[k]] != -1 and labels1[matches_x[k]] == labels2[matches_y[k]])
    # unmatched_labels1 = {labels1[i] for i in range(len(labels1)) if labels1[i] != -1 and i not in matches_x}
    # unmatched_labels2 = {labels2[i] for i in range(len(labels2)) if labels2[i] != -1 and i not in matches_y}
    # TN = len(unmatched_labels1 - set(labels2)) + len(unmatched_labels2 - set(labels1))
    TN = 0
    for i in range(len(labels1)):
        if labels1[i] == -1 or i in matches_x:
            continue
        if labels1[i] not in labels2:
            TN += 1
    for i in range(len(labels2)):
        if labels2[i] == -1 or i in matches_y:
            continue
        if labels2[i] not in labels1:
            TN += 1
    
    labels1_set = {x for x in labels1.tolist() if x != -1}
    labels2_set = {x for x in labels2.tolist() if x != -1}
    R_tot = len(labels1_set.intersection(labels2_set))
    A_tot = len(labels1_set.union(labels2_set))   
    
    label_matrix = np.zeros((len(labels1), len(labels2)), dtype=bool)
    label_mask = np.ones((len(labels1), len(labels2)), dtype=bool)
    for i in range(len(labels1)):
        for j in range(len(labels2)):
            if labels1[i] == labels2[j]:
                label_matrix[i, j] = True
            if labels1[i] == -1 or labels2[j] == -1:
                label_mask[i, j] = False
                
    result = {}
    result['scores'] = score_matrix.flatten()[label_mask.flatten()]
    result['labels'] = label_matrix.flatten()[label_mask.flatten()]
    result['correct'] = correct
    result['precision_total'] = P_tot
    result['recall_total'] = R_tot
    result['IPAA_total'] = A_tot
    result['IPAA_correct'] = correct + TN
    
    # if result['IPAA_correct'] > result['IPAA_total']:
    #     print("\n" + "=" * 80)
    #     print("WARNING: IPAA_correct > IPAA_total")
    #     print("unmatched_labels1:", unmatched_labels1)
    #     print("unmatched_labels2:", unmatched_labels2)
    #     print("set(labels1):", set(labels1))
    #     print("set(labels2):", set(labels2))
    #     print("unmatched_labels1 - set(labels2):", unmatched_labels1 - set(labels2))
    #     print("unmatched_labels2 - set(labels1):", unmatched_labels2 - set(labels1))
    #     print("=" * 80)
    #     print("labels1:", labels1.tolist())
    #     print("labels2:", labels2.tolist())
    #     print("matches_x:", matches_x)
    #     print("matches_y:", matches_y)
    #     print("correct:", correct)
    #     print("TN:", TN)
    #     print("IPAA_correct:", correct + TN)
    #     print("IPAA_total:", A_tot)
    #     print("labels1_set:", labels1_set)
    #     print("labels2_set:", labels2_set)
    #     print("=" * 80)
    
    return result

def FPR_95(labels, scores):
    recall_point = 0.95
    # Sort label-score tuples by the score in descending order.
    indices = np.argsort(scores)[::-1]
    sorted_labels = labels[indices]
    sorted_scores = scores[indices]
    n_match = sum(sorted_labels)
    n_thresh = recall_point * n_match
    thresh_index = np.argmax(np.cumsum(sorted_labels) >= n_thresh)
    FP = np.sum(sorted_labels[:thresh_index] == 0)
    TN = np.sum(sorted_labels[thresh_index:] == 0)
    return float(FP) / float(FP + TN)