"""
Calculate sensitivity and specificity metrics:
 - Precision
 - Recall
 - F-score
"""

import numpy as np
from data_io import imread
import torch
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def cal_prf_metrics(pred_list, gt_list, epoch):
    final_accuracy_all = []
    statistics = []
    for pred, gt in zip(pred_list, gt_list):
        if pred is None or gt is None:
            continue  # Skip this pair if either pred or gt is None
        #pred_img = pred
        #gt_img = gt
        #gt = gt.squeeze(0).squeeze(0).numpy()
        gt = gt.cpu().numpy()
        gt_img = gt.astype('uint8')
        pred = pred.cpu().numpy()
        pred_img = pred.astype('uint8')
        # calculate each image

        statistics.append(get_statistics(pred_img, gt_img))
            # Check if there are valid statistics
        if not statistics:
            continue  # Skip if there are no valid statistics

    # get tp, fp, fn
    tp = np.sum([v[0] for v in statistics])
    fp = np.sum([v[1] for v in statistics])
    fn = np.sum([v[2] for v in statistics])

    # calculate precision
    p_acc = 1.0 if tp == 0 and fp == 0 else tp/(tp+fp)
    # calculate recall
    r_acc = tp/(tp+fn)
    # calculate f-score
    final_accuracy_all.append([epoch, p_acc, r_acc, 2*p_acc*r_acc/(p_acc+r_acc)])
    print(f'Epoch {epoch}: Precision: {p_acc:.6f}, Recall: {r_acc:.6f}, f1:{2*p_acc*r_acc/(p_acc+r_acc):.6f}')
    save_results_to_txt(final_accuracy_all, 'results.txt')
    return final_accuracy_all
def save_results_to_txt(results, file_path):
    with open(file_path, 'a') as file:
        for result in results:
            file.write(f'Epoch {result[0]}: Precision: {result[1]:.6f}, Recall: {result[2]:.6f}, f1: {result[3]:.6f}\n')

            # Assuming final_accuracy_all is the result you want to save
#save_results_to_txt(final_accuracy_all, 'results.txt')
def get_statistics(pred, gt):
    """
    return tp, fp, fn
    """
    #if pred.shape != gt.shape:
        #pred = pred.transpose(1, 0)
    #print("pred.shape:", pred.shape)
    #print("gt.shape:", gt.shape)
    tp = np.sum((pred == 1.0) & (gt == 1.0))
    fp = np.sum((pred == 1.0) & (gt == 0.0))
    fn = np.sum((pred == 0.0) & (gt == 1.0))
    return [tp, fp, fn]


# def cal_prf_metrics(pred_list, gt_list, epoch):
#     final_accuracy_all = []
#     statistics = []
#
#     for pred, gt in zip(pred_list, gt_list):
#         if pred is None or gt is None:
#             continue  # Skip this pair if either pred or gt is None
#
#         gt = gt.squeeze(0).squeeze(0).numpy()
#         gt = gt.astype('uint8')
#         pred = pred.squeeze(0).squeeze(0).cpu().numpy()
#         pred = pred.astype('uint8')
#         if pred.shape != gt.shape:
#             pred = pred.transpose(1, 0)
#         tp = torch.sum((pred == 1) & (gt == 1)).item()
#         fp = torch.sum((pred == 1) & (gt == 0)).item()
#         fn = torch.sum((pred == 0) & (gt == 1)).item()
#
#         statistics.append([tp, fp, fn])
#
#     tp = sum(v[0] for v in statistics)
#     fp = sum(v[1] for v in statistics)
#     fn = sum(v[2] for v in statistics)
#
#     p_acc = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp)
#     r_acc = tp / (tp + fn)
#     f1 = 2 * p_acc * r_acc / (p_acc + r_acc)
#
#     final_accuracy_all.append([epoch, p_acc, r_acc, f1])
#     print(f'Epoch {epoch}: Precision: {p_acc:.6f}, Recall: {r_acc:.6f}, F1: {f1:.6f}')
#     return final_accuracy_all
