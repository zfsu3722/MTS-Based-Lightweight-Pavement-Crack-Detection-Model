
import numpy as np
from data_io import imread


def cal_semantic_metrics(pred_list, gt_list, epoch, num_cls=2):
    final_accuracy_all = []


    global_accuracy_cur = []
    statistics = []

    for pred, gt in zip(pred_list, gt_list):
        #print("gt:", gt.shape)
        #print("pred:", pred.shape)
        gt = gt.cpu().squeeze(0).squeeze(0).numpy()
        gt_img = gt.astype('uint8')
        pred = pred.cpu().squeeze(0).squeeze(0).numpy()
        pred_img = pred.astype('uint8')
        # calculate each image
        global_accuracy_cur.append(cal_global_acc(pred_img, gt_img))
        statistics.append(get_statistics(pred_img, gt_img, num_cls))

    # get global accuracy with corresponding threshold: (TP+TN)/all_pixels
    global_acc = np.sum([v[0] for v in global_accuracy_cur])/np.sum([v[1] for v in global_accuracy_cur])

    # get tp, fp, fn
    counts = []
    for i in range(num_cls):
        tp = np.sum([v[i][0] for v in statistics])
        fp = np.sum([v[i][1] for v in statistics])
        fn = np.sum([v[i][2] for v in statistics])
        counts.append([tp, fp, fn])

    # calculate mean accuracy
    mean_acc = np.sum([v[0]/(v[0]+v[2]) for v in counts])/num_cls
    # calculate mean iou
    mean_iou_acc = np.sum([v[0]/(np.sum(v)) for v in counts])/num_cls
    final_accuracy_all.append([epoch, global_acc, mean_acc, mean_iou_acc])
    save_results_to_txt(final_accuracy_all, 'Segmentation metrics.txt')
    return final_accuracy_all


def save_results_to_txt(results, file_path):
    with open(file_path, 'a') as file:
        for result in results:
            file.write(f'Epoch {result[0]}: global_acc: {result[1]:.6f}, mean_acc: {result[2]:.6f}, mean_iou_acc: {result[3]:.6f}\n')


def cal_global_acc(pred, gt):
    """
    acc = (TP+TN)/all_pixels
    """
    #gt = gt.squeeze(0).squeeze(0)
    #print("gt.shape:", gt.shape)
    h,w = gt.shape
    return [np.sum(pred == gt), float(h*w)]


def get_statistics(pred, gt, num_cls=2):
    """
    return tp, fp, fn
    """
    h, w = gt.shape
    statistics = []
    for i in range(num_cls):
        tp = np.sum((pred == i) & (gt == i))
        fp = np.sum((pred == i) & (gt != i))
        fn = np.sum((pred != i) & (gt == i))
        statistics.append([tp, fp, fn])
    return statistics

