import numpy as np
import self_imq_implement_support as imq_spt
import crack500_support as cr5_spt
import crack_self_proc as cr_sf_proc
import matplotlib.pyplot as plt
import time as tm
import img_quality_metrics as img_qm
import cv2

import scipy.stats as stats
from sklearn.metrics import r2_score
from scipy import sparse
from scipy.stats import kurtosis
import self_imq_implement_support as imq_spt
import multi_level_thresholding_segmentation_evaluation_support as mle_spt
plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 25,
    'axes.labelsize': 25,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'legend.fontsize': 20
})

EVAL_RESULT_DUMP_FILE_PATH = "E:\\Histogram_classification\\single_result\\seg_eval_dump"
EVAL_STATISTIC_RESULT_FILE_PATH =  "E:\\Histogram_classification\\single_result\\eval_statistics_result\\eval_statistics_result_1"
FIGURE_FILE_OUTPUT_PATH = "E:\\Histogram_classification\\AAA\\time_complex_peak.png"
FIGURE_FILE_OUTPUT_PATH_1 = "E:\\Histogram_classification\\AAA\\fsim_single_peak.png"
IS_FIGURE_SHOW = False

additional_params = dict()
additional_params[mle_spt.EVAL_METRIC_KEY_LIST_PARAM] = [mle_spt.FSIM_KEY, mle_spt.SSIM_KEY, mle_spt.SEG_TIME_KEY, mle_spt.PSNR_KEY, mle_spt.RMSE_KEY]
eval_metric_statistics_result = mle_spt.analyse_eval_metrics_mean_curve_statistics(EVAL_RESULT_DUMP_FILE_PATH, additional_params, EVAL_STATISTIC_RESULT_FILE_PATH)
print(eval_metric_statistics_result)
metric_curve_list = mle_spt.get_class_type_metric_curves(EVAL_STATISTIC_RESULT_FILE_PATH, mle_spt.EVAL_HIST_CLASS_1, mle_spt.FSIM_KEY)
print(metric_curve_list)



#Segmentation Time
#SEG_TIME
marker_list = ['x', 'h', '+', 'o']
color_list = ['green', 'black', 'red', 'blue']
x_label_name = "Threshold Number"
y_label_name = "FSIM"
title_name = "FSIM on unimodal Images"
line_width = 1.5
marker_size = 7
mle_spt.display_seg_performance_curve_mean_only(metric_curve_list, marker_list, color_list, x_label_name, y_label_name, file_save_path=FIGURE_FILE_OUTPUT_PATH_1, is_figure_show=IS_FIGURE_SHOW, title_name=title_name, line_width=line_width, marke_rsize=marker_size)
