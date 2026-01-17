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

IS_SEGMENT_MOD = False
EVAL_HIST_CLASS = mle_spt.EVAL_HIST_CLASS_3
EVAL_FILE_PATH =  "E:\\Histogram_classification\\use"
EVAL_THRESHOLD_NUM_LIST = np.array([ 3, 5, 7, 9, 11, 13, 15, 17, 19, 21])
EVAL_RESULT_DUMP_FILE_PATH = "E:\\Histogram_classification\\seg_eval_dump\\eval_dump_test_eqb_test"
IMG_PLANE_IDX = cr5_spt.IMG_MAP_GRAY_IDX
IS_SEGMENT_INT = False
EVAL_TRIAL_NUM = 2
EVAL_SEG_METHOD = mle_spt.EVAL_SEG_INTEGER#mle_spt.EVAL_SEG_OPT#mle_spt.EVAL_SEG_INTEGER
EVAL_SEG_METHOD_NAME = mle_spt.EVAL_SEG_METHOD_INTEGER#mle_spt.EVAL_SEG_METHOD_EQB#mle_spt.EVAL_SEG_METHOD_INTEGER
EVAL_OPT_OBJ_METHOD = mle_spt.OPT_OBJ_FUN_PARAM

additional_params = dict()
additional_params[mle_spt.IS_SEGMENT_MOD_PARAM] = IS_SEGMENT_MOD
additional_params[mle_spt.IS_IMG_INT_PARAM] = IS_SEGMENT_INT
additional_params[mle_spt.EVAL_SEG_METHOD_PARAM] = EVAL_SEG_METHOD
additional_params[mle_spt.EVAL_SEG_METHOD_NAME_PARAM] = EVAL_SEG_METHOD_NAME
additional_params[mle_spt.EVAL_HIST_CLASS_PARAM] = EVAL_HIST_CLASS
additional_params[mle_spt.EVAL_TRIAL_NUM_PARAM] = EVAL_TRIAL_NUM
additional_params[mle_spt.EVAL_RESULT_DUMP_FILE_PATH_PARAM] = EVAL_RESULT_DUMP_FILE_PATH
additional_params[mle_spt.OPT_OBJ_FUN_PARAM] = EVAL_OPT_OBJ_METHOD
additional_params[mle_spt.EVAL_METRIC_LIST_PARAM] = [[mle_spt.FSIM_KEY, mle_spt.EVAL_FSIM_FUN], [mle_spt.SSIM_KEY, mle_spt.EVAL_SSIM_FUN], [mle_spt.PSNR_KEY, mle_spt.EVAL_PSNR_FUN], [mle_spt.RMSE_KEY, mle_spt.EVAL_RMSE_FUN]]
img_plane_list, img_plane_name_list = mle_spt.load_eval_img_plane(EVAL_FILE_PATH, IMG_PLANE_IDX, is_img_int=IS_SEGMENT_INT)
eval_result = mle_spt.eval_img_plane_list_multi_threshold_num(img_plane_list, img_plane_name_list, EVAL_THRESHOLD_NUM_LIST, additional_params)
print(eval_result)
