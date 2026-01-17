import numpy as np
import pickle as pik
import os
import cv2
import time as tm
import gc
import gauss_fun_generator as gs_gen
import matplotlib.pyplot as plt
import imutils
import math as mat
import torch
from itertools import combinations
import pydicom
import img_quality_metrics as img_qm
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from scipy.stats import pearsonr
from scipy.signal import convolve2d
import self_imq_implement_support as imq_spt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
from scipy.cluster.vq import kmeans, whiten, vq
import phasepack as pc

R_FACTOR = 0.2999
G_FACTOR = 0.587
B_FACTOR = 0.114
INT_GRAY_LEVEL_BAR = 255
IMG_SINGLE_STEP_VALUE_ABS = 1/INT_GRAY_LEVEL_BAR
RGB_IMG_C = 3
TRAIN_IMG_LIST_INDEX = 0
LABEL_IMG_LIST_INDEX = 1
IMAGE_TRANS_PRINT_COUNT = 2 #5
IMAGE_DUMP_EXT = ".crset"
IMAGE_PACK_FILE_NUM = 8 #300
IMAGE_ROUND_PRECISION = 8
CRACK_BASE_INCREMENT_FACTOR = 1.3
INTENSITY_RES = 0.2  # 0.16 1.1 1.4
INTENSITY_CONSTRAINT = 0.8 # 0.8
SCAN_DOWN_WARD = 1
SCAN_UP_WARD = -1
SCALE_NORMAL = 1
SCALE_ZERO = 0
SCALE_ONE = 100
DIRECTION_H = 0
DIRECTION_V = 1
SCAN_WIDTH_DISTRIBUTION_FACTOR = 6
SCAN_STEP_DISTRIBUTION_FACTOR = 20#20
G_FILTER_NORM_CONCAT_COL_NUM = 4
CRACK_SHAPE_CONCAVE = True
CRACK_SHAPE_BULGE = False
MASK_BLACK = 0
MASK_WHITE = 1
CRACK_CORE_DARK = 0
CRACK_CORE_LIGHT = 1
SEG_DIRECTION_ROW = 0
SEG_DIRECTION_COL = 1
SEG_SCAN_POS = 1
SEG_SCAN_NEG = -1
SEG_RENDER_INC = 1
SEG_RENDER_DEC = -1
SEG_IMG_COPY = True
SEG_IMG_NO_COPY = False
SEG_SHAPE_BULGE = 0
SEG_SHAPE_CONCAVE = 1
SEG_SHAPE_LINEAR = 2
SEG_RANDOM = True
SEG_REGULAR = False
SEG_INTENSITY_LOWER_BOUND = 0.01
SEG_ANGLE_LOWER_BOUND = 0.005
SEG_CURVE_BULGE_DIVERSION = 0
SEG_CURVE_CONCAVE_DIVERSION = 1
SEG_CURVE_LINEAR = 2
SEG_CURVE_STATIC = 3
SEG_CURVE_FLAT = 4
SEG_CURVE_STABLE = 5
SEG_CURVE_DOWNWARD_TRIANGLE_LINEAR = 6
SEG_CURVE_DOWNWARD_TRIANGLE_LINEAR_GRAD = 0.005
SEG_CURVE_INTENSITY_KEPT_STATIC = 7
SEG_CURVE_VAL_STATIC_DEFAULT = 0
SEG_CURVE_VAL_STATIC = SEG_CURVE_VAL_STATIC_DEFAULT
SEG_HALF_PI_BOUND_RATIO = 0.9
SEG_CURVE_SHRINK_RATIO_DEFAULT = 1#0.002
SEG_CURVE_INITIAL_ANGLE_BOUND_RATIO = 0.9
SEG_CURVE_INTENSITY_UPPER_BOUND = 1
CURVE_DISPLAY = False
SEG_CURVE_CLIMB_SPEED_RATIO = 1
SEG_RET_CURVE_LIST = False
SEG_CURVE_DISPLAY_COL_NUM = 10
SEG_MONOTONIC_UPPER_BOUND_RATIO = 0.95
SEG_MONOTONIC_LOWER_BOUND_RATIO = 0.005
SEG_BULGE_GRAD_MONOTONIC_INITIAL_BOUND_RATIO = 0.4#0.3
SEG_CONCAVE_GRAD_MONOTONIC_INITIAL_BOUND_RATIO = 0.05
SEG_CONCAVE_MONOTONIC_GRAD_UPPER_BOUND_RATIO = 0.2#0.7
SEG_CONCAVE_MONOTONIC_GRAD_LOWER_BOUND_RATIO = 0.4#0.6#0.005
SEG_BULGE_MONOTONIC_GRAD_UPPER_BOUND_RATIO = 0.6#0.7
SEG_BULGE_MONOTONIC_GRAD_LOWER_BOUND_RATIO = 0.15#0.005
SEG_FLAT_INITIAL_SECOND_BOUND_RATIO_LOWER = 0.3
SEG_FLAT_INITIAL_SECOND_BOUND_RATIO_UPPER = 0.1
SEG_FLAT_UPPER_BOUND_RATIO = 0.9
SEG_RECT_TRANSPOSE = True
SEG_RECT_NO_TRANSPOSE = False
SEG_CIRCLE_DELTA_THETA = 0.1
DISTANCE_SEG_NUM =8#4#9#17#20#100#60#50#15#30#100
DISTANCE_SEG_MAX_INTENSITY = 1#1
DISTANCE_SEG_MIN_INTENSITY = 0.01#0.01
INTENSITY_STEP_VAL = 0.06
FRAGMENT_FILTER_ALGORITHM_POISSON = 0
FRAGMENT_FILTER_ALGORITHM_STEP = 1
FRAGMENT_FILTER_ALGORITHM_NORMALIZED_STD = 2
FRAGMENT_FILTER_ALGORITHM_DISTANCE_METRIC_AVG_STD = 3
FRAGMENT_FILTER_ALGORITHM_BINARY_ONE_ZERO_RATIO_METRIC = 4
FRAGMENT_FILTER_ALGORITHM = FRAGMENT_FILTER_ALGORITHM_POISSON
MASK_IMG_VAL_ZERO = 0
MASK_IMG_VAL_ONE = 1
MASK_IMG_VAL_RANDOM = 2
SEG_MARK_DARK = -1
SEG_EDGE_INTENSITY_ORIGINAL = -1
SEG_EDGE_STATE_INITIAL = 0
SEG_EDGE_STATE_DOWN = -1
SEG_EDGE_STATE_UP = 1
SEG_EDGE_TEXTURE = 0
SEG_EDGE_THIN_LINE = 1
SEG_EDGE_THIN_LINE_3D = 2
SEG_EDGE_CHECK_STRATEGY = SEG_EDGE_TEXTURE
SEG_EDGE_BODY_STATE_FLAT = 0
SEG_EDGE_BODY_STATE_UP = 1
SEG_EDGE_BODY_STATE_DOWN = 2
SEG_EDGE_RELATIVE_RATIO_THRESHOLD = 0.45
SEG_EDGE_MIN_DETECTABLE_INTENSITY = 0.07
SEG_EDGE_UPDATE_INDEX_NO_UPDATE = -1
SEG_EDGE_SCAN_TYPE_STATIC_CONTRAST = 0
SEG_EDGE_SCAN_TYPE_DYNAMIC_CONTRAST = 1
SEG_EDGE_SCAN_TYPE = SEG_EDGE_SCAN_TYPE_STATIC_CONTRAST
SEG_EDGE_RELATIVE_RATIO_THRESHOLD_ROW = 0.5
SEG_EDGE_RELATIVE_RATIO_THRESHOLD_COL = 0.5
TRADITIONAL_EDGE_DETECTOR_SOBEL = 0
TRADITIONAL_EDGE_DETECTOR_CANNY = 1
TRADITIONAL_EDGE_DETECTOR_CANNY_THRESHOLD_1 = 10
TRADITIONAL_EDGE_DETECTOR_CANNY_THRESHOLD_2 = 10
TRADITIONAL_EDGE_DETECTOR_INPUT_GAUSS_BLUR = 0
TRADITIONAL_EDGE_DETECTOR_INPUT_POISSON_DISTANCE = 1
TRADITIONAL_EDGE_DETECTOR_INPUT_DIRECT_REAL = 2
TRIPLE_PIXEL_SECOND_ORDER_GRAD_Z_ANGLE_HALF_PI = 0.15
TRIPLE_PIXEL_SECOND_ORDER_GRAD_Z_ANGLE_PI = 0.3
IMG_EXTRACT_POS = 0
IMG_EXTRACT_NEG = 1
COORDINATE_BASIC_STEP = 0.001
COORDINATE_VIBRATION_SIGNIFICANCE_THRESHOLD = 2.3#1.5
COORDINATE_INITIAL_TOTAL_ABS_DELTA = 0
ANGLE_GRAD_UPWARD = 1
ANGLE_GRAD_DOWNWARD = -1
ANGLE_GRAD_STEADY = 0
SEG_IMG_PLANE_EXTRACT_NONZERO_RATIO_THRESHOLD = 0.265#0.165
IMG_PLANE_BINARY = 0
IMG_PLANE_REAL = 1
IMG_SEG_DISTANCE_AVG_DISTANCE_IDX = 0
IMG_SEG_DISTANCE_SORTED_ARRAY_DIX = 1
IMG_SEG_DISTANCE_SUB_ARRAY_IDX = 2
IMG_SEG_DISTANCE_NONZERO_ARRAY_LEN_IDX =3
IMG_SEG_DISTANCE_MAX_DISTANCE_IDX = 4
IMG_SEG_DISTANCE_STD_DISTANCE_IDX = 5
CMP_OPERATOR_LARGE = 0
CMP_OPERATOR_SMALL = 1
CMP_EQU_INCLUDE = True
CMP_EQU_EXCLUDE = False
CMP_VAL_MASK_ONLY = True
CMP_VAL_ORIGINAL = False
CMP_VAL_ZERO_MASK = True
CMP_VAL_ZERO_NO_MASK = False
EDGE_EXTRACTION_ANGLE_DETECTION = 0
EDGE_EXTRACTION_MASK_FILTER = 1
PLANE_KEPT_ANY = 0
IMG_RANGE_INT_VAL_IDX = 0
IMG_RANGE_COUNT_IDX = 1
IMG_RANGE_STAIR_RANGE_VAL_IDX = 0
IMG_RANGE_STAIR_STAIR_VAL_IDX = 1
IMG_RANGE_STAIR_STAIR_COUNT_RATIO_IDX = 2
IMG_RANGE_STAIR_RANGE_PREV_REF_IDX = 3
IMG_RANGE_STAIR_RANGE_NEXT_REF_IDX = 4
IMG_RANGE_STAIR_STATUS_IDX = 5
IMG_RANGE_STAIR_STATUS_VALID = 0
IMG_RANGE_STAIR_STATUS_INVALID = -1
AVG_DIST_ADD_MES_AVG_DIST_IDX = 0
AVG_DIST_ADD_MES_STD_IDX = 1
AVG_DIST_ADD_MES_SQRT_ARRAY_LEN = 2
AVG_DIST_IMG_DIVISION_INFO = 3
AVG_DIST_IMG_NONE_ZERO_DISTANCE = 4
AVG_DIST_IMG_MAP_EXTRACT_TIME = 5
AVG_DIST_IMG_MAP_DISTANCE_LIST = 6
DISTANCE_MIN_RESOLUTION_CONST = 0.004
GRAY_LEVEL_COMPARE_STANDARD = 1
IMG_MAP_RED_IDX = 0
IMG_MAP_GREEN_IDX = 1
IMG_MAP_BLUE_IDX = 2
IMG_MAP_GRAY_IDX = -1
MAX_P_SSIM = 65
MAX_P_PSNR = 5
MAX_P_RMSE = 1
DIFF_SCENARIO_MAX = 0
DIFF_SCENARIO_MIDDLE = 1
DIFF_SCENARIO_MINI = 2
NON_SPARSE_SEG_RIGHT_IDX = 1
IMG_GRAY_SOURCE_TARGET_RECT_X_TL_IDX = 0
IMG_GRAY_SOURCE_TARGET_RECT_Y_TL_IDX = 1
IMG_GRAY_SOURCE_TARGET_RECT_X_BR_IDX = 2
IMG_GRAY_SOURCE_TARGET_RECT_Y_BR_IDX = 3
RECONSTRUCTION_PLANE_MASK_EMPTY = {}
DICOM_NORM_FACTOR = 1000
MONOCULAR_3D_SHAPE_BULGE = 1
MONOCULAR_3D_SHAPE_FLAT = 0
MONOCULAR_3D_SHAPE_CONCAVE = -1
MONOCULAR_3D_SHAPE_CONCAVE_DIALECT = -0
MONOCULAR_3D_SCAN_STATIC_POINT_COUNT = 2
MONOCULAR_3D_CODE_SINGLE_ELEMENT_U = 1
MONOCULAR_3D_CODE_SINGLE_ELEMENT_F = 0
MONOCULAR_3D_CODE_SINGLE_ELEMENT_D = -1
MONOCULAR_3D_CODE_SINGLE_ELEMENT_SE = -2
MONOCULAR_3D_CODE_WORD_DELTA_DIFF_DIRECTION_F = 0
MONOCULAR_3D_CODE_WORD_DELTA_DIFF_DIRECTION_U = 1
MONOCULAR_3D_CODE_WORD_DELTA_DIFF_DIRECTION_D = -1
MONOCULAR_3D_CODE_WORD_DELTA_DIFF_DIRECTION_NONE = -2
#MONOCULAR_3D_CODE_SINGLE_ELEMENT_U, element_len, element_val_diff_list
MONOCULAR_3D_CODE_WORD_TUPLE_CODE_ELEMENT_IDX = 0
MONOCULAR_3D_CODE_WORD_TUPLE_ELEMENT_LEN_IDX = 1
MONOCULAR_3D_CODE_WORD_TUPLE_ELEMENT_VAL_DELTA_LIST_IDX = 2
MONOCULAR_3D_CODE_WORD_DELTA_DIFF_DIRECTION_IDX = 3
MONOCULAR_3D_CODE_WORD_PARAM_WORD_TUPLE_IDX = 0
MONOCULAR_3D_CODE_WORD_PARAM_WORD_SCAN_INDEX_IDX = 1
MONOCULAR_3D_CODE_WORD_PARAM_NEXT_CODE_ELEMENT_IDX = 2
MONOCULAR_3D_CODE_SCAN_STATE_FLAT_LEFT = 0
MONOCULAR_3D_CODE_SCAN_STATE_OCULAR = 1
MONOCULAR_3D_CODE_SCAN_STATE_OCULAR_OPPOSITE = 2
MONOCULAR_3D_CODE_SCAN_STATE_FLAT_RIGHT = 3
MONOCULAR_3D_CODE_OCULAR_COHERENT = 0
SINUNO_FUN_PARAM_F_IDX = 0
SINUNO_FUN_PARAM_M_IDX = 1
SINUNO_FUN_PARAM_THETA_IDX = 2
SINUNO_FUN_PARAM_L0_IDX = 3
SINUNO_FUN_PARAM_CYCLE_WIDTH_IDX = 4
SINUNO_FUN_PARAM_CYCLE_NUM_IDX = 5
SINUNO_FUN_PARAM_TRIANGLE_FUN_IDX = 6
SINUNO_TRIANGLE_FUN_LIST = [np.sin, np.cos]
SINUNO_TRIANGLE_FUN_SIN = 0
SINUNO_TRIANGLE_FUN_COS = 1
SQUARE_GRATING_L0_IDX = 0
SQUARE_GRATING_V_ABS_MAX_IDX = 1
SQUARE_GRATING_CYCLE_NUM = 2
SQUARE_GRATING_HALF_CYCLE_WIDTH = 3
IMG_GRAY_DIM_HORIZONTAL = 0
IMG_GRAY_DIM_VERTICAL = 1
CONT_THRESH_ILLUSION_HOR_LEFT = 0
CONT_THRESH_ILLUSION_HOR_RIGHT = 1
CONT_THRESH_ILLUSION_VER_TOP = 0
CONT_THRESH_ILLUSION_VER_BOTTOM = 1
CONT_THRESH_ILLUSION_HOR = 0
CONT_THRESH_ILLUSION_VER = 1
SQUARE_GRATING_CONST_VAL_IDX = 4
SQUARE_GRATING_CONST_AVG = 0
SQUARE_GRATING_CONST_LOWEST = 1
MOS_SIM_METRIC_FSIM_IDX = 0
MOS_SIM_METRIC_SSIM_IDX = 1
MOS_SIM_METRIC_HFSIM_IDX = 2
MOS_SIM_METRIC_HSMSIM_IDX = 3


def nan_proc_replace(val, replace_val=0):
    if np.isnan(val):
        val_res = replace_val
    else:
        val_res = val
    return val_res


def load_img_training_data(img_train_file_root, img_label_file_root, train_img_file_name_pref, label_img_file_name_pref, img_file_num):
    print("loading images...")
    img_train_list = load_img_data(img_train_file_root, train_img_file_name_pref, img_file_num)
    print("train set loaded")
    img_label_list = load_img_data(img_label_file_root, label_img_file_name_pref, img_file_num)
    print("label set loaded")
    return [img_train_list, img_label_list]


def load_img_data(img_file_root, img_file_name_pref, img_file_num):
    img_file_path_pref = path_join(img_file_root, img_file_name_pref)
    img_list = []
    i = 0
    while i < img_file_num:
        img_file_name = form_file_name_ext(img_file_path_pref, i)
        img_sub_list = extract_img_objects(img_file_name)
        list_append_copy(img_sub_list, img_list)
        i = i + 1
    return img_list


def img_gray_resize(img, resize_ratio,  interpolation=cv2.INTER_NEAREST):
    img_shape = img.shape
    new_row = np.int_(np.ceil(img_shape[1]*resize_ratio))
    new_col = np.int_(np.ceil(img_shape[0]*resize_ratio))
    img_resized = np.round(cv2.resize(img, (new_row, new_col), interpolation), 0)
    return img_resized


def img_gray_resize_dim_wise(img, resize_ratio_col,  resize_ratio_row, interpolation=cv2.INTER_NEAREST):
    img_shape = img.shape
    new_row = np.int_(np.ceil(img_shape[1]*resize_ratio_col))
    new_col = np.int_(np.ceil(img_shape[0]*resize_ratio_row))
    img_resized = np.round(cv2.resize(img, (new_row, new_col), interpolation), 0)
    return img_resized


def img_gray_resize_dim_wise_spec(img, resize_col,  resize_row, interpolation=cv2.INTER_NEAREST):
    img_shape = img.shape
    #new_row = np.int_(np.ceil(img_shape[1]*resize_ratio_col))
    #new_col = np.int_(np.ceil(img_shape[0]*resize_ratio_row))
    img_resized = np.round(cv2.resize(img, (resize_row, resize_col), interpolation), 0)
    return img_resized


def list_append_copy(src_list, target_list):
    src_list_len = len(src_list)
    i = 0
    while i < src_list_len:
        target_list.append(src_list[i])
        i = i + 1


def package_train_label_files_divide(train_file_dump_root, train_file_dump_name_pref, label_file_dump_root, label_file_dump_name_pref, src_img_root, label_img_root,
                                  file_ext_list=None):
    img_file_lists = get_gary_img_list_from_dataset_files_path(src_img_root, label_img_root, file_ext_list)
    img_train_file_list = img_file_lists[TRAIN_IMG_LIST_INDEX]
    img_label_file_list = img_file_lists[LABEL_IMG_LIST_INDEX]
    file_num = len(img_train_file_list)
    d_file_group_num = int(file_num / IMAGE_PACK_FILE_NUM)
    m_file_group_num = file_num % IMAGE_PACK_FILE_NUM
    start_idx = 0
    end_idx_step = IMAGE_PACK_FILE_NUM - 1
    file_dump_name_surf = 0
    while d_file_group_num > 0:
        end_idx = start_idx + end_idx_step
        package_train_label_files_divide_do(img_train_file_list, img_label_file_list, train_file_dump_root,
                                            train_file_dump_name_pref, label_file_dump_root, label_file_dump_name_pref,
                                            file_dump_name_surf, start_idx, end_idx)
        d_file_group_num = d_file_group_num - 1
        start_idx = end_idx + 1
        file_dump_name_surf = file_dump_name_surf + 1
    if m_file_group_num > 0:
        end_idx = start_idx + m_file_group_num - 1
        package_train_label_files_divide_do(img_train_file_list, img_label_file_list, train_file_dump_root,
                                            train_file_dump_name_pref, label_file_dump_root, label_file_dump_name_pref,
                                            file_dump_name_surf, start_idx, end_idx)


def package_train_label_files_divide_do(img_train_file_list, img_label_file_list, train_file_dump_root, train_file_dump_name_pref, label_file_dump_root, label_file_dump_name_pref, file_dump_name_surf, start_idx, end_idx):
    img_train_list = get_gray_img_by_path_list_range(img_train_file_list, start_idx, end_idx)
    dump_file_name = get_dump_file_name(train_file_dump_root, train_file_dump_name_pref, file_dump_name_surf)
    package_img_file(dump_file_name, img_train_list)
    del img_train_list
    gc.collect()
    img_label_list = get_gray_img_by_path_list_range(img_label_file_list, start_idx, end_idx)
    dump_file_name = get_dump_file_name(label_file_dump_root, label_file_dump_name_pref, file_dump_name_surf)
    package_img_file(dump_file_name, img_label_list)
    del img_label_list
    gc.collect()


def get_dump_file_name(dump_file_root, dump_file_name_pref, dump_file_name_suf):
    #file_name = dump_file_name_pref+str(dump_file_name_suf)+IMAGE_DUMP_EXT
    file_name = form_file_name_ext(dump_file_name_pref, dump_file_name_suf)
    dump_file_name = path_join(dump_file_root, file_name)
    return dump_file_name


def form_file_name_ext(name_pref, name_num_suf):
    name_ext = name_pref + str(name_num_suf) + IMAGE_DUMP_EXT
    return name_ext


def package_train_label_files_sep(train_file_dump_path, label_file_dump_path, src_img_root, label_img_root,
                                  file_ext_list=None):
    img_lists = get_gary_img_list_from_dataset_files(src_img_root, label_img_root, file_ext_list, True)
    img_list = img_lists[TRAIN_IMG_LIST_INDEX]
    img_lists[TRAIN_IMG_LIST_INDEX] = None
    package_img_file(train_file_dump_path, img_list)
    del img_list
    gc.collect()
    img_list = get_gray_img_by_path_list(img_lists[LABEL_IMG_LIST_INDEX])
    package_img_file(label_file_dump_path, img_list)


def package_train_label_files_src(train_file_dump_path, label_file_dump_path, src_img_root, label_img_root,
                                  file_ext_list=None):
    img_lists = get_gary_img_list_from_dataset_files(src_img_root, label_img_root, file_ext_list)
    package_img_file(train_file_dump_path, img_lists[TRAIN_IMG_LIST_INDEX])
    package_img_file(label_file_dump_path, img_lists[LABEL_IMG_LIST_INDEX])


def package_train_label_files(img_lists, train_file_dump_path, label_file_dump_path):
    package_img_file(train_file_dump_path, img_lists[TRAIN_IMG_LIST_INDEX])
    package_img_file(label_file_dump_path, img_lists[LABEL_IMG_LIST_INDEX])


def get_gary_img_list_from_dataset_files_path(src_img_root, label_img_root, file_ext_list=None):
    src_img_list = []
    label_img_list = []
    counter = 0
    for root, dirs, files in os.walk(src_img_root):
        for name in files:
            img_path = path_join(root, name)
            src_img_list.append(img_path)
            if file_ext_list is None:
                label_file_name = name
            else:
                label_file_name = replace_file_extension(name, file_ext_list)
            img_path = path_join(label_img_root, label_file_name)
            label_img_list.append(img_path)
            counter = counter + 1
    return [src_img_list, label_img_list]


def get_gray_img_by_path_list_range(path_list, start_idx, end_idx):
    img_num = (end_idx-start_idx)+1
    label_img_list = [None]*img_num
    i = start_idx
    item_idx = 0
    while item_idx < img_num:
        gray_img = get_gray_img_from_file(path_list[i])
        label_img_list[item_idx] = gray_img
        i = i + 1
        item_idx = item_idx + 1
        if item_idx % IMAGE_TRANS_PRINT_COUNT == 0:
            print("Images (Single) ", i, " Transformed")
    return label_img_list


def get_gary_img_list_from_dataset_files(src_img_root, label_img_root, file_ext_list=None, label_img_path_only=False):
    src_img_list = []
    label_img_list = []
    counter = 0
    for root, dirs, files in os.walk(src_img_root):
        for name in files:
            gray_img = get_gray_img_from_file(path_join(root, name))
            src_img_list.append(gray_img)
            #src_img_list[counter]=gray_img
            if file_ext_list is None:
                label_file_name = name
            else:
                label_file_name = replace_file_extension(name, file_ext_list)
            if not label_img_path_only:
                gray_img = get_gray_img_from_file(path_join(label_img_root, label_file_name))
            else:
                gray_img = path_join(label_img_root, label_file_name)
            label_img_list.append(gray_img)
            counter = counter + 1
            if counter % IMAGE_TRANS_PRINT_COUNT == 0:
                print("Images ", counter, " Transformed")
            if counter >= 300:
                break;
    return [src_img_list, label_img_list]


def get_gray_img_by_path_list(path_list):
    img_num = len(path_list)
    label_img_list = []
    i = 0
    while i < img_num:
        gray_img = get_gray_img_from_file(path_list[i])
        label_img_list.append(gray_img)
        i = i + 1
        if i % IMAGE_TRANS_PRINT_COUNT == 0:
            print("Images (Single) ", i, " Transformed")
    return label_img_list


def get_gray_img_from_file(file_path, img_map_idx=IMG_MAP_GRAY_IDX):
    # raw_img = cv2.imread(file_path)
    # raw_img = img_gbr_to_rgb(raw_img)
    raw_img = get_rgb_img_from_file(file_path)
    if img_map_idx == IMG_MAP_GRAY_IDX:
        gray_img = img_rgb_to_gray_array_cal(raw_img)
    else:
        gray_img = raw_img[:, :, img_map_idx]/INT_GRAY_LEVEL_BAR
    return gray_img


def get_rgb_img_from_file(file_path):
    raw_img = cv2.imread(file_path)
    raw_img = img_gbr_to_rgb(raw_img)
    return raw_img


def get_bgr_img_from_file(file_path):
    raw_img = cv2.imread(file_path)
    return raw_img


def get_img_gray_simple_seg(img_gray, img_seg_row_start, img_seg_row_end, img_seg_col_start, img_seg_col_end, is_copy=SEG_IMG_COPY):
    img_gray_seg = img_gray[img_seg_row_start:img_seg_row_end+1, img_seg_col_start:img_seg_col_end+1]
    if is_copy:
        img_gray_seg = img_copy(img_gray_seg)
    return img_gray_seg


def img_copy(img):
    #copied_img = img*1
    copied_img = img_mask_extract(img, 1)
    return copied_img


def get_img_gray_identical_intensity(img_dim, img_intensity):
    img_gray = np.ones(img_dim)
    img_gray = img_mask_extract(img_gray, img_intensity)
    return img_gray


def get_img_instance_copy_choice(img, is_img_copy):
    if is_img_copy:
        img_res = img_copy(img)
    else:
        img_res = img
    return img_res


def img_rgb_to_gray(rgb_img):
    global R_FACTOR
    global G_FACTOR
    global B_FACTOR
    global INT_GRAY_LEVEL_BAR

    # test
    # s_time = tm.time();
    # test end
    img_shape = rgb_img.shape
    img_h = img_shape[0]
    img_w = img_shape[1]
    img_gray = np.zeros((img_h, img_w))
    i = 0
    while i < img_h:
        img_row = rgb_img[i]
        j = 0
        while j < img_w:
            img_col = img_row[j]
            img_gray_level = (img_col[0] * R_FACTOR + img_col[1] * G_FACTOR + img_col[
                2] * B_FACTOR) / INT_GRAY_LEVEL_BAR
            if img_gray_level > 1:
                img_gray_level = 1
            img_gray[i, j] = img_gray_level
            j = j + 1
        i = i + 1
    # test
    # e_time = tm.time()
    # p_time = e_time - s_time
    # print("transfer time is ", p_time, "s")
    # test end
    return img_gray


def img_rgb_to_gray_array_cal(rgb_img):
    global R_FACTOR
    global G_FACTOR
    global B_FACTOR
    global INT_GRAY_LEVEL_BAR
    img_gray = np.floor((rgb_img[:, :, 0]*R_FACTOR+rgb_img[:, :, 1]*G_FACTOR+rgb_img[:, :, 2]*B_FACTOR))/INT_GRAY_LEVEL_BAR
    int_img_gray = np.int_(img_gray)
    #test
    #temp = img_gray*int_img_gray
    #test end
    #img_gray = img_gray-(img_gray*int_img_gray)+int_img_gray
    return img_gray


def img_gbr_to_rgb(gbr_img):
    rgb_img = cv2.cvtColor(gbr_img, cv2.COLOR_BGR2RGB)#gbr_img[:, :, ::-1]
    return rgb_img


def path_join(root, file_name):
    path_name = root + "/" + file_name
    return path_name


def package_img_file(dump_file_path, img_list):
    img_list_file = open(dump_file_path, 'wb')
    pik.dump(img_list, img_list_file)
    img_list_file.close()
    return 0


def extract_img_objects(img_file_path):
    img_list_file = open(img_file_path, 'rb')
    img_list = pik.load(img_list_file)
    img_list_file.close()
    return img_list


def init_list_storage_map(key_list):
    key_num = len(key_list)
    map_result = dict()
    i = 0
    while i < key_num:
        key = key_list[i]
        map_result[key] = []
        i += 1
    return map_result


def replace_file_extension(file_name, file_ext_list):
    res_file_name = file_name.replace(file_ext_list[0], file_ext_list[1], 1)
    return res_file_name


def get_mask_img(shape, mask_val=MASK_IMG_VAL_ONE):
    if mask_val == MASK_IMG_VAL_ONE:
        img_mask = np.ones(shape)
    elif mask_val == MASK_IMG_VAL_ZERO:
        img_mask = np.zeros(shape)
    elif mask_val == MASK_IMG_VAL_RANDOM:
        img_mask = np.random.ranf(shape)
    else:
        img_mask = np.ones(shape)*mask_val
    return img_mask


def get_seg_thin_line_mask_img_gray(shape, mask_val, seg_direction, seg_start, start_point, seg_pixel_num, seg_intensity_val):
    img_mask = get_mask_img(shape, mask_val)
    img_mask = img_gray_set_thin_line(img_mask, seg_direction, seg_start, start_point, seg_pixel_num, seg_intensity_val)
    return img_mask


def img_gray_set_thin_line(img_gray, seg_direction, seg_start, start_point, seg_pixel_num, seg_intensity_val):
    end_point = start_point + seg_pixel_num
    if seg_direction == SEG_DIRECTION_ROW:
        img_gray[seg_start, start_point:end_point] = seg_intensity_val
    else:
        img_gray[start_point:end_point, seg_start] = seg_intensity_val
    return img_gray


def whiten_img(img, map_positive=0):
    img_avg = np.average(img)
    # img_std = np.std(img)
    whitened_img = (img - img_avg) #/img_std
    if map_positive  == 1:
        whitened_img = img_positive_mapping(whitened_img)
    return whitened_img


def img_positive_mapping(img):
    pos_factor = np.abs(img.min())
    pos_map_img = img + pos_factor
    return pos_map_img


def img_mask_extract(img, img_mask):
    extracted_img = img*img_mask
    return extracted_img


def img_mask_erase(img, img_mask):
    img_shape = img_mask.shape
    img_reversed_mask = get_mask_img(img_shape, 1) - img_mask
    erased_img = img*img_reversed_mask
    return erased_img


def img_gray_extract_seg(img, seg_direction, seg_start):
    if seg_direction == SEG_DIRECTION_ROW:
        img_seg = img[seg_start, :]
    else:
        img_seg = img[:, seg_start]
    return img_seg


def get_img_gray_windowed_mask(img_shape, residual_window_list, mask_type=MASK_WHITE):
    boundary_mask = 1-mask_type
    mask_img_whole = get_mask_img(img_shape, boundary_mask)
    residual_window_num = len(residual_window_list)
    i = 0
    while i < residual_window_num:
        residual_window = residual_window_list[i]
        residual_window_left = residual_window[0]
        residual_window_right = residual_window[1]
        residual_window_top = residual_window[2]
        residual_window_bottom = residual_window[3]
        mask_shape = (residual_window_bottom - residual_window_top, residual_window_right - residual_window_left)
        mask_img_window = get_mask_img(mask_shape, mask_type)
        mask_img_whole[residual_window_top:residual_window_bottom, residual_window_left:residual_window_right] = mask_img_window
        i += 1
    return mask_img_whole


def img_gray_erased_boundary_whiten(img_erased, img_shape, residual_window_list):
    img_windowed_mask = get_img_gray_windowed_mask(img_shape, residual_window_list, MASK_BLACK)
    img_erased_whiten = img_erased + img_windowed_mask
    return img_erased_whiten


def img_gray_erase(img, residual_window_list, boundary_type=MASK_BLACK):
    img_shape = img.shape
    img_windowed_mask = get_img_gray_windowed_mask(img_shape, residual_window_list, MASK_WHITE)
    img_erased = img*img_windowed_mask
    if boundary_type == MASK_WHITE:
        img_erased = img_gray_erased_boundary_whiten(img_erased, img_shape, residual_window_list)
    return img_erased


def img_masked_insertion(img, img_mask, img_to_insert, is_erase_mask=[True, True]):
    if is_erase_mask[0]:
        erased_img = img_mask_erase(img, img_mask)
    else:
        erased_img = img
    if is_erase_mask[1]:
        masked_img_to_insert = img_to_insert*img_mask
    else:
        masked_img_to_insert = img_to_insert
    inserted_img = erased_img + masked_img_to_insert
    return inserted_img


def generate_factored_img_mask(img_mask, mask_factor):
    factored_img_mask = img_mask*mask_factor
    return factored_img_mask


def generate_uni_random_img_mask(img_mask):
    img_shape = img_mask.shape
    img_rnd = get_mask_img(img_shape, 2)
    uni_random_img_mask = img_masked_insertion(img_mask, img_mask, img_rnd)
    return uni_random_img_mask


def inverse_img(img):
    img_inverse = 2*np.average(img)-img
    return img_inverse


def img_gray_fourier_trans(img, is_norm=True):
    img_fft = np.fft.fft2(img)
    img_fft = np.absolute(img_fft)
    if is_norm:
        img_fft = img_fft/np.max(img_fft)
    return img_fft


def segment_list_fourier_trans(seg_list, is_norm=True):
    seg_fft = np.fft.fft(seg_list)
    seg_fft_abs = np.absolute(seg_fft)
    if is_norm:
        seg_fft_abs = seg_fft_abs/np.max(seg_fft_abs)
    list_len = len(seg_fft_abs)
    x_axs = generate_curve_display_x_axs(list_len)
    return [x_axs, seg_fft_abs, seg_fft]


def segment_list_fourier_inverse_trans(seg_list):
    seg_fft_i = np.fft.ifft(seg_list)
    seg_fft_r = np.real(seg_fft_i)
    list_len = len(seg_fft_i)
    x_axs = generate_curve_display_x_axs(list_len)
    return [x_axs, seg_fft_r, seg_fft_i]


def segment_list_frequency_filter(seg_list, frequency_range):
    frequency_scan = frequency_range[0]
    frequency_end = frequency_range[1]
    while frequency_scan <= frequency_end:
        seg_list[frequency_scan] = 0
        frequency_scan = frequency_scan + 1
    return seg_list


def img_gray_sift_key_points(img, is_output_gray=False, point_color=(255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS):
    sift_encoder = cv2.xfeatures2d.SIFT_create()
    img_gray1_int = np.int_(img * 255)
    img_gray1_int = img_gray1_int.astype(np.uint8)
    kp = sift_encoder.detect(img_gray1_int, None)
    img_gray1_sift = cv2.drawKeypoints(img_gray1_int, kp, np.array([]), point_color, flags)
    if is_output_gray:
        img_gray1_sift = img_rgb_to_gray_array_cal(img_gray1_sift)
    return img_gray1_sift


def img_gray_nonzero_val_extract(img):
    res_nonzero_val = []
    for val_row in img:
        for val in val_row:
            if val != 0:
                res_nonzero_val.append(val)
    res_nonzero_val = np.array(res_nonzero_val)
    return res_nonzero_val


def img_gray_nonzero_val_col_extract(img, is_none_zero_included=False):
    shape = img.shape
    r_len = shape[0]
    c_len = shape[1]
    res_col_list = []
    i = 0
    while i < c_len:
        col_val = img[:, i]
        col_res = []
        j = 0
        while j < r_len:
            val = col_val[j]
            if is_none_zero_included or val != 0:
                col_res.append(val)
            j = j + 1
        if len(col_res) > 0:
            col_res_np_array = np.array(col_res)
            res_col_list.append(col_res_np_array)
        i = i + 1
    return res_col_list


def img_gray_nonzero_val_row_extract(img, is_none_zero_included=False):
    shape = img.shape
    r_len = shape[0]
    c_len = shape[1]
    res_col_list = []
    i = 0
    while i < r_len:
        col_val = img[i, :]
        col_res = []
        j = 0
        while j < c_len:
            val = col_val[j]
            if is_none_zero_included or val != 0:
                col_res.append(val)
            j = j + 1
        if len(col_res) > 0:
            col_res_np_array = np.array(col_res)
            res_col_list.append(col_res_np_array)
        i = i + 1
    return res_col_list


def extract_img_seg_col(img_gray, seg_start):
    img_seg = img_gray[:, seg_start]
    return img_seg


def extract_img_seg_row(img_gray, seg_start):
    img_seg = img_gray[seg_start, :]
    return img_seg


def val_curve_display_extract(val_list, clip=0):
    if clip > 0:
        val_list = val_list[0:clip]
    list_len = len(val_list)
    x_axs = generate_curve_display_x_axs(list_len)
    return [x_axs, val_list]


def get_img_gray_seg(img_gray, seg_direction, seg_start, start_point, end_point):
    img_seg_extract_fun = get_img_gray_seg_extract_fun(seg_direction)
    img_seg = img_seg_extract_fun(img_gray, seg_start)
    img_seg = val_curve_extract(img_seg, [start_point, end_point])
    return img_seg


def get_img_gray_seg_extract_fun(seg_direction):
    if seg_direction == SEG_DIRECTION_ROW:
        img_seg_extract_fun = extract_img_seg_row
    else:
        img_seg_extract_fun = extract_img_seg_col
    return img_seg_extract_fun


def val_curve_extract(val_list, clip):
    extracted_curve = val_list[clip[0]:clip[1]]
    return extracted_curve


def generate_curve_display_x_axs(list_len):
    x_axs_list = [None] * list_len
    i = 0
    while i < list_len:
        x_axs_list[i] = i
        i = i + 1
    x_axs = np.array(x_axs_list)
    return x_axs


def mark_img_gray_seg(img_gray, seg_start, seg_direction, seg_point_start, seg_point_end, is_img_copy=SEG_IMG_COPY, mark_intensity=SEG_MARK_DARK):
    img_res = get_img_instance_copy_choice(img_gray, is_img_copy)
    img_seg_whole = get_img_gray_direction_seg(img_res, seg_start, seg_direction)
    seg_len = seg_point_end - seg_point_start
    seg_mark = np.zeros(np.int_(seg_len))
    if mark_intensity != SEG_MARK_DARK:
        seg_mark = mark_intensity
    img_seg_whole[seg_point_start:seg_point_end] = seg_mark
    return img_res


def generate_local_crack(img, center_point_row, center_point_col, scan_width, increment_factor, direction, local_direction, is_update_center, scale_factor=SCALE_NORMAL, crack_shape=CRACK_SHAPE_CONCAVE, crack_core_type=CRACK_CORE_DARK):  #local_direction should be 1 and -1 for upward and downward scan
    i = 1
    base_point_row = center_point_row
    base_point_col = center_point_col
    scan_point_row = base_point_row
    scan_point_col = base_point_col
    if is_update_center:
        base_intensity = img[base_point_row, base_point_col]
        base_intensity = np.fmin(base_intensity * INTENSITY_RES*scale_factor, 1)
        img[base_point_row, base_point_col] = base_intensity

    #if crack_shape == CRACK_SHAPE_BULGE:
    if crack_core_type == CRACK_CORE_DARK:
        base_intensity = 0.08#0.18#0.2#0.7#0.8#0.2
        scan_factor = 1
    else:
        base_intensity = 0.8
        scan_factor = -1
    img[base_point_row, base_point_col] = base_intensity
        #scan_width *= 3
    while i <= scan_width:
        base_intensity = img[base_point_row, base_point_col]
        # img[base_point_row, base_point_col] = base_intensity
        if direction == 0:
            scan_point_row = base_point_row + local_direction
            # scan_point_col = base_point_col
        else:
            # scan_point_row = base_point_row
            scan_point_col = base_point_col + local_direction
        #scan_intensity = img[scan_point_row, scan_point_col]
        #scan_intensity = scan_intensity * INTENSITY_RES
        #diff_intensity = scan_intensity - base_intensity
        #if diff_intensity < 0:
        # scan_intensity = base_intensity * CRACK_BASE_INCREMENT_FACTOR  # scan_intensity - 2*diff_intensity
        #scan_intensity = scan_intensity + increment_factor # /np.power(i, 2)
        #scan_width-i-1 i-1
        #scan_intensity = np.fmin(np.fmin(base_intensity + increment_factor*np.exp((1-scan_width)/SCAN_WIDTH_DISTRIBUTION_FACTOR+(4.7*(i)-1)/SCAN_STEP_DISTRIBUTION_FACTOR), INTENSITY_CONSTRAINT) * scale_factor, 1)# /np.power(i, 2) increment_factor/(i)
        if crack_shape == CRACK_SHAPE_BULGE:
            #scan_intensity = np.fmin(np.fmin(base_intensity + increment_factor * gs_gen.gauss_fun_trans_grad((i-1)*10, 5), INTENSITY_CONSTRAINT) * scale_factor, 1)  # /np.power(i, 2) increment_factor/(i)#0.8 9
            #scan_intensity = np.fmin(np.fmin(base_intensity + increment_factor * gs_gen.neg_exp_trans_grad((i)), INTENSITY_CONSTRAINT) * scale_factor, 1)
            if i <= scan_width*0.6:
                inc_factor = 0.03#-0.1#0.03
            else:
                inc_factor = 0.11#-0.1#0.11
            #scan_intensity = np.fmin(np.fmin(base_intensity + increment_factor * inc_factor, INTENSITY_CONSTRAINT) * scale_factor, 1)
            #scan_intensity = np.fmin(np.fmin(base_intensity + increment_factor * inc_factor*((0.35*i)), INTENSITY_CONSTRAINT) * scale_factor, 1)#lighter
            #scan_intensity = np.fmin(np.fmin(base_intensity + scan_factor*increment_factor * ((0.012 * i)), INTENSITY_CONSTRAINT) * scale_factor, 1)
            scan_intensity = np.fmin(np.fmin(base_intensity + scan_factor * increment_factor * gs_gen.square_fun_grad(i, 0.006, 0), INTENSITY_CONSTRAINT) * scale_factor, 1)
        else:

            #if i <= scan_width*0.4:
                #inc_factor = -0.04#0.04#-0.04
            #else:
                #inc_factor = 0.3#-0.3#0.3

            #if i <= scan_width*0.2:
                #inc_factor = -0.004
            #elif i <= scan_width*0.4:
                #inc_factor = 0.004
            #elif i <= scan_width*0.6:
                #inc_factor = 0.04
            #else:
                #inc_factor = 0.02
            if i <= scan_width*0.3:
                inc_factor = 0
            else:
                inc_factor = 1

            #scan_intensity = np.fmin(np.fmin(base_intensity + increment_factor * 2*0.125*(i-1), INTENSITY_CONSTRAINT) * scale_factor, 1)
            #scan_intensity = np.fmin(np.fmin(base_intensity + increment_factor * inc_factor, INTENSITY_CONSTRAINT) * scale_factor,1)
            scan_intensity = np.fmin(np.fmin(base_intensity + scan_factor*increment_factor * inc_factor * gs_gen.sigmoid_fun_grad(i, 100, 100, 5.5, 1.15), INTENSITY_CONSTRAINT) * scale_factor, 1)
        img[scan_point_row, scan_point_col] = scan_intensity
        if direction == 0:
            base_point_row = scan_point_row
        else:
            base_point_col = scan_point_col
        i = i + 1
    return img


def generate_line_crack_img_gary(img, crack_region, crack_width, increment_factor, direction=DIRECTION_H, scale_factor=SCALE_NORMAL, crack_shape=CRACK_SHAPE_CONCAVE, crack_core_type=CRACK_CORE_DARK):
    #img_cracked = img * 1
    img_cracked =img_copy(img)
    crack_range_scan = crack_region[0]
    crack_range_end = crack_region[1]
    crack_center_point_start = crack_region[2]
    if direction == 0:
        scan_row = crack_center_point_start
        scan_col = crack_range_scan
    else:
        scan_row = crack_range_scan
        scan_col = crack_center_point_start
    scan_width = crack_width/2
    crack_width_up = np.floor(scan_width)
    crack_width_down = np.ceil(scan_width)
    while crack_range_scan <= crack_range_end:
        img_cracked = generate_local_crack(img_cracked, scan_row, scan_col, crack_width_up, increment_factor, direction, SCAN_UP_WARD, True, scale_factor, crack_shape, crack_core_type)
        img_cracked = generate_local_crack(img_cracked, scan_row, scan_col, crack_width_down, increment_factor, direction, SCAN_DOWN_WARD, False, scale_factor, crack_shape, crack_core_type)
        if direction == 0:
            scan_col = scan_col + 1
        else:
            scan_row = scan_row + 1
        crack_range_scan = crack_range_scan + 1
    return img_cracked


def concat_images_from_list(image_list, col_num=G_FILTER_NORM_CONCAT_COL_NUM, clip=0):
    list_len = len(image_list)
    if clip > 0:
        list_len = clip
    row_num = np.int_(np.ceil(list_len/col_num))
    image = image_list[0]
    image_shape = image.shape
    image_row = image_shape[0]
    image_col = image_shape[1]
    concat_col_num = image_col*col_num
    concat_row_num = image_row*row_num
    res_concat_image = np.zeros((concat_row_num, concat_col_num))
    i = 0
    j = 0
    row_scan = 0
    row_scan_end = row_scan + image_row
    col_scan = 0
    while i < list_len:
        col_scan_end = col_scan+image_col
        res_concat_image[row_scan:row_scan_end, col_scan:col_scan_end] = image_list[i]
        col_scan = col_scan_end
        j = j + 1
        if j == col_num:
            row_scan = row_scan_end
            row_scan_end = row_scan + image_row
            col_scan = 0
            j = 0
        i = i + 1
    return res_concat_image


def generate_self_image_file_path_name(self_name_pref, self_name_ext, self_image_num):
    self_image_path_name = self_name_pref+str(self_image_num)+self_name_ext
    return self_image_path_name


def get_img_seg_wave_diff_rate(img_seg, seg_len):
    w_diff_sum = 0
    w_intensity_sum = img_seg[0]
    i = 1
    while i < seg_len:
        w_diff = np.abs(img_seg[i] - img_seg[i-1])
        w_diff_sum += w_diff
        w_intensity_sum += img_seg[i]
        i += 1
    w_diff_rate = w_diff_sum/seg_len
    w_intensity_sum_rate = w_intensity_sum/seg_len
    return [w_diff_rate, w_intensity_sum_rate]


def save_gray_image(img_gray, file_path_name, is_source_normalized=True):
    #img_gray = np.expand_dims(img, axis=2)
    #img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    if is_source_normalized:
        img_gray = np.round(img_gray*255, 0)
    cv2.imwrite(file_path_name, img_gray)
    return img_gray


def render_img_gray_seg_to_bc_shape(img, seg_start, seg_direction, seg_mod_start, seg_mod_range, scan_direction, render_direction, bc_shape, is_img_copy=SEG_IMG_COPY):
    img_res = get_img_instance_copy_choice(img, is_img_copy)
    if seg_direction == SEG_DIRECTION_ROW:
        img_seg = img_res[seg_start, :]
    else:
        img_seg = img[:, seg_start]
    scan_adjust_param = get_scan_point_adjust_param(seg_mod_range, scan_direction)
    scan_adjust_param_1 = scan_adjust_param[0]
    scan_adjust_param_2 = scan_adjust_param[1]
    i = 1
    seg_intensity = img_seg[seg_mod_start]
    j = seg_mod_start + scan_direction
    while i < seg_mod_range:
        scan_pos = scan_adjust_param_1+scan_adjust_param_2*i
        if bc_shape == SEG_SHAPE_BULGE:
            seg_intensity = crack_bulge_intensity_generator(seg_intensity, scan_pos, render_direction)
        elif bc_shape == SEG_SHAPE_CONCAVE:
            seg_intensity = crack_concave_intensity_generator(seg_intensity, scan_pos, render_direction)
        else:
            seg_intensity = crack_linear_intensity_generator(seg_intensity, scan_pos, render_direction)
        img_seg[j] = seg_intensity
        i += 1
        j += scan_direction
    return img_res


def crack_bulge_intensity_generator(base_intensity, scan_pos, scan_factor=SEG_RENDER_INC, increment_factor=0.8, scale_factor=1):
    '''
    scan_intensity = np.fmin(
        np.fmin(base_intensity + scan_factor * increment_factor * gs_gen.square_fun_grad(scan_pos, 0.006, 0),
                INTENSITY_CONSTRAINT) * scale_factor, 1)#0.006 0.004
    '''
    scan_intensity = np.fmin(
        np.fmin(base_intensity + scan_factor * increment_factor * gs_gen.cubic_fun_grad(scan_pos, 0.0008, 0, 0, 0),
                INTENSITY_CONSTRAINT) * scale_factor, 1)  # 0.006 0.004
    return scan_intensity


def crack_concave_intensity_generator(base_intensity, scan_pos, scan_factor=SEG_RENDER_INC, increment_factor=0.8, scale_factor=1):#increment_factor=0.4
    '''
    scan_intensity = np.fmin(np.fmin(
        base_intensity + scan_factor * increment_factor * gs_gen.sigmoid_fun_grad(scan_pos, 100, 100, 5.5, 1.15),
        INTENSITY_CONSTRAINT) * scale_factor, 1)#12
    '''
    scan_intensity = np.fmin(np.fmin(
        base_intensity + scan_factor * increment_factor * gs_gen.square_root_fun_grad(scan_pos, 0.15),
        INTENSITY_CONSTRAINT) * scale_factor, 1)  # 12
    return scan_intensity


def crack_linear_intensity_generator(base_intensity, scan_pos, scan_factor=SEG_RENDER_INC, increment_factor=1, scale_factor=1):#increment_factor=0.4
    scan_intensity = np.fmin(np.fmin(
        base_intensity + scan_factor * increment_factor * gs_gen.linear_fun_grad(scan_pos, 0.01, 0),
        INTENSITY_CONSTRAINT) * scale_factor, 1)
    return scan_intensity


def img_gray_generate_shaped_crack_line(img_gray, seg_start, seg_end, seg_direction, seg_scan_direction, scan_bias, center_status, start_point, half_range, seg_shape, initial_copy, is_random=SEG_RANDOM):
    initial_inc_params = get_intensity_increment_param(center_status, seg_scan_direction)
    initial_points = get_initial_points(seg_scan_direction, start_point, half_range)
    scan_directions = get_scan_directions(seg_scan_direction)
    initial_inc_param_1 = initial_inc_params[0]
    initial_inc_param_2 = initial_inc_params[1]
    initial_point_1 = initial_points[0]
    initial_point_2 = initial_points[1]
    scan_direction_1 = scan_directions[0]
    scan_direction_2 = scan_directions[1]
    img_gray_mod = render_img_gray_seg_to_bc_shape(img_gray, seg_start, seg_direction, initial_point_1, half_range, scan_direction_1, initial_inc_param_1, seg_shape, initial_copy)
    img_gray_mod = render_img_gray_seg_to_bc_shape(img_gray_mod, seg_start, seg_direction, initial_point_2, half_range, scan_direction_2, initial_inc_param_2, seg_shape, SEG_IMG_NO_COPY)
    i = seg_start+1
    j = scan_bias
    while i <= seg_end:
        img_gray_mod = render_img_gray_seg_to_bc_shape(img_gray_mod, i, seg_direction, initial_point_1+j, half_range, scan_direction_1, initial_inc_param_1, seg_shape, SEG_IMG_NO_COPY)
        img_gray_mod = render_img_gray_seg_to_bc_shape(img_gray_mod, i, seg_direction, initial_point_2+j, half_range, scan_direction_2, initial_inc_param_2, seg_shape, SEG_IMG_NO_COPY)
        i = i + 1
        if is_random:
            j = j + np.random.randint(1, 6)
        j = j + scan_bias
    return img_gray_mod


def get_intensity_increment_param(center_status, scan_direction):
    if scan_direction == SEG_SCAN_POS:
        if center_status == CRACK_CORE_LIGHT:
            inc_parameter = [SEG_RENDER_INC, SEG_RENDER_DEC]
        else:
            inc_parameter = [SEG_RENDER_DEC, SEG_RENDER_INC]
    else:
        if center_status == CRACK_CORE_LIGHT:
            inc_parameter = [SEG_RENDER_INC, SEG_RENDER_INC]
        else:
            inc_parameter = [SEG_RENDER_DEC, SEG_RENDER_DEC]
    return inc_parameter


def get_initial_points(scan_direction, start_point, half_range):
    if scan_direction == SEG_SCAN_POS:
        initial_points = [start_point, start_point+(half_range-1)]
    else:
        initial_points = [start_point, start_point]
    return initial_points


def get_scan_directions(seg_scan_direction):
    if seg_scan_direction == SEG_SCAN_POS:
        scan_directions = [SEG_SCAN_POS, SEG_SCAN_POS]
    else:
        scan_directions = [SEG_SCAN_NEG, SEG_SCAN_POS]
    return scan_directions


def get_scan_point_adjust_param(scan_range, scan_direction):
    if scan_direction == SEG_SCAN_POS:
        adjust_param = [scan_range, -1]
    else:
        adjust_param = [0, 1]
    return adjust_param


def get_seg_shape_param(seg_shape):
    if isinstance(seg_shape, list):
        seg_shape_1 = seg_shape[0]
        seg_shape_2 = seg_shape[1]
    else:
        seg_shape_1 = seg_shape
        seg_shape_2 = seg_shape
    return [seg_shape_1, seg_shape_2]


def get_seg_climb_range_param(climb_range):
    if isinstance(climb_range, list):
        seg_climb_range_1 = climb_range[0]
        seg_climb_range_2 = climb_range[1]
    else:
        seg_climb_range_1 = climb_range
        seg_climb_range_2 = climb_range
    return [seg_climb_range_1, seg_climb_range_2]


def img_gray_generate_shaped_crack_line_diverted_seg(img_gray, seg_start, seg_end, seg_direction, seg_scan_direction, scan_bias, center_status_inc_param, start_point, half_range, seg_shape, initial_copy, is_random=SEG_RANDOM):
    #if isinstance(center_status_inc_param, list):
        #initial_inc_params = center_status_inc_param
    #else:
        #initial_inc_params = get_intensity_increment_param_diverted_seg(center_status_inc_param, seg_scan_direction)
    initial_inc_params = get_seg_bc_shape_inc_param(center_status_inc_param, seg_scan_direction)
    '''
    if isinstance(seg_shape, list):
        seg_shape_1 = seg_shape[0]
        seg_shape_2 = seg_shape[1]
    else:
        seg_shape_1 = seg_shape
        seg_shape_2 = seg_shape
    '''
    seg_shape_param = get_seg_shape_param(seg_shape)
    seg_shape_1 = seg_shape_param[0]
    seg_shape_2 = seg_shape_param[1]
    seg_climb_range_param = get_seg_climb_range_param(half_range)
    seg_climb_range_1 = seg_climb_range_param[0]
    seg_climb_range_2 = seg_climb_range_param[1]
    seg_curve_list = []
    initial_points = get_initial_points(seg_scan_direction, start_point, seg_climb_range_1)
    scan_directions = get_scan_directions(seg_scan_direction)
    #initial_inc_param_1 = initial_inc_params[0]
    #initial_inc_param_2 = initial_inc_params[1]
    initial_inc_fun = initial_inc_params[0]
    initial_inc_fun_param = initial_inc_params[1]
    seg_scan_idx = 0
    initial_point_1 = initial_points[0]
    initial_point_2 = initial_points[1]
    scan_direction_1 = scan_directions[0]
    scan_direction_2 = scan_directions[1]
    initial_inc_param = initial_inc_fun(initial_inc_fun_param, seg_scan_idx)
    img_gray_mod = render_img_gray_seg_to_bc_shape_diverted_seg(img_gray, seg_start, seg_direction, initial_point_1, seg_climb_range_1, scan_direction_1, initial_inc_param[0], seg_shape_1, initial_copy)
    img_gray_mod = extract_diverted_seg_img_gray_mod(img_gray_mod, seg_curve_list)
    img_gray_mod = render_img_gray_seg_to_bc_shape_diverted_seg(img_gray_mod, seg_start, seg_direction, initial_point_2, seg_climb_range_2, scan_direction_2, initial_inc_param[1], seg_shape_2, SEG_IMG_NO_COPY)
    img_gray_mod = extract_diverted_seg_img_gray_mod(img_gray_mod, seg_curve_list)
    i = seg_start+1
    j = scan_bias
    while i <= seg_end:
        seg_scan_idx += 1
        initial_inc_param = initial_inc_fun(initial_inc_fun_param, seg_scan_idx)
        img_gray_mod = render_img_gray_seg_to_bc_shape_diverted_seg(img_gray_mod, i, seg_direction, initial_point_1+j, seg_climb_range_1, scan_direction_1, initial_inc_param[0], seg_shape_1, SEG_IMG_NO_COPY)
        img_gray_mod = extract_diverted_seg_img_gray_mod(img_gray_mod, seg_curve_list)
        img_gray_mod = render_img_gray_seg_to_bc_shape_diverted_seg(img_gray_mod, i, seg_direction, initial_point_2+j, seg_climb_range_2, scan_direction_2, initial_inc_param[1], seg_shape_2, SEG_IMG_NO_COPY)
        img_gray_mod = extract_diverted_seg_img_gray_mod(img_gray_mod, seg_curve_list)
        i = i + 1
        if is_random:
            j = j + np.random.randint(1, 6)
        j = j + scan_bias
    if SEG_RET_CURVE_LIST:
        img_gray_mod = [img_gray_mod, seg_curve_list]
    return img_gray_mod


def img_gray_generate_bc_shape_edge_rect(img_gray, rect_top_left, rect_bottom_right, edge_width, edge_climb_ratio, center_static_val, edge_shape):
    rect_center_param = get_img_gray_bc_shape_rect_center_scan_param(rect_top_left, rect_bottom_right, edge_width)
    rect_vertical_param = get_img_gray_bc_shape_rect_vertical_scan_param(rect_top_left, rect_bottom_right, edge_width)
    rect_horizontal_param = get_img_gray_bc_shape_rect_horizontal_scan_param(rect_top_left, rect_bottom_right, edge_width)
    rect_center_mod_section = get_img_gray_rect_mod_section(img_gray, rect_center_param[0], rect_center_param[1], rect_center_param[2], rect_center_param[3])
    rect_horizontal_mod_section_1 = get_img_gray_rect_mod_section(img_gray, rect_horizontal_param[2], rect_horizontal_param[3], rect_horizontal_param[0], rect_horizontal_param[1], SEG_RECT_TRANSPOSE)
    rect_horizontal_mod_section_2 = get_img_gray_rect_mod_section(img_gray, rect_horizontal_param[4], rect_horizontal_param[5], rect_horizontal_param[0], rect_horizontal_param[1], SEG_RECT_TRANSPOSE)
    rect_vertical_mod_section_1 = get_img_gray_rect_mod_section(img_gray, rect_vertical_param[0], rect_vertical_param[1], rect_vertical_param[2], rect_vertical_param[3])
    rect_vertical_mod_section_2 = get_img_gray_rect_mod_section(img_gray, rect_vertical_param[0], rect_vertical_param[1], rect_vertical_param[4], rect_vertical_param[5])
    original_static_val = set_seg_curve_val_static(center_static_val)
    render_img_gray_bc_shape_rect_section_row_wise(rect_center_mod_section, edge_climb_ratio[0], SEG_CURVE_STATIC)#SEG_CURVE_STATIC
    set_seg_curve_val_static(original_static_val)
    render_img_gray_bc_shape_rect_section_row_wise(rect_horizontal_mod_section_1,edge_climb_ratio[0], edge_shape)
    render_img_gray_bc_shape_rect_section_row_wise(rect_horizontal_mod_section_2, edge_climb_ratio[1], edge_shape, SEG_SCAN_NEG)
    render_img_gray_bc_shape_rect_section_row_wise(rect_vertical_mod_section_1, edge_climb_ratio[0], edge_shape)
    render_img_gray_bc_shape_rect_section_row_wise(rect_vertical_mod_section_2, edge_climb_ratio[1], edge_shape, SEG_SCAN_NEG)
    reset_seg_curve_val_static()
    return img_gray


def get_img_gray_rect_mod_section(img_gray, row_start, row_end_idx, col_start, col_end_idx, is_transpose=SEG_RECT_NO_TRANSPOSE):
    img_gray_rect_mod_section = img_gray[row_start:row_end_idx, col_start:col_end_idx]
    if is_transpose:
        img_gray_rect_mod_section = img_gray_rect_mod_section.transpose()
    return img_gray_rect_mod_section


def render_img_gray_bc_shape_rect_section_row_wise(rect_section, edge_climb_ratio, edge_shape, scan_direction=SEG_SCAN_POS):
    img_gray_mod = rect_section
    rect_shape = img_gray_mod.shape
    climb_range = rect_shape[1]
    seg_num = rect_shape[0]
    seg_curve_list = []
    i = 0
    if scan_direction == SEG_SCAN_NEG:
        initial_point = climb_range - 1
    else:
        initial_point = 0
    while i < seg_num:
        img_gray_mod = render_img_gray_seg_to_bc_shape_diverted_seg(img_gray_mod, i, SEG_DIRECTION_ROW, initial_point,
                                                                    climb_range, scan_direction, edge_climb_ratio,
                                                                    edge_shape, SEG_IMG_NO_COPY)
        img_gray_mod = extract_diverted_seg_img_gray_mod(img_gray_mod, seg_curve_list)
        i += 1
    return img_gray_mod


def get_img_gray_bc_shape_rect_vertical_scan_param(rect_top_left, rect_bottom_right, edge_width):
    v_seg_start = rect_top_left[0]
    v_seg_end = rect_bottom_right[0] + 1
    v_start_point_1 = rect_top_left[1]
    v_end_point_1 = v_start_point_1 + edge_width
    v_end_point_2 = rect_bottom_right[1] + 1
    v_start_point_2 = v_end_point_2 - edge_width
    return [v_seg_start, v_seg_end, v_start_point_1, v_end_point_1, v_start_point_2, v_end_point_2]


def get_img_gray_bc_shape_rect_horizontal_scan_param(rect_top_left, rect_bottom_right, edge_width):
    h_seg_start = rect_top_left[1] + edge_width
    h_seg_end = rect_bottom_right[1] - edge_width + 1
    h_seg_start_point_1 = rect_top_left[0]
    h_seg_end_point_1 = h_seg_start_point_1 + edge_width
    h_seg_end_point_2 = rect_bottom_right[0] + 1
    h_seg_start_point_2 = h_seg_end_point_2 - edge_width -1
    return [h_seg_start, h_seg_end, h_seg_start_point_1, h_seg_end_point_1, h_seg_start_point_2, h_seg_end_point_2]


def get_img_gray_bc_shape_rect_center_scan_param(rect_top_left, rect_bottom_right, edge_width):
    c_seg_start = rect_top_left[0] + edge_width
    c_seg_end = rect_bottom_right[0] - edge_width
    c_start_point = rect_top_left[1] + edge_width
    c_end_point = rect_bottom_right[1] - edge_width + 1
    return [c_seg_start, c_seg_end, c_start_point, c_end_point]


def simple_seg_bc_curve_texture_func(inc_param, seg_scan_idx):
    return inc_param


def seg_core_interchange_texture_func(inc_param, seg_scan_idx):
    #param_idx = np.int_(seg_scan_idx/3) % 2
    param_idx = np.int_(seg_scan_idx) % 2
    return inc_param[param_idx]


def seg_core_interchange_texture_func_1(inc_param, seg_scan_idx):
    param_idx = np.int_(seg_scan_idx/3) % 2
    #param_idx = np.int_(seg_scan_idx) % 2
    return inc_param[param_idx]


def seg_core_interchange_texture_static_func(inc_param, seg_scan_idx):
    #global SEG_CURVE_VAL_STATIC
    interchange_checker = np.int_(seg_scan_idx/3) % 2
    #interchange_checker = np.int_(seg_scan_idx) % 2
    if interchange_checker == 0:
        #SEG_CURVE_VAL_STATIC = inc_param[0]
        set_seg_curve_val_static(inc_param[0])
    else:
        set_seg_curve_val_static(inc_param[1])
        #SEG_CURVE_VAL_STATIC = inc_param[1]
    return [-0.9, 0.9]


def reset_seg_curve_val_static():
    global SEG_CURVE_VAL_STATIC
    SEG_CURVE_VAL_STATIC = SEG_CURVE_VAL_STATIC_DEFAULT
    return


def set_seg_curve_val_static(val_static):
    global SEG_CURVE_VAL_STATIC
    original_static_val = SEG_CURVE_VAL_STATIC
    SEG_CURVE_VAL_STATIC = val_static
    return original_static_val


def get_seg_bc_shape_inc_param(center_status_inc_param, seg_scan_direction):
    if isinstance(center_status_inc_param, list):
        if callable(center_status_inc_param[0]):
            initial_inc_params = center_status_inc_param
        else:
            initial_inc_params = [simple_seg_bc_curve_texture_func, center_status_inc_param]
    else:
        initial_inc_params = get_intensity_increment_param_diverted_seg(center_status_inc_param, seg_scan_direction)
    return initial_inc_params


def get_intensity_increment_param_diverted_seg(center_status, scan_direction):
    if scan_direction == SEG_SCAN_POS:
        if center_status == CRACK_CORE_LIGHT:
            inc_parameter = [3, -0.4]
        else:
            inc_parameter = [-0.8, 8]
    else:
        if center_status == CRACK_CORE_LIGHT:
            inc_parameter = [-0.8, -0.8]
        else:
            inc_parameter = [8, 8]
    return [simple_seg_bc_curve_texture_func, inc_parameter]


def extract_diverted_seg_img_gray_mod(img_gray_mod, seg_curve_list):
    seg_curve_list.append(img_gray_mod[1])
    return img_gray_mod[0]


def get_img_gray_direction_seg(img_gray, seg_start, seg_direction):
    if seg_direction == SEG_DIRECTION_ROW:
        img_seg = img_gray[seg_start, :]
    else:
        img_seg = img_gray[:, seg_start]
    return img_seg


def render_img_gray_seg_to_bc_shape_diverted_seg(img, seg_start, seg_direction, seg_mod_start, seg_mod_range, scan_direction, intensity_climb_ratio, climb_shape, is_img_copy=SEG_IMG_COPY):
    img_res = get_img_instance_copy_choice(img, is_img_copy)
    '''
    if seg_direction == SEG_DIRECTION_ROW:
        img_seg = img_res[seg_start, :]
    else:
        img_seg = img_res[:, seg_start]
    '''
    img_seg = get_img_gray_direction_seg(img_res, seg_start, seg_direction)
    seg_intensity = img_seg[seg_mod_start]
    #seg_curve = generate_linear_diverted_random_climb_line_seg(seg_intensity, intensity_climb_ratio, seg_mod_range, climb_shape, scan_direction)
    seg_curve = generate_linear_diverted_random_climb_line_seg_monotonic(seg_intensity, intensity_climb_ratio, seg_mod_range, climb_shape, scan_direction)
    if scan_direction == SEG_SCAN_NEG:
        seg_mod_start = seg_mod_start - seg_mod_range + 1
    img_seg[seg_mod_start:seg_mod_start+seg_mod_range] = seg_curve[:]
    if CURVE_DISPLAY:
        climb_curve_display = val_curve_display_extract(seg_curve)
        plt.plot(climb_curve_display[0], climb_curve_display[1])
        plt.show()
    return [img_res, seg_curve]


def generate_linear_diverted_random_climb_line_seg(initial_intensity, intensity_climb_ratio, climb_range, climb_shape, scan_direction, initial_angle_bound_ratio=SEG_CURVE_INITIAL_ANGLE_BOUND_RATIO, intensity_upper_bound=SEG_CURVE_INTENSITY_UPPER_BOUND, index_shrink_ratio=SEG_CURVE_SHRINK_RATIO_DEFAULT):
    target_intensity = np.fmax(np.fmin(initial_intensity*(1+intensity_climb_ratio), intensity_upper_bound), SEG_INTENSITY_LOWER_BOUND)
    abs_intensity_delta = np.abs(initial_intensity - target_intensity)
    intensity_tan = abs_intensity_delta/(climb_range*index_shrink_ratio)
    linear_theta = np.arctan(intensity_tan)
    start_intensity = np.fmin(target_intensity, initial_intensity)
    end_intensity = np.fmax(target_intensity, initial_intensity)
    climb_shape_param = get_climb_line_seg_shape_param(np.int_(climb_shape))
    climb_shape_param_1 = climb_shape_param[0]
    climb_shape_param_2 = climb_shape_param[1]
    index_adjust_param = get_climb_index_adjust_param(initial_intensity, end_intensity, start_intensity, scan_direction, climb_range)
    index_adjust_pref = index_adjust_param[0]
    index_adjust_cof = index_adjust_param[1]
    angle_bound = get_climb_initial_angle_bound(linear_theta, initial_angle_bound_ratio, climb_shape, end_intensity, index_shrink_ratio)
    climb_seg = np.ones(climb_range)
    i = 0
    scan_intensity = start_intensity
    scan_angle = linear_theta
    seg_index = get_adjusted_climb_index(index_adjust_pref, index_adjust_cof, i)#index_adjust_pref + index_adjust_cof * i
    climb_seg[seg_index] = scan_intensity
    scan_angle += get_climb_adjust_angle(climb_shape_param_1, angle_bound)#climb_shape_param_1*get_random_angle(angle_bound)
    i += 1
    if start_intensity == initial_intensity:
        gen_round_num = climb_range
    else:
        gen_round_num = climb_range - 1
    while i < gen_round_num:
        tan_scan_angle = np.tan(scan_angle)
        scan_intensity = start_intensity + i * index_shrink_ratio * tan_scan_angle
        seg_index = get_adjusted_climb_index(index_adjust_pref, index_adjust_cof, i)#index_adjust_pref + index_adjust_cof * i
        climb_seg[seg_index] = np.fmin(scan_intensity, intensity_upper_bound)
        angle_bound = get_scan_angle_bound(linear_theta, scan_angle)
        scan_angle += get_climb_adjust_angle(climb_shape_param_2, angle_bound)#climb_shape_param_2*get_random_angle(angle_bound)
        i += 1
    if end_intensity == initial_intensity:
        seg_index = get_adjusted_climb_index(index_adjust_pref, index_adjust_cof, i)# index_adjust_pref + index_adjust_cof * i
        climb_seg[seg_index] = end_intensity
    return climb_seg


def get_climb_line_seg_shape_param(climb_shape):
    if climb_shape == SEG_CURVE_BULGE_DIVERSION:
        shape_param = [-1, 1]
    elif climb_shape == SEG_CURVE_CONCAVE_DIVERSION:
        shape_param = [1, -1]
    else:
        shape_param = [0, 0]
    return shape_param


def get_climb_initial_angle_bound(initial_angle, initial_angle_bound_ratio, climb_shape, initial_max, index_shrink_ratio):
    angle_bound = initial_angle
    if climb_shape == SEG_CURVE_CONCAVE_DIVERSION:
        angle_bound = np.arctan(initial_max/index_shrink_ratio) - angle_bound
    angle_bound = angle_bound * initial_angle_bound_ratio
    return angle_bound


def get_random_angle(angle_bound):
    random_angle = np.fmax(np.random.ranf()*angle_bound*SEG_CURVE_CLIMB_SPEED_RATIO, SEG_ANGLE_LOWER_BOUND)
    return random_angle


def get_scan_angle_bound(linear_angle, scan_angle):
    angle_bound = np.fmax(linear_angle, scan_angle) - np.fmin(linear_angle, scan_angle)
    return angle_bound


def get_climb_index_adjust_param(initial_intensity, max_intensity, mini_intensity, scan_direction, climb_range):
    if (initial_intensity == max_intensity and scan_direction == SEG_SCAN_POS) or (initial_intensity == mini_intensity and scan_direction == SEG_SCAN_NEG):
        index_adjust_param = [climb_range-1, -1]
    else:
        index_adjust_param = [0, 1]
    return index_adjust_param


def get_adjusted_climb_index(index_adjust_pref, index_adjust_cof, scan_index):
    adjusted_index = index_adjust_pref + index_adjust_cof * scan_index
    return adjusted_index


def get_climb_adjust_angle(climb_shape_param, angle_bound):
    adjust_angle = climb_shape_param * get_random_angle(angle_bound)
    return adjust_angle


def curve_list_display(curve_list):
    curve_list_len = len(curve_list)
    curve_display_col_num = np.int_(SEG_CURVE_DISPLAY_COL_NUM)
    curve_display_row_num = np.int_(np.ceil(curve_list_len/curve_display_col_num))
    i = 1
    while i <= curve_list_len:
        climb_curve_display = val_curve_display_extract(curve_list[i-1])
        plt.subplot(curve_display_row_num, curve_display_col_num, i)
        plt.plot(climb_curve_display[0], climb_curve_display[1])
        i += 1
    plt.show()
    return


def generate_ranged_float_rand(lower_bound, upper_bound):
    r = np.random.ranf()
    r = r * (upper_bound - lower_bound) + lower_bound
    return r


def get_concave_val(prev_val, line_val, target_val, upper_bound_ratio=1, lower_bound_ratio=0):
    upper_bound = target_val * upper_bound_ratio
    lower_bound = np.fmax(prev_val, line_val)
    lower_bound_exp = lower_bound*(1+lower_bound_ratio)
    if lower_bound_exp < upper_bound:
        lower_bound = lower_bound_exp
    else:
        lower_bound = lower_bound+(upper_bound-lower_bound)*0.1
    concave_val = generate_ranged_float_rand(lower_bound, upper_bound)
    return concave_val


def get_concave_val_grad_monotonic(prev_val_1, prev_val_2, line_alpha, x, b, target_val, shrinked_target_x, index_shrink_ratio):
    if x == 1:
        line_val = gs_gen.linear_fun(x*index_shrink_ratio, line_alpha, b)
        lower_bound = line_val * (1 + SEG_CONCAVE_GRAD_MONOTONIC_INITIAL_BOUND_RATIO)
        upper_bound = lower_bound*(1 + SEG_CONCAVE_GRAD_MONOTONIC_INITIAL_BOUND_RATIO)
    else:
        prev_line_param = gs_gen.linear_parameter_resolver(prev_val_1, prev_val_2, (x-2)*index_shrink_ratio, (x-1)*index_shrink_ratio)
        prev_target_line_param = gs_gen.linear_parameter_resolver(prev_val_2, target_val, (x-1)*index_shrink_ratio, shrinked_target_x)
        upper_bound = gs_gen.linear_fun(x*index_shrink_ratio, prev_line_param[0], prev_line_param[1])
        lower_bound = gs_gen.linear_fun(x*index_shrink_ratio, prev_target_line_param[0], prev_target_line_param[1])
        if upper_bound < lower_bound:
            print("bound mixed")
        bound_delta = upper_bound - lower_bound
        bound_adjuster = generate_ranged_float_rand(0.3, 0.4) #(0.4,0.6)
        upper_bound = upper_bound - bound_delta*SEG_CONCAVE_MONOTONIC_GRAD_UPPER_BOUND_RATIO*x*bound_adjuster
        lower_bound = lower_bound + bound_delta*SEG_CONCAVE_MONOTONIC_GRAD_LOWER_BOUND_RATIO/(x*bound_adjuster)

    concave_val = np.fmin(generate_ranged_float_rand(lower_bound, upper_bound), target_val)
    if concave_val < prev_val_2:
        print("value unchanged")
    return concave_val


def get_bulge_val_grad_monotonic(prev_val_1, prev_val_2, line_alpha, x, b, target_val, shrinked_target_x, index_shrink_ratio):
    if x == 1:
        line_val = gs_gen.linear_fun(x*index_shrink_ratio, line_alpha, b)
        bound_delta = line_val - prev_val_2
        upper_bound = line_val - bound_delta*SEG_BULGE_GRAD_MONOTONIC_INITIAL_BOUND_RATIO
        lower_bound = prev_val_2 + bound_delta*SEG_BULGE_GRAD_MONOTONIC_INITIAL_BOUND_RATIO
    else:
        prev_line_param = gs_gen.linear_parameter_resolver(prev_val_1, prev_val_2, (x-2)*index_shrink_ratio, (x-1)*index_shrink_ratio)
        prev_target_line_param = gs_gen.linear_parameter_resolver(prev_val_2, target_val, (x-1)*index_shrink_ratio, shrinked_target_x)
        lower_bound = gs_gen.linear_fun(x*index_shrink_ratio, prev_line_param[0], prev_line_param[1])
        upper_bound = gs_gen.linear_fun(x*index_shrink_ratio, prev_target_line_param[0], prev_target_line_param[1])
        bound_delta = upper_bound - lower_bound
        bound_adjuster = generate_ranged_float_rand(0.4, 0.8)
        upper_bound = upper_bound - bound_delta*SEG_BULGE_MONOTONIC_GRAD_UPPER_BOUND_RATIO/(x*bound_adjuster)
        lower_bound = lower_bound + bound_delta*SEG_BULGE_MONOTONIC_GRAD_LOWER_BOUND_RATIO*x*bound_adjuster
    bulge_val = np.fmin(generate_ranged_float_rand(lower_bound, upper_bound), target_val)
    return bulge_val


def get_linear_val_grad_monotonic(prev_val_1, prev_val_2, line_alpha, x, b, target_val, shrinked_target_x, index_shrink_ratio):
    line_val = gs_gen.linear_fun(x*index_shrink_ratio, line_alpha, b)
    return line_val


def get_static_val_grad_monotonic(prev_val_1, prev_val_2, line_alpha, x, b, target_val, shrinked_target_x, index_shrink_ratio):
    return SEG_CURVE_VAL_STATIC


def get_flat_initial_lower_bound_adjuster(line_val, target_val):
    rnd = np.random.ranf()
    if rnd <= 0.5:
        bound_adjuster = SEG_FLAT_INITIAL_SECOND_BOUND_RATIO_LOWER*line_val-line_val
    else:
        target_line_delta = target_val - line_val
        bound_adjuster = target_line_delta*SEG_FLAT_INITIAL_SECOND_BOUND_RATIO_UPPER
    return bound_adjuster


def get_flat_val_monotonic_grad_flat(prev_val_1, prev_val_2, line_alpha, x, b, target_val, shrinked_target_x, index_shrink_ratio):
    if x == 1:
        line_val = gs_gen.linear_fun(x * index_shrink_ratio, line_alpha, b)
        initial_bound_adjuster = get_flat_initial_lower_bound_adjuster(line_val, target_val)
        adjusted_line_val = line_val + initial_bound_adjuster
        upper_bound = np.fmax(line_val, adjusted_line_val)
        lower_bound = np.fmin(line_val, adjusted_line_val)
    else:
        lower_bound = prev_val_2
        bound_delta = target_val - lower_bound
        upper_bound = target_val - bound_delta*SEG_FLAT_UPPER_BOUND_RATIO
    flat_val = generate_ranged_float_rand(lower_bound, upper_bound)
    return flat_val


def get_stable_val_monotonic_grad_stable(prev_val_1, prev_val_2, line_alpha, x, b, target_val, shrinked_target_x, index_shrink_ratio):
    return prev_val_1


def get_triangle_val_monotonic_grad_linear(prev_val_1, prev_val_2, line_alpha, x, b, target_val, shrinked_target_x, index_shrink_ratio):
    if prev_val_1 >= prev_val_2:
        intensity_val = prev_val_2 + SEG_CURVE_DOWNWARD_TRIANGLE_LINEAR_GRAD
        intensity_val = np.fmin(intensity_val, 1)
    else:
        intensity_val = prev_val_1
    return intensity_val


def get_intensity_kept_val_monotonic_static(prev_val_1, prev_val_2, line_alpha, x, b, target_val, shrinked_target_x, index_shrink_ratio):
    return prev_val_2


def get_val_grad_monotonic_func(climb_shape):
    if climb_shape == SEG_CURVE_CONCAVE_DIVERSION:
        return get_concave_val_grad_monotonic
    elif climb_shape == SEG_CURVE_BULGE_DIVERSION:
        return get_bulge_val_grad_monotonic
    elif climb_shape == SEG_CURVE_LINEAR:
        return get_linear_val_grad_monotonic
    elif climb_shape == SEG_CURVE_STATIC:
        return get_static_val_grad_monotonic
    elif climb_shape == SEG_CURVE_FLAT:
        return get_flat_val_monotonic_grad_flat
    elif climb_shape == SEG_CURVE_DOWNWARD_TRIANGLE_LINEAR:
        return get_triangle_val_monotonic_grad_linear
    elif climb_shape == SEG_CURVE_INTENSITY_KEPT_STATIC:
        return get_intensity_kept_val_monotonic_static
    else:
        return get_stable_val_monotonic_grad_stable


def get_bulge_val(prev_val, line_val, target_val, upper_bound_ratio=1, lower_bound_ratio=0):
    lower_bound = prev_val*(1+lower_bound_ratio)
    upper_bound = line_val*upper_bound_ratio
    bulge_val = generate_ranged_float_rand(lower_bound, upper_bound)
    return bulge_val


def get_linear_val(prev_val, line_val, target_val, upper_bound_ratio=1, lower_bound_ratio=1):
    return line_val


def get_static_val(prev_val, line_val, target_val, upper_bound_ratio=1, lower_bound_ratio=1):
    return SEG_CURVE_VAL_STATIC


def get_val_func(climb_shape):
    if climb_shape == SEG_CURVE_CONCAVE_DIVERSION:
        return get_concave_val
    elif climb_shape == SEG_CURVE_BULGE_DIVERSION:
        return get_bulge_val
    elif climb_shape == SEG_CURVE_LINEAR:
        return get_linear_val
    else:
        return get_static_val


def get_climb_end_points_val(initial_intensity, target_intensity, climb_shape):
    if climb_shape == SEG_CURVE_STATIC:
        end_points_val = [initial_intensity, SEG_CURVE_VAL_STATIC]
        #end_points_val = [SEG_CURVE_VAL_STATIC, SEG_CURVE_VAL_STATIC]
    elif climb_shape == SEG_CURVE_DOWNWARD_TRIANGLE_LINEAR:
        end_points_val = [initial_intensity, initial_intensity]
    elif climb_shape == SEG_CURVE_INTENSITY_KEPT_STATIC:
        end_points_val = [initial_intensity, initial_intensity]
    else:
        start_intensity = np.fmin(target_intensity, initial_intensity)
        end_intensity = np.fmax(target_intensity, initial_intensity)
        end_points_val = [start_intensity, end_intensity]
    return end_points_val


def generate_linear_diverted_random_climb_line_seg_monotonic(initial_intensity, intensity_climb_ratio, climb_range, climb_shape, scan_direction, intensity_upper_bound=SEG_CURVE_INTENSITY_UPPER_BOUND, index_shrink_ratio=SEG_CURVE_SHRINK_RATIO_DEFAULT):
    target_intensity = np.fmax(np.fmin(initial_intensity*(1+intensity_climb_ratio), intensity_upper_bound), SEG_INTENSITY_LOWER_BOUND)
    abs_intensity_delta = np.abs(initial_intensity - target_intensity)
    end_intensity_x = climb_range - 1
    intensity_tan = abs_intensity_delta / (end_intensity_x * index_shrink_ratio)
    end_points_val = get_climb_end_points_val(initial_intensity, target_intensity, climb_shape)
    start_intensity = end_points_val[0]#np.fmin(target_intensity, initial_intensity)
    end_intensity = end_points_val[1]#np.fmax(target_intensity, initial_intensity)
    shrinked_end_intensity_x = end_intensity_x*index_shrink_ratio
    climb_shape_val_fun = get_val_grad_monotonic_func(climb_shape)
    index_adjust_param = get_climb_index_adjust_param(initial_intensity, end_intensity, start_intensity, scan_direction, climb_range)
    index_adjust_pref = index_adjust_param[0]
    index_adjust_cof = index_adjust_param[1]
    climb_seg = np.ones(climb_range)
    i = 0
    scan_intensity = start_intensity
    seg_index = get_adjusted_climb_index(index_adjust_pref, index_adjust_cof, i)
    climb_seg[seg_index] = scan_intensity
    i += 1
    #if start_intensity == initial_intensity:
        #gen_round_num = climb_range
    #else:
    gen_round_num = end_intensity_x
    scan_intensity_prev_1 = start_intensity
    scan_intensity_prev_2 = start_intensity
    while i < gen_round_num:
        scan_intensity = climb_shape_val_fun(scan_intensity_prev_1, scan_intensity_prev_2, intensity_tan, i, start_intensity, end_intensity, shrinked_end_intensity_x, index_shrink_ratio)
        scan_intensity_prev_1 = scan_intensity_prev_2
        scan_intensity_prev_2 = scan_intensity
        seg_index = get_adjusted_climb_index(index_adjust_pref, index_adjust_cof, i)
        climb_seg[seg_index] = np.fmin(scan_intensity, intensity_upper_bound)
        i += 1
    #if end_intensity == initial_intensity:
    seg_index = get_adjusted_climb_index(index_adjust_pref, index_adjust_cof, i)# index_adjust_pref + index_adjust_cof * i
    climb_seg[seg_index] = end_intensity
    return climb_seg


def get_rectangle_relative_rotation_coordinate(rect_len, rect_width, rotation_theta):
    rect_coordinate = np.zeros((rect_width, rect_len), dtype=complex)
    center_coordinate_x = np.int_(rect_len/2)-1
    center_coordinate_y = np.int_(rect_width/2)-1
    center_coordinate = complex(center_coordinate_x, center_coordinate_y)
    i = 0
    while i < rect_width:
        k = 0
        while k < rect_len:
            y = i - center_coordinate_y
            x = k - center_coordinate_x
            rect_coordinate[i, k] = complex(x, y)
            k += 1
        i += 1
    rotation_vector = complex(np.cos(rotation_theta), np.sin(rotation_theta))
    rect_coordinate = rect_coordinate * rotation_vector + center_coordinate
    #print(rect_coordinate[center_coordinate_y, center_coordinate_x])
    return rect_coordinate


def get_rectangle_absolute_rotation_coordinate(rect_top_lef_coordinate, rect_len, rect_width, rotation_theta, bias=[0, 0]):
    rect_relative_coordinate = get_rectangle_relative_rotation_coordinate(rect_len, rect_width, rotation_theta)
    original_coordinate = complex(rect_top_lef_coordinate[0], rect_top_lef_coordinate[1])
    bias_coordinate = complex(bias[0], bias[1])
    rect_absolute_coordinate = rect_relative_coordinate + original_coordinate + bias_coordinate
    return rect_absolute_coordinate


def img_gray_coordinate_matrix_rect_copy(img_gray, rect_val_matrix, coordinate_matrix):
    coordinate_matrix_shape = coordinate_matrix.shape
    matrix_row_num = coordinate_matrix_shape[0]
    matrix_col_num = coordinate_matrix_shape[1]
    i = 0
    while i < matrix_row_num:
        j = 0
        while j < matrix_col_num:
            x_y = coordinate_matrix[i, j]
            x = np.int_(np.round(np.real(x_y)))
            y = np.int_(np.round(np.imag(x_y)))
            img_gray[y, x] = rect_val_matrix[i, j]
            j += 1
        i += 1
    return img_gray


def img_gray_rotate_rect_region(img_gray, rect_top_lef_coordinate, rect_bottom_right_coordinate, rotation_theta, bias=[0, 0], is_img_copy=SEG_IMG_COPY):
    img_rect_rotated = get_img_instance_copy_choice(img_gray, is_img_copy)
    rect_top_left_x = rect_top_lef_coordinate[0]
    rect_top_left_y = rect_top_lef_coordinate[1]
    rect_width = rect_bottom_right_coordinate[1] - rect_top_left_y + 1
    rect_len = rect_bottom_right_coordinate[0] - rect_top_left_x + 1
    rect_absolute_coordinate = get_rectangle_absolute_rotation_coordinate(rect_top_lef_coordinate, rect_len, rect_width, rotation_theta, bias)
    rect_val = img_rect_rotated[rect_top_left_y:rect_top_left_y+rect_width, rect_top_left_x:rect_top_left_x+rect_len]
    img_rect_rotated = img_gray_coordinate_matrix_rect_copy(img_rect_rotated, rect_val, rect_absolute_coordinate)
    return img_rect_rotated


def img_gray_circle_single_pixel(img_gray, center_point, radius, intensity):
    theta = 0
    delta_theta = SEG_CIRCLE_DELTA_THETA
    while theta <= 2*np.pi:
        x = np.int_(np.round(radius*np.cos(theta) + center_point[0]))
        y = np.int_(np.round(radius*np.sin(theta) + center_point[1]))
        if x>=0 and y>=0:
            img_gray[y, x] = intensity
        theta += delta_theta
    return img_gray


def img_gray_circle(img_gray, center_point, max_radius, intensity, thickness):
    i = 0
    img_gray_with_circle = img_gray
    while i < thickness:
        img_gray_with_circle = img_gray_circle_single_pixel(img_gray_with_circle, center_point, max_radius-i, intensity)
        i += 1
    return img_gray_with_circle


def get_img_gray_circle_seg_mark_center_point(seg_start, seg_direction, center_point_ref):
    if seg_direction == SEG_DIRECTION_ROW:
        center_point = [center_point_ref, seg_start]
    else:
        center_point = [seg_start, center_point_ref]
    return center_point


def img_gray_circle_mark_seg_area(img_gray, seg_start, seg_direction, center_point_list, radius, intensity, thickness, is_img_copy=SEG_IMG_COPY):
    img_gray_circle_mark = get_img_instance_copy_choice(img_gray, is_img_copy)
    point_list_len = len(center_point_list)
    i = 0
    while i < point_list_len:
        center_point = get_img_gray_circle_seg_mark_center_point(seg_start, seg_direction, center_point_list[i])
        img_gray_circle_mark = img_gray_circle(img_gray_circle_mark, center_point, radius, intensity, thickness)
        i += 1
    return img_gray_circle_mark


def get_img_gray_poisson_seg_fragmented_overlap(img_gray, fragment_shape, stride_row=1, stride_col=1, intensity_map=None, intensity_pixel_num_map=None):
    img_shape = img_gray.shape
    img_row_num = img_shape[0]
    img_col_num = img_shape[1]
    fragment_row_num = fragment_shape[0]
    fragment_col_num = fragment_shape[1]
    img_poisson_seg_result = np.zeros(img_shape)
    row_wise_fragment_num = img_row_num - fragment_row_num + 1
    col_wise_fragment_num = img_col_num - fragment_col_num + 1
    row_idx = 0
    i = 0
    while i < row_wise_fragment_num:
        poisson_seg_fragment_row_info = img_gray_poison_seg_fragmented_overlap_single_row(img_gray, img_poisson_seg_result, row_idx, col_wise_fragment_num, stride_col, fragment_col_num, fragment_row_num)
        if intensity_map is not None:
            intensity_map.append(poisson_seg_fragment_row_info[1])
        if intensity_pixel_num_map is not None:
            intensity_pixel_num_map.append(poisson_seg_fragment_row_info[2])
        i += stride_row
        row_idx += stride_row
    if i != row_wise_fragment_num - 1 + stride_row:
        row_idx = row_wise_fragment_num - 1
        poisson_seg_fragment_row_info = img_gray_poison_seg_fragmented_overlap_single_row(img_gray, img_poisson_seg_result, row_idx, col_wise_fragment_num, stride_col, fragment_col_num, fragment_row_num)
        if intensity_map is not None:
            intensity_map.append(poisson_seg_fragment_row_info[1])
        if intensity_pixel_num_map is not None:
            intensity_pixel_num_map.append(poisson_seg_fragment_row_info[2])
    return img_poisson_seg_result


def img_gray_poison_seg_fragmented_overlap_single_row(img_gray, img_poisson_seg_result, row_idx, col_wise_fragment_num, stride_col, fragment_col_num, fragment_row_num):
    row_wise_intensity_map = []
    row_wise_intensity_pixel_num_map = []
    fragment_row_end = row_idx + fragment_row_num
    img_gray_poisson_seg_fragmented_overlap_col_scan(img_gray, img_poisson_seg_result, row_idx, fragment_row_end, col_wise_fragment_num, stride_col, fragment_col_num, row_wise_intensity_map, row_wise_intensity_pixel_num_map)
    return [img_poisson_seg_result, row_wise_intensity_map, row_wise_intensity_pixel_num_map]


def img_gray_poisson_seg_fragmented_overlap_col_scan(img_gray, img_poisson_seg_result, row_idx, fragment_row_end, col_wise_fragment_num, stride_col, fragment_col_num, row_wise_intensity_map, row_wise_intensity_pixel_num_map):
    col_idx = 0
    j = 0
    while j < col_wise_fragment_num:
        fragment_col_end = col_idx + fragment_col_num
        integrated_get_single_fragment_poisson_seg(img_gray, img_poisson_seg_result, row_idx, fragment_row_end, col_idx, fragment_col_end, row_wise_intensity_map, row_wise_intensity_pixel_num_map)
        j += stride_col
        col_idx += stride_col
    if j != col_wise_fragment_num - 1 + stride_col:
        col_idx = col_wise_fragment_num - 1
        fragment_col_end = col_idx + fragment_col_num
        integrated_get_single_fragment_poisson_seg(img_gray, img_poisson_seg_result, row_idx, fragment_row_end, col_idx, fragment_col_end, row_wise_intensity_map, row_wise_intensity_pixel_num_map)
    return [img_poisson_seg_result]


def integrated_get_single_fragment_poisson_seg(img_gray, img_poisson_seg_result, row_idx, fragment_row_end, col_idx, fragment_col_end, row_wise_intensity_map, row_wise_intensity_pixel_num_map):
    col_wise_intensity_map = []
    col_wise_intensity_pixel_num_map = []
    img_fragment = img_gray[row_idx:fragment_row_end, col_idx:fragment_col_end]
    get_single_fragment_poisson_seg(img_fragment, img_poisson_seg_result, row_idx, fragment_row_end, col_idx, fragment_col_end, col_wise_intensity_map, col_wise_intensity_pixel_num_map)
    row_wise_intensity_map.append(col_wise_intensity_map)
    row_wise_intensity_pixel_num_map.append(col_wise_intensity_pixel_num_map)
    return [img_poisson_seg_result]


def get_img_gray_seg_poisson_seg_fragmented(img_gray, fragment_shape, intensity_map=None, intensity_pixel_num_map=None):
    img_shape = img_gray.shape
    img_row_num = img_shape[0]
    img_col_num = img_shape[1]
    fragment_row_num = fragment_shape[0]
    fragment_col_num = fragment_shape[1]
    row_wise_complete_fragment_num = np.floor(img_row_num/fragment_row_num)
    col_wise_complete_fragment_num = np.floor(img_col_num/fragment_col_num)
    row_wise_residual_fragment_num = img_row_num % fragment_row_num
    col_wise_residual_fragment_num = img_col_num % fragment_col_num
    img_poisson_seg_result = np.zeros(img_shape)
    total_col_wise_fragment_num = col_wise_complete_fragment_num + 1
    i = 0
    row_idx = 0
    while i < row_wise_complete_fragment_num:
        fragment_scan_info = img_gray_seg_poisson_fragmented_row_scan(img_gray, img_poisson_seg_result, row_idx, fragment_row_num, fragment_col_num, col_wise_residual_fragment_num, total_col_wise_fragment_num)
        fill_poisson_seg_info_map(intensity_map, intensity_pixel_num_map, fragment_scan_info[1], fragment_scan_info[2])
        i += 1
        row_idx += fragment_row_num
    fragment_scan_info = img_gray_seg_poisson_fragmented_row_scan(img_gray, img_poisson_seg_result, row_idx, row_wise_residual_fragment_num, fragment_col_num, col_wise_residual_fragment_num, total_col_wise_fragment_num)
    fill_poisson_seg_info_map(intensity_map, intensity_pixel_num_map, fragment_scan_info[1], fragment_scan_info[2])
    return img_poisson_seg_result


def img_gray_seg_poisson_fragmented_row_scan(img_gray, img_poisson_seg_result, row_idx, row_fragment_num, col_complete_fragment_num, col_residual_fragment_num, total_fragment_num):
    i = 0
    j = 0
    complete_col_fragment_num = total_fragment_num - 1
    row_fragment_end = row_idx + row_fragment_num
    fragment_intensity_map_list = []
    fragment_intensity_pixel_num_map_list = []
    while i < complete_col_fragment_num:
        fragment_intensity_map = []
        fragment_intensity_pixel_num_map = []
        col_fragment_end = j + col_complete_fragment_num
        img_fragment = img_gray[row_idx:row_fragment_end, j:col_fragment_end]
        get_single_fragment_poisson_seg(img_fragment, img_poisson_seg_result, row_idx, row_fragment_end, j, col_fragment_end, fragment_intensity_map, fragment_intensity_pixel_num_map)
        j = col_fragment_end
        fragment_intensity_map_list.append(fragment_intensity_map)
        fragment_intensity_pixel_num_map_list.append(fragment_intensity_pixel_num_map)
        i += 1
    col_fragment_end = j + col_residual_fragment_num
    img_fragment = img_gray[row_idx:row_fragment_end, j:col_fragment_end]
    fragment_intensity_map = []
    fragment_intensity_pixel_num_map = []
    get_single_fragment_poisson_seg(img_fragment, img_poisson_seg_result, row_idx, row_fragment_end, j, col_fragment_end, fragment_intensity_map, fragment_intensity_pixel_num_map)
    if len(fragment_intensity_map) > 0:
        fragment_intensity_map_list.append(fragment_intensity_map)
        fragment_intensity_pixel_num_map_list.append(fragment_intensity_pixel_num_map)
    return [img_poisson_seg_result, fragment_intensity_map_list, fragment_intensity_pixel_num_map_list]


def get_single_fragment_poisson_seg(img_fragment, img_poisson_seg_result, fragment_row_start, fragment_row_end, fragment_col_start, fragment_col_end, fragment_intensity_map=None, fragment_intensity_pixel_num_map=None):
    img_fragment_size = img_fragment.size
    if img_fragment_size > 1:
        if FRAGMENT_FILTER_ALGORITHM == FRAGMENT_FILTER_ALGORITHM_POISSON:
            img_fragmented_poisson_seg = get_img_gray_poisson_seg(img_fragment, fragment_intensity_map, fragment_intensity_pixel_num_map, SEG_IMG_NO_COPY)
        elif FRAGMENT_FILTER_ALGORITHM == FRAGMENT_FILTER_ALGORITHM_STEP:
            img_fragmented_poisson_seg = get_img_gray_intensity_step_blur(img_fragment)
        elif FRAGMENT_FILTER_ALGORITHM == FRAGMENT_FILTER_ALGORITHM_NORMALIZED_STD:
            img_fragmented_poisson_seg = get_img_gray_seg_normalized_std(img_fragment) + img_poisson_seg_result[fragment_row_start:fragment_row_end, fragment_col_start:fragment_col_end]
        elif FRAGMENT_FILTER_ALGORITHM == FRAGMENT_FILTER_ALGORITHM_DISTANCE_METRIC_AVG_STD:
            img_fragmented_poisson_seg = get_img_gray_seg_dist_avg_metric(img_fragment) + img_poisson_seg_result[fragment_row_start:fragment_row_end, fragment_col_start:fragment_col_end]
        else:
            img_fragmented_poisson_seg = get_img_gray_binary_one_zero_ratio_metric(img_fragment) + img_poisson_seg_result[fragment_row_start:fragment_row_end, fragment_col_start:fragment_col_end]
        img_poisson_seg_result[fragment_row_start:fragment_row_end, fragment_col_start:fragment_col_end] = img_fragmented_poisson_seg
    elif img_fragment_size == 1:
        img_poisson_seg_result[fragment_row_start:fragment_row_end, fragment_col_start:fragment_col_end] = img_fragment
    return [img_poisson_seg_result]


def fill_poisson_seg_info_map(intensity_map, intensity_pixel_num_map, intensity_info, intensity_pixel_num_info):
    if intensity_map is not None:
        intensity_map.append(intensity_info)
    if intensity_pixel_num_map is not None:
        intensity_pixel_num_map.append(intensity_pixel_num_info)
    return [intensity_info, intensity_pixel_num_info]


def get_img_gray_window_poisson_seg_enhancement(img_gray, img_gray_blur, img_window_row_start, img_window_row_end, img_window_col_start, img_window_col_end, is_copy=SEG_IMG_NO_COPY):
    if is_copy:
        img_gray_blur_cp = img_copy(img_gray_blur)
    else:
        img_gray_blur_cp = img_gray_blur
    window_seg_col_end = img_window_col_end + 1
    window_seg_row_end = img_window_row_end + 1
    img_window_seg = img_gray[img_window_row_start:window_seg_row_end, img_window_col_start:window_seg_col_end]
    img_window_poisson_seg = get_img_gray_poisson_seg(img_window_seg)
    img_gray_blur_cp[img_window_row_start:window_seg_row_end, img_window_col_start:window_seg_col_end] = img_window_poisson_seg
    return [img_gray_blur_cp, img_window_poisson_seg]


def get_img_gray_poisson_seg(img_gray, intensity_map=None, intensity_pixel_num_map=None, is_copy=SEG_IMG_COPY, avg_distance_additional_measurements=None, is_torch=False, device=torch.device('cpu'), intensity_plane=None, plane_pixel_num_list=None):
    s_time = tm.time()
    if is_copy:
        img_cp = img_copy(img_gray)
    else:
        img_cp = img_gray
    img_shape = img_cp.shape
    img_size = img_shape[0]*img_shape[1]
    img_cp = img_cp.reshape(img_size)
    avg_distance_info = get_img_gray_average_intensity_distance(img_cp)
    #e_time = tm.time()
    avg_distance = avg_distance_info[0]
    img_sorted_array = avg_distance_info[1]
    img_nonzero_len = avg_distance_info[3]
    #distance_seg_num = get_img_gray_distance_seg_num(img_sorted_array, avg_distance)
    #s_time = tm.time()
    if is_torch:
        img_gray_distance_seg_mapped = get_img_gray_distance_seg_mapped_torch(img_gray, avg_distance, img_nonzero_len, intensity_map, intensity_pixel_num_map, device, intensity_plane)
    else:
        #s_time = tm.time()
        img_gray_distance_seg_mapped = get_img_gray_distance_seg_mapped_np_where(img_gray, avg_distance, img_nonzero_len, intensity_map, intensity_pixel_num_map, intensity_plane, plane_pixel_num_list=plane_pixel_num_list)
    e_time = tm.time()
    #e_time = tm.time()
    print("seg total in ", e_time - s_time)
    if avg_distance_additional_measurements is not None:
        ex_time = e_time - s_time
        avg_distance_info.append(ex_time)
        fill_seg_avg_distance_additional_measurements(avg_distance_info, avg_distance_additional_measurements)
    return img_gray_distance_seg_mapped


def fill_seg_avg_distance_additional_measurements(avg_distance_info, avg_distance_additional_measurements):
    time_idx = len(avg_distance_info)-1
    avg_distance_additional_measurements.append(avg_distance_info[0])
    avg_distance_additional_measurements.append(avg_distance_info[5])
    avg_distance_additional_measurements.append(mat.sqrt(avg_distance_info[3]))
    avg_distance_additional_measurements.append(avg_distance_info[6])
    avg_distance_additional_measurements.append(avg_distance_info[7])
    avg_distance_additional_measurements.append(avg_distance_info[time_idx])
    avg_distance_additional_measurements.append(avg_distance_info[2])
    return avg_distance_additional_measurements


def get_img_gray_avg_distance_std(img_gray):
    img_cp = img_copy(img_gray)
    img_shape = img_cp.shape
    img_size = img_shape[0] * img_shape[1]
    img_cp = img_cp.reshape(img_size)
    avg_distance_info = get_img_gray_average_intensity_distance(img_cp)
    avg_distance_additional_measurements = []
    fill_seg_avg_distance_additional_measurements(avg_distance_info, avg_distance_additional_measurements)
    return avg_distance_additional_measurements


def get_img_gray_average_intensity_distance(img_array):
    #img_sorted_array = img_array#np.sort(img_array)
    #test
    #img_array =filter_low_resolution_distance_array(img_array)
    #test end

    #img_sorted_array = np.sort(img_array[img_array != 0])
    img_sorted_array = np.sort(img_array)

    #img_sorted_array_cp = img_copy(img_sorted_array)
    array_len = len(img_sorted_array)
    if array_len > 1:
        #test
        #img_sorted_array = filter_low_resolution_distance_array(img_sorted_array)
        #test end
        img_array_0 = img_sorted_array[0:array_len-1]
        img_array_1 = img_sorted_array[1:array_len]
        img_array_sub = img_array_1 - img_array_0
        #test
        #img_array_sub = filter_low_resolution_distance_array(img_array_sub)
        #test end
        avg_distance = np.average(img_array_sub)
        std_distance_info = cal_pixel_gray_level_coherent_std(img_array_sub, avg_distance)#np.std(img_array_sub)
        std_distance = std_distance_info[0]
        img_division_info = std_distance_info[1]
        none_zero_distance_array = std_distance_info[2]
        if array_len > 0:
            max_distance = np.max(img_array_sub)
        else:
            max_distance = 0
        if np.isnan(avg_distance):
            avg_distance = 0
    else:
        avg_distance = 0
        img_sorted_array = None
        img_array_sub = None
        max_distance = 0
        std_distance = 0
    return [avg_distance, img_sorted_array, img_array_sub, array_len, max_distance, std_distance, img_division_info, none_zero_distance_array]


def filter_low_resolution_distance_array(distance_array):
    filter_mask = np.ceil((np.sign(distance_array - DISTANCE_MIN_RESOLUTION_CONST) + 1)/2)
    filtered_distance_array = distance_array*filter_mask
    return filtered_distance_array


def cal_pixel_gray_level_coherent_std(distance_array, avg_distance):
    nonzero_idx = np.flatnonzero(distance_array)
    none_zero_distance_array = distance_array[nonzero_idx]
    none_zero_distance_array_std = none_zero_distance_array
    none_zero_distance_array_len = len(none_zero_distance_array_std)
    distance_array_len = len(distance_array)
    group_num_info = cal_pixel_group_std(nonzero_idx)
    group_num_std = group_num_info[0]
    group_num_list = group_num_info[1]
    distance_array_coefficient_info = cal_distance_array_coefficient(group_num_list, distance_array_len)
    distance_array_coefficient = distance_array_coefficient_info[0]
    #none_zero_distance_avg = np.average(none_zero_distance_array_std*distance_array_coefficient)
    none_zero_distance_sum = np.sum(none_zero_distance_array_std)
    none_zero_distance_array_std = none_zero_distance_array_std/none_zero_distance_sum
    none_zero_distance_avg = np.sum(none_zero_distance_array_std * distance_array_coefficient)
    none_zero_distance_std_avg = np.average(none_zero_distance_array_std)
    #none_zero_distance_std = 2*np.std(none_zero_distance_array_std)/(np.abs(np.max(none_zero_distance_array_std)-none_zero_distance_std_avg)+np.abs(np.min(none_zero_distance_array_std)-none_zero_distance_std_avg))
    #none_zero_distance_std = none_zero_distance_avg*none_zero_distance_std #none_zero_distance_avg*(1-none_zero_distance_std)#group_num_std*none_zero_distance_avg#group_num_std*(distance_array_len/none_zero_distance_array_len)*np.std(none_zero_distance_array_std)#mat.sqrt(np.sum(np.power(none_zero_distance_array_std - avg_distance, 2))/none_zero_distance_len)
    #new calculator
    complete_group_num_list = distance_array_coefficient_info[1]
    group_array_division_info = cal_array_division_info(complete_group_num_list)
    img_division_info = none_zero_distance_array_std*group_array_division_info[4]
    none_zero_distance_std = np.sum(img_division_info)
    #new calculator end
    return [none_zero_distance_std, img_division_info, none_zero_distance_array]


def cal_distance_array_coefficient(group_num_list, distance_array_len):
    total_img_array_len = distance_array_len + 1
    last_group_num = total_img_array_len - np.sum(group_num_list)
    complete_group_num_list = np.append(group_num_list, last_group_num)
    complete_group_num_list_len = len(complete_group_num_list)
    #distance_array_coefficient = (complete_group_num_list[0:complete_group_num_list_len-1]+complete_group_num_list[1:complete_group_num_list_len])/total_img_array_len
    complete_group_num_list_effect = complete_group_num_list[1:complete_group_num_list_len]
    distance_array_coefficient = complete_group_num_list_effect/ sum(complete_group_num_list_effect)
    #distance_array_coefficient = np.sqrt(1-np.power(distance_array_coefficient, 2))#1-complete_group_num_list[1:complete_group_num_list_len]/total_img_array_len
    return [distance_array_coefficient, complete_group_num_list]


def cal_array_division_info(group_num_list):
    group_num_list_len = len(group_num_list)
    division_list_len = group_num_list_len - 1
    division_array_left = [None]*division_list_len
    division_array_right = [None]*division_list_len
    group_num_list_len += 1
    i = 1
    j = 0
    while i <= division_list_len:
        division_array_left[j] = np.sum(group_num_list[0:i])
        division_array_right[j] = np.sum(group_num_list[i:group_num_list_len])
        i += 1
        j += 1
    division_array_left = np.array(division_array_left)
    division_array_right = np.array(division_array_right)
    division_array_max = np.fmax(division_array_left, division_array_right)
    division_array_min = np.fmin(division_array_left, division_array_right)
    division_array_min_max_ratio = division_array_min/division_array_max
    return [division_array_left, division_array_right, division_array_max, division_array_min, division_array_min_max_ratio]


def cal_pixel_group_std(nonzero_idx):
    group_index_high = nonzero_idx + 1
    group_index_len = len(nonzero_idx)
    group_index_low = group_index_high[0:group_index_len-1]
    group_index_low = np.insert(group_index_low, 0, 0)
    group_index_high = group_index_high - group_index_low
    group_num_std = np.std(group_index_high)/np.average(group_index_high)
    return [group_num_std, group_index_high]


def get_img_gray_distance_seg_num(img_sorted_array, avg_distance):
    img_inv_sorted_array = img_sorted_array[::-1]
    array_len = len(img_inv_sorted_array)
    img_array_mask = (np.ones(array_len)).transpose()
    img_array_seg_erased = img_inv_sorted_array
    img_max_idx = 0
    seg_num = 0
    seg_combine_distance = get_seg_combine_distance(array_len, avg_distance)
    while img_max_idx < array_len:
        seg_num += 1
        img_min_val = img_array_seg_erased[img_max_idx] - seg_combine_distance
        img_array_seg_sign_val = img_array_seg_erased - img_min_val
        img_array_seg_sign_val = np.sign(img_array_seg_sign_val)
        img_array_seg_sign_val = np.ceil((img_array_seg_sign_val + 1)/2)
        img_max_idx += np.int_(img_array_seg_sign_val @ img_array_mask)
        img_array_seg_sign_val = img_array_seg_sign_val * img_array_seg_erased
        img_array_seg_erased = img_array_seg_erased - img_array_seg_sign_val
    return seg_num


def img_poisson_seg_similarity_metric(avg_dist_metric_1, avg_dist_metric_2):
    img_poisson_seg_similarity_metric_val = 2 * np.fmin(avg_dist_metric_1, avg_dist_metric_2) / (avg_dist_metric_1 + avg_dist_metric_2)
    return img_poisson_seg_similarity_metric_val


def img_gray_contrast_metric(avg_dist_metric):
    avg_dist_metric_average = np.average(avg_dist_metric)
    #avg_dist_metric_std = np.std(avg_dist_metric)
    avg_dist_abs_deviation_std_residual = np.abs(avg_dist_metric - avg_dist_metric_average)
    avg_dist_metric_std = np.average(avg_dist_abs_deviation_std_residual)
    avg_dist_abs_deviation_std_residual = np.abs(avg_dist_abs_deviation_std_residual - avg_dist_metric_std)
    avg_dist_abs_deviation_std_residual_max = np.max(avg_dist_abs_deviation_std_residual)
    avg_dist_abs_deviation_std_residual_min = np.min(avg_dist_abs_deviation_std_residual)
    avg_dist_abs_deviation_std_residual_medium = np.median(avg_dist_abs_deviation_std_residual)
    #avg_dist_abs_deviation_std_residual_max_medium_sum = avg_dist_abs_deviation_std_residual_max + avg_dist_abs_deviation_std_residual_medium
    avg_dist_abs_deviation_std_residual_sum = avg_dist_abs_deviation_std_residual_max + avg_dist_abs_deviation_std_residual_medium/avg_dist_abs_deviation_std_residual_min#avg_dist_abs_deviation_std_residual_max_medium_sum
    if avg_dist_abs_deviation_std_residual_sum > 0:
        img_gray_contrast_metric_val = avg_dist_abs_deviation_std_residual_max/(avg_dist_abs_deviation_std_residual_sum)
    else:
        img_gray_contrast_metric_val = 1/2
    #img_gray_contrast_metric_val = 2*img_gray_contrast_metric_val - 1#2*(img_gray_contrast_metric_val - 1/2)
    return [img_gray_contrast_metric_val, avg_dist_abs_deviation_std_residual_max, avg_dist_abs_deviation_std_residual_min]


def img_gray_contrast_metric_1(avg_dist_metric):
    avg_dist_hist, avg_dist_edge = np.histogram(avg_dist_metric)
    avg_dist_hist_len = len(avg_dist_hist)
    avg_dist_hist_residual_sum = np.sum(avg_dist_hist[1:avg_dist_hist_len-1])
    avg_dist_hist_sum = avg_dist_hist_residual_sum + avg_dist_hist[0]
    if avg_dist_hist_residual_sum != 0:
        img_gray_contrast_metric_val = avg_dist_hist_residual_sum/avg_dist_hist_sum
    else:
        img_gray_contrast_metric_val = 1
    return [img_gray_contrast_metric_val]


def get_img_gray_distance_seg_mapped(img_gray, avg_distance, img_nonzero_len, intensity_map=None, intensity_pixel_num_map=None, intensity_plane=None):
    img_shape = img_gray.shape
    img_gray_erased = img_copy(img_gray)
    img_distance_seg_mapped = np.zeros(img_shape)
    img_size = img_shape[0]*img_shape[1]
    img_array_mask = np.ones(img_size).transpose()
    seg_combine_distance = get_seg_combine_distance(img_nonzero_len, avg_distance)#get_seg_combine_distance(img_size, avg_distance)
    seg_map_distance = get_seg_map_distance()
    img_left = img_size
    seg_map_val = DISTANCE_SEG_MAX_INTENSITY
    seg_num = 0
    img_max = np.max(img_gray_erased)
    seg_distance_step = 1

    seg_num_max = DISTANCE_SEG_NUM - 1

    while seg_num < DISTANCE_SEG_NUM and img_left > 0 and img_max > 0:
        img_min_val = img_max - seg_combine_distance

        s_time_seg_one_epoch = tm.time()
        img_temp = np.where(img_gray_erased >= img_min_val, 1, 0)
        e_time_seg_one_epoch = tm.time()
        print("seg_where in ", e_time_seg_one_epoch - s_time_seg_one_epoch)
        if seg_num == seg_num_max or img_min_val <= 0:
            img_min_val = 0#img_max
        seg_num += seg_distance_step
        img_gray_seg_sign_val = img_gray_erased - img_min_val
        #s_time_seg_one_epoch = tm.time()
        img_gray_seg_sign_val = np.sign(img_gray_seg_sign_val)
        #e_time_seg_one_epoch = tm.time()
        #print("seg_one_epoch in ", e_time_seg_one_epoch - s_time_seg_one_epoch)
        #s_time_seg_one_epoch = tm.time()
        img_gray_seg_sign_val = np.ceil((img_gray_seg_sign_val + 1)/2)
        if intensity_plane is not None:
            intensity_plane.append(img_gray_seg_sign_val)
        #e_time_seg_one_epoch = tm.time()
        #print("seg_one_epoch in ", e_time_seg_one_epoch - s_time_seg_one_epoch)
        img_distance_seg_mapped = img_distance_seg_mapped + img_gray_seg_sign_val*seg_map_val
        img_gray_seg_sign_val = img_gray_seg_sign_val*img_gray_erased
        img_gray_erased = img_gray_erased - img_gray_seg_sign_val
        img_gray_erased_array = img_gray_erased.reshape(img_size)
        img_left_temp = np.abs(np.sign(img_gray_erased_array)) @ img_array_mask
        if intensity_pixel_num_map is not None:
            intensity_pixel_num_map.append(img_left - img_left_temp)
        img_left = img_left_temp
        img_max = np.max(img_gray_erased)
        #if np.max(img_distance_seg_mapped) > 1:
            #img_max = 0
        seg_distance_step = get_seg_distance_step(img_max, img_min_val, seg_combine_distance, seg_num)
        if intensity_map is not None:
            intensity_map.append(seg_map_val)
        seg_map_val = seg_map_val - seg_map_distance*np.float_(seg_distance_step)
        #if seg_num == 48:
            #img_max = np.max(img_gray_erased)
        #e_time_seg_one_epoch = tm.time()
        #print("seg_one_epoch in ", e_time_seg_one_epoch-s_time_seg_one_epoch)
    return img_distance_seg_mapped


def get_img_gray_distance_seg_mapped_torch(img_gray, avg_distance, img_nonzero_len, intensity_map=None, intensity_pixel_num_map=None, device=torch.device('cpu'), intensity_plane=None):
    s_time_seg_one_epoch = tm.time()
    img_shape = img_gray.shape
    img_gray_erased = img_copy(img_gray)
    img_gray_erased = torch.from_numpy(img_gray_erased).to(device)
    img_distance_seg_mapped = torch.zeros(img_shape).double().to(device)
    img_size = img_shape[0]*img_shape[1]

    img_array_mask = torch.ones(img_size).transpose(-1, 0).double().to(device)

    seg_combine_distance = torch.as_tensor(get_seg_combine_distance(img_nonzero_len, avg_distance)).double().to(device)#get_seg_combine_distance(img_size, avg_distance))
    seg_map_distance = torch.as_tensor(get_seg_map_distance()).double().to(device)
    img_left = img_size
    seg_map_val = torch.as_tensor(DISTANCE_SEG_MAX_INTENSITY).double().to(device)
    seg_num = 0
    img_max = torch.max(img_gray_erased)
    seg_distance_step = 1

    seg_num_max = DISTANCE_SEG_NUM - 1
    e_time_seg_one_epoch = tm.time()
    print("initial_epoch in ", e_time_seg_one_epoch - s_time_seg_one_epoch)

    while seg_num < DISTANCE_SEG_NUM and img_left > 0 and img_max > 0:
        img_min_val = img_max - seg_combine_distance
        if seg_num == seg_num_max or img_min_val <= 0:
            img_min_val = torch.as_tensor(0).to(device)#img_max
        seg_num += seg_distance_step
        img_gray_seg_sign_val = img_gray_erased - img_min_val
        #s_time_seg_one_epoch = tm.time()
        img_gray_seg_sign_val = torch.sign(img_gray_seg_sign_val)
        #e_time_seg_one_epoch = tm.time()
        #print("seg_one_epoch in ", e_time_seg_one_epoch - s_time_seg_one_epoch)
        #s_time_seg_one_epoch = tm.time()
        img_gray_seg_sign_val = torch.ceil((img_gray_seg_sign_val + 1)/2)
        if intensity_plane is not None:
            intensity_plane.append(img_gray_seg_sign_val.cpu().int().numpy())
        #e_time_seg_one_epoch = tm.time()
        #print("seg_one_epoch in ", e_time_seg_one_epoch - s_time_seg_one_epoch)
        s_time_seg_one_epoch = tm.time()
        img_distance_seg_mapped = img_distance_seg_mapped + img_gray_seg_sign_val*seg_map_val
        e_time_seg_one_epoch = tm.time()
        print("img_map_gen in ", e_time_seg_one_epoch-s_time_seg_one_epoch)
        img_gray_seg_sign_val = img_gray_seg_sign_val*img_gray_erased
        img_gray_erased = img_gray_erased - img_gray_seg_sign_val
        img_gray_erased_array = img_gray_erased.reshape(img_size)
        img_left_temp = torch.abs(torch.sign(img_gray_erased_array)) @ img_array_mask
        if intensity_pixel_num_map is not None:
            intensity_pixel_num_map.append(img_left - img_left_temp)
        img_left = img_left_temp
        img_max = torch.max(img_gray_erased)
        #if np.max(img_distance_seg_mapped) > 1:
            #img_max = 0
        seg_distance_step = get_seg_distance_step_torch(img_max, img_min_val, seg_combine_distance, seg_num, device)
        if intensity_map is not None:
            intensity_map.append(seg_map_val.cpu().numpy())

        seg_map_val = seg_map_val - seg_map_distance*seg_distance_step
        #if seg_num == 48:
            #img_max = np.max(img_gray_erased)
        #e_time_seg_one_epoch = tm.time()
        #print("seg_one_epoch in ", e_time_seg_one_epoch-s_time_seg_one_epoch)
    return img_distance_seg_mapped.cpu().numpy()


def get_img_gray_distance_seg_mapped_np_where(img_gray, avg_distance, img_nonzero_len, intensity_map=None, intensity_pixel_num_map=None, intensity_plane=None, plane_pixel_num_list=None):
    img_shape = img_gray.shape
    img_gray_erased = img_copy(img_gray)
    img_distance_seg_mapped = np.zeros(img_shape)
    img_size = img_shape[0]*img_shape[1]
    img_array_mask = np.ones(img_size).transpose()
    seg_combine_distance = get_seg_combine_distance(img_nonzero_len, avg_distance)#get_seg_combine_distance(img_size, avg_distance)
    seg_map_distance = get_seg_map_distance()
    img_left = img_size
    seg_map_val = DISTANCE_SEG_MAX_INTENSITY
    seg_num = 0
    img_max = np.max(img_gray_erased)
    seg_distance_step = 1

    seg_num_max = DISTANCE_SEG_NUM - 1

    while seg_num < DISTANCE_SEG_NUM and img_left > 0 and img_max > 0:
        img_min_val = img_max - seg_combine_distance

        #s_time_seg_one_epoch = tm.time()
        img_gray_seg_sign_val = np.where(img_gray_erased >= img_min_val, 1, 0)
        #e_time_seg_one_epoch = tm.time()
        #print("seg_where in ", e_time_seg_one_epoch - s_time_seg_one_epoch)

        if seg_num == seg_num_max or img_min_val <= 0:
            img_min_val = 0#img_max
        seg_num += seg_distance_step
        if intensity_plane is not None:
            intensity_plane.append(img_gray_seg_sign_val)
        img_distance_seg_mapped = img_distance_seg_mapped + img_gray_seg_sign_val*seg_map_val
        plane_pixel_num = np.count_nonzero(img_gray_seg_sign_val)
        if plane_pixel_num_list is not None:
            plane_pixel_num_list.append(plane_pixel_num)
        img_left = img_left - plane_pixel_num #np.count_nonzero(img_gray_seg_sign_val)
        #if intensity_pixel_num_map is not None:
            #intensity_pixel_num_map.append(img_left)
        img_gray_erased = np.where(img_gray_erased < img_min_val, img_gray_erased, 0)
        img_max = np.max(img_gray_erased)
        #if np.max(img_distance_seg_mapped) > 1:
            #img_max = 0
        seg_distance_step = 1.0#get_seg_distance_step(img_max, img_min_val, seg_combine_distance, seg_num)
        #if intensity_map is not None:
            #intensity_map.append(seg_map_val)
        seg_map_val = seg_map_val - seg_map_distance*seg_distance_step
        #if seg_num == 48:
            #img_max = np.max(img_gray_erased)
        #e_time_seg_one_epoch = tm.time()
        #print("seg_one_epoch in ", e_time_seg_one_epoch-s_time_seg_one_epoch)
    return img_distance_seg_mapped


def get_seg_combine_distance(array_len, avg_distance):
    seg_combine_distance = (array_len / (DISTANCE_SEG_NUM)) * avg_distance#*0.8
    return seg_combine_distance


def get_seg_map_distance():
    seg_map_distance = (DISTANCE_SEG_MAX_INTENSITY - DISTANCE_SEG_MIN_INTENSITY)/(DISTANCE_SEG_NUM - 1)
    return seg_map_distance


def get_seg_distance_step_torch(img_max, img_max_exp, seg_combine_distance, current_seg_num, device):
    if seg_combine_distance > 0:
        seg_combine_distance_ratio = (img_max_exp - img_max)/seg_combine_distance
        if seg_combine_distance_ratio < 1:
            seg_combine_distance_ratio = torch.as_tensor(1).double().to(device)
        seg_combine_distance_ratio = torch.round(seg_combine_distance_ratio)
        seg_distance_step = torch.fmin(seg_combine_distance_ratio, torch.as_tensor(DISTANCE_SEG_NUM-current_seg_num).double().to(device))
        #seg_distance_step = seg_distance_step
    else:
        seg_distance_step = torch.as_tensor(1).double()
    return seg_distance_step


def get_seg_distance_step(img_max, img_max_exp, seg_combine_distance, current_seg_num):
    if seg_combine_distance > 0:
        seg_combine_distance_ratio = (img_max_exp - img_max)/seg_combine_distance
        if seg_combine_distance_ratio < 1:
            seg_combine_distance_ratio = 1
        seg_combine_distance_ratio = np.round(seg_combine_distance_ratio)
        seg_distance_step = np.int_(np.fmin(seg_combine_distance_ratio, DISTANCE_SEG_NUM-current_seg_num))
    else:
        seg_distance_step = 1
    return seg_distance_step


def get_img_gray_intensity_step_blur(img_gray, is_copy=SEG_IMG_COPY):
    if is_copy:
        img_cp = img_copy(img_gray)
    else:
        img_cp = img_gray
    img_min_val = np.min(img_cp)
    img_step_blur = np.floor((img_cp - img_min_val)/INTENSITY_STEP_VAL)*INTENSITY_STEP_VAL+img_min_val
    return img_step_blur


def img_gray_edge_scan(img_gray, img_gray_edge_res, abs_delta_threshold, seg_mark_intensity=1):
    global SEG_EDGE_RELATIVE_RATIO_THRESHOLD
    abs_delta_threshold_param = get_img_gray_abs_delta_threshold_param(abs_delta_threshold)
    SEG_EDGE_RELATIVE_RATIO_THRESHOLD = SEG_EDGE_RELATIVE_RATIO_THRESHOLD_ROW
    img_gray_edge_scan_single_direction(img_gray, img_gray_edge_res, SEG_DIRECTION_ROW, abs_delta_threshold_param[0], seg_mark_intensity)
    SEG_EDGE_RELATIVE_RATIO_THRESHOLD = SEG_EDGE_RELATIVE_RATIO_THRESHOLD_COL
    img_gray_edge_scan_single_direction(img_gray, img_gray_edge_res, SEG_DIRECTION_COL, abs_delta_threshold_param[1], seg_mark_intensity)
    return img_gray_edge_res


def img_gray_edge_scan_single_direction(img_gray, img_gray_edge_res, seg_direction, abs_delta_threshold, seg_mark_intensity=1):
    img_shape = img_gray.shape
    if seg_direction == SEG_DIRECTION_ROW:
        seg_len = img_shape[0]
    else:
        seg_len = img_shape[1]
    i = 0
    edge_scanner = get_seg_edge_scanner()
    while i < seg_len:
        img_seg = img_gray_extract_seg(img_gray, seg_direction, i)
        edge_result_seg = img_gray_extract_seg(img_gray_edge_res, seg_direction, i)
        #img_gray_seg_edge_scan(img_seg, abs_delta_threshold, edge_result_seg, seg_mark_intensity)
        #img_gray_seg_edge_scan_relative_contrast(img_seg, abs_delta_threshold, edge_result_seg, seg_mark_intensity)
        edge_scanner(img_seg, abs_delta_threshold, edge_result_seg, seg_mark_intensity)
        i += 1
    return img_gray_edge_res


def get_seg_edge_scanner():
    if SEG_EDGE_SCAN_TYPE == SEG_EDGE_SCAN_TYPE_STATIC_CONTRAST:
        edge_scanner = img_gray_seg_edge_scan
    else:
        edge_scanner = img_gray_seg_edge_scan_relative_contrast
    return edge_scanner


def get_img_gray_abs_delta_threshold_param(abs_delta_threshold):
    if isinstance(abs_delta_threshold, list):
        res_abs_delta_threshold = [abs_delta_threshold[0], abs_delta_threshold[1]]
    else:
        res_abs_delta_threshold = [abs_delta_threshold, abs_delta_threshold]
    return res_abs_delta_threshold


def img_gray_seg_edge_scan_relative_contrast(img_seg, abs_intensity_threshold, edge_result_seg, seg_mark_intensity=1):
    seg_len = len(img_seg)
    edge_renderer = get_edge_renderer()
    edge_detection_window = np.zeros(3)
    edge_detection_mask = np.ones(3)
    edge_info_window = [None]*3
    i = 2
    i += extract_seg_triple_pixel_info(img_seg, edge_detection_window, edge_info_window, i, 0)
    i += extract_seg_triple_pixel_info(img_seg, edge_detection_window, edge_info_window, i, 1)
    j = 2
    while i < seg_len:
        top_idx_step = extract_seg_triple_pixel_info(img_seg, edge_detection_window, edge_info_window, i, j)
        j = (j+1) % 3
        if j == 0:
            #edge_window_update_info = get_edge_window_update_info(edge_detection_window, edge_detection_mask, edge_info_window, abs_intensity_threshold)
            edge_window_update_info = get_edge_window_update_info_shape_statement(edge_detection_window, edge_detection_mask, edge_info_window, abs_intensity_threshold)
            edge_window_update_idx = edge_window_update_info[0]
            shift_start_idx = 0
            if edge_window_update_idx != SEG_EDGE_UPDATE_INDEX_NO_UPDATE:
                seg_triple_pixel_monotonic_info = edge_window_update_info[1]
                update_idx = seg_triple_pixel_monotonic_info[gs_gen.TRIPLE_PIXEL_UPDATE_IDX]
                top_idx = i - 2 + edge_window_update_idx
                edge_renderer(img_seg, edge_result_seg, top_idx, update_idx, seg_mark_intensity)
                #shift_start_idx = edge_window_update_idx #2
            shift_range = shift_start_idx + 1
            j = left_shift_window(edge_detection_window, edge_info_window, shift_start_idx, shift_range)
        i += top_idx_step
    return edge_result_seg


def left_shift_window(edge_detection_window, edge_info_window, shift_start_idx, shift_range):
    i = shift_start_idx
    j = -1
    window_len = len(edge_detection_window)
    while i < window_len:
        jump_idx = i - shift_range
        if jump_idx >= 0:
            j = jump_idx
            edge_detection_window[j] = edge_detection_window[i]
            edge_info_window[j] = edge_info_window[i]
        i += 1
    update_start_idx = j+1
    return update_start_idx


def get_edge_window_update_info(edge_detection_window, edge_detection_mask, edge_info_window, abs_intensity_threshold):
    update_index = SEG_EDGE_UPDATE_INDEX_NO_UPDATE
    max_delta_edge_seg_triple_pixel_monotonic_info = None
    triple_abs_delta_total = edge_detection_window@edge_detection_mask
    if triple_abs_delta_total > 0:
        max_abs_delta_idx = np.argmax(edge_detection_window)
        #min_abs_delta_idx = np.argmin(edge_detection_window)
        max_abs_delta = edge_detection_window[max_abs_delta_idx]
        #min_abs_delta = edge_detection_window[min_abs_delta_idx]
        edge_delta_checker = max_abs_delta/triple_abs_delta_total
        max_delta_edge_seg_triple_pixel_monotonic_info = edge_info_window[max_abs_delta_idx]
        max_edge_intensity_val = max_delta_edge_seg_triple_pixel_monotonic_info[gs_gen.TRIPLE_PIXEL_MAX_DATA_IDX]
        is_max_edge_monotonic = max_delta_edge_seg_triple_pixel_monotonic_info[gs_gen.TRIPLE_PIXEL_IS_MONOTONIC_IDX]
        min_delta_edge_intensity_val = np.min(edge_detection_window)
        min_abs_delta = get_z_plane_edge_check_min_abs_delta(edge_detection_window)
        min_max_abs_delta_ratio = min_abs_delta/max_abs_delta
        max_edge_abs_grad_delta = max_delta_edge_seg_triple_pixel_monotonic_info[gs_gen.TRIPLE_PIXEL_ABS_GRAD_DELTA_RATIO_IDX]
        if is_max_edge_monotonic:
            climb_speed_checker = check_edge_climb_speed_stability(edge_info_window, max_abs_delta_idx)
        else:
            climb_speed_checker = 0
        if min_delta_edge_intensity_val > 0 and is_max_edge_monotonic and edge_delta_checker >= SEG_EDGE_RELATIVE_RATIO_THRESHOLD and max_edge_intensity_val >= abs_intensity_threshold:# and climb_speed_checker >= 0.8: # and min_max_abs_delta_ratio <= 0.2:# and edge_delta_checker <= 0.45:
            update_index = max_abs_delta_idx
    return [update_index, max_delta_edge_seg_triple_pixel_monotonic_info]


def get_edge_window_update_info_shape_statement(edge_detection_window, edge_detection_mask, edge_info_window, abs_intensity_threshold):
    update_index = SEG_EDGE_UPDATE_INDEX_NO_UPDATE
    seg_triple_pixel_monotonic_info_0 = edge_info_window[0]
    seg_triple_pixel_monotonic_info_1 = edge_info_window[1]
    seg_triple_pixel_monotonic_info_2 = edge_info_window[2]
    monotonic_type_0 = seg_triple_pixel_monotonic_info_0[gs_gen.TRIPLE_PIXEL_GRAD_MONOTONIC_TYPE_IDX]
    monotonic_type_1 = seg_triple_pixel_monotonic_info_1[gs_gen.TRIPLE_PIXEL_GRAD_MONOTONIC_TYPE_IDX]
    monotonic_type_2 = seg_triple_pixel_monotonic_info_2[gs_gen.TRIPLE_PIXEL_GRAD_MONOTONIC_TYPE_IDX]
    monotonic_type_sum = monotonic_type_0 + monotonic_type_1
    monotonic_climb_ratio_0 = seg_triple_pixel_monotonic_info_0[gs_gen.TRIPLE_PIXEL_ABS_GRAD_DELTA_RATIO_IDX]
    monotonic_climb_ratio_1 = seg_triple_pixel_monotonic_info_1[gs_gen.TRIPLE_PIXEL_ABS_GRAD_DELTA_RATIO_IDX]
    monotonic_climb_ratio_sum = monotonic_climb_ratio_1 + monotonic_climb_ratio_0
    is_strict_cb_shape = monotonic_type_0 == gs_gen.TRIPLE_PIXEL_GRAD_MONOTONIC_CONCAVE and monotonic_type_1 == gs_gen.TRIPLE_PIXEL_GRAD_MONOTONIC_BULGE
    is_strict_cb_shape_1 = monotonic_type_2 == gs_gen.TRIPLE_PIXEL_GRAD_MONOTONIC_SIMPLE
    if monotonic_type_0 > 0 and monotonic_type_1 > 0 and is_strict_cb_shape and is_strict_cb_shape_1 and np.fmax(monotonic_climb_ratio_0, monotonic_climb_ratio_1)/monotonic_climb_ratio_sum > 0: #monotonic_type_sum == 3: monotonic_type_sum == 3#
        update_index = 1
    return [update_index, seg_triple_pixel_monotonic_info_0]


def check_edge_climb_speed_stability(edge_info_window, max_delta_idx):
    climb_speed_window = np.zeros(3)
    climb_speed_mask = np.ones(3)
    climb_speed_mask[max_delta_idx] = 0
    fill_climb_speed_window(edge_info_window, climb_speed_window)
    climb_speed_checker = climb_speed_window@climb_speed_mask#np.max(climb_speed_window * climb_speed_mask)
    if climb_speed_checker > 0:
        climb_speed_checker = climb_speed_window[max_delta_idx]/climb_speed_checker
    return climb_speed_checker


def fill_climb_speed_window(edge_info_window, climb_speed_window):
    i = 0
    window_len = len(climb_speed_window)
    while i < window_len:
        seg_triple_pixel_monotonic_info = edge_info_window[i]
        climb_speed_val = seg_triple_pixel_monotonic_info[gs_gen.TRIPLE_PIXEL_ABS_GRAD_DELTA_RATIO_IDX]
        if seg_triple_pixel_monotonic_info[gs_gen.TRIPLE_PIXEL_IS_MONOTONIC_IDX] and seg_triple_pixel_monotonic_info[gs_gen.TRIPLE_PIXEL_GRAD_MONOTONIC_TYPE_IDX] != gs_gen.TRIPLE_PIXEL_GRAD_MONOTONIC_SIMPLE and climb_speed_val <= 0.66:
            climb_speed_window[i] = seg_triple_pixel_monotonic_info[gs_gen.TRIPLE_PIXEL_ABS_GRAD_DELTA_RATIO_IDX]
        i += 1
    return climb_speed_window


def get_z_plane_edge_check_min_abs_delta(edge_detection_window):
    max_abs_delta_idx = np.argmax(edge_detection_window)
    if max_abs_delta_idx != 1:
        min_abs_delta = edge_detection_window[1]
    else:
        min_abs_delta = np.min(edge_detection_window)
    return min_abs_delta


def extract_seg_triple_pixel_info(img_seg, edge_detection_window, edge_info_window, scan_idx, fill_idx):
    seg_triple_pixel_monotonic_info = gs_gen.seg_triple_pixel_monotonic_analysis(img_seg, scan_idx)
    top_idx_step = fill_edge_detection_window(edge_detection_window, fill_idx, seg_triple_pixel_monotonic_info)
    edge_info_window[fill_idx] = seg_triple_pixel_monotonic_info
    return top_idx_step


def fill_edge_detection_window(edge_detection_window, fill_idx, seg_triple_pixel_monotonic_info):
    is_monotonic = seg_triple_pixel_monotonic_info[gs_gen.TRIPLE_PIXEL_IS_MONOTONIC_IDX]
    top_idx_step = 1
    #if is_monotonic:
    edge_detection_window[fill_idx] = seg_triple_pixel_monotonic_info[gs_gen.TRIPLE_PIXEL_ABS_TOTAL_DELTA_VAL_IDX]
    #edge_detection_window[fill_idx] = seg_triple_pixel_monotonic_info[gs_gen.TRIPLE_PIXEL_ABS_GRAD_DELTA_RATIO_IDX]
    #else:
        #edge_detection_window[fill_idx] = 0
    if is_monotonic:
        top_idx_step = 1
    return top_idx_step


def img_gray_seg_edge_scan(img_seg, abs_delta_threshold, edge_result_seg, seg_mark_intensity=1):
    seg_len = len(img_seg)
    i = 2
    current_state = SEG_EDGE_STATE_INITIAL
    edge_renderer = get_edge_renderer()
    prev_top_idx = i
    current_body_state = SEG_EDGE_BODY_STATE_FLAT
    while i < seg_len:
        seg_triple_pixel_monotonic_info = gs_gen.seg_triple_pixel_monotonic_analysis(img_seg, i)
        edge_check_info = edge_check(seg_triple_pixel_monotonic_info, abs_delta_threshold, current_state, prev_top_idx, i)
        is_edge = edge_check_info[0]
        #if seg_triple_pixel_monotonic_info[gs_gen.TRIPLE_PIXEL_IS_MONOTONIC_IDX] and seg_triple_pixel_monotonic_info[gs_gen.TRIPLE_PIXEL_ABS_TOTAL_DELTA_VAL_IDX] >= abs_delta_threshold:
        if is_edge:
            current_state = edge_check_info[1]
            if seg_triple_pixel_monotonic_info[gs_gen.TRIPLE_PIXEL_DATA_TOP_TYPE_IDX] == gs_gen.TRIPLE_PIXEL_MAX_DATA_TOP and seg_triple_pixel_monotonic_info[gs_gen.TRIPLE_PIXEL_GRAD_MONOTONIC_TYPE_IDX] == gs_gen.TRIPLE_PIXEL_GRAD_MONOTONIC_BULGE:
                current_body_state = SEG_EDGE_BODY_STATE_UP
            elif seg_triple_pixel_monotonic_info[gs_gen.TRIPLE_PIXEL_DATA_TOP_TYPE_IDX] == gs_gen.TRIPLE_PIXEL_MIN_DATA_TOP and seg_triple_pixel_monotonic_info[gs_gen.TRIPLE_PIXEL_GRAD_MONOTONIC_TYPE_IDX] == gs_gen.TRIPLE_PIXEL_GRAD_MONOTONIC_BULGE:
                current_body_state = SEG_EDGE_BODY_STATE_DOWN
            else:
                current_body_state = SEG_EDGE_BODY_STATE_FLAT
            #if seg_triple_pixel_monotonic_info[gs_gen.TRIPLE_PIXEL_GRAD_MONOTONIC_TYPE_IDX] == gs_gen.TRIPLE_PIXEL_GRAD_MONOTONIC_CONCAVE:
            #if seg_triple_pixel_monotonic_info[gs_gen.TRIPLE_PIXEL_MIN_DATA_IDX] <= 0.4 and seg_triple_pixel_monotonic_info[gs_gen.TRIPLE_PIXEL_MAX_DATA_IDX] <= 0.5 and seg_triple_pixel_monotonic_info[gs_gen.TRIPLE_PIXEL_GRAD_MONOTONIC_TYPE_IDX] == gs_gen.TRIPLE_PIXEL_GRAD_MONOTONIC_CONCAVE:
            #if seg_triple_pixel_monotonic_info[gs_gen.TRIPLE_PIXEL_MAX_DATA_IDX] >= 0.61 and seg_triple_pixel_monotonic_info[gs_gen.TRIPLE_PIXEL_MIN_DATA_IDX] <= 0.55 and current_body_state != SEG_EDGE_BODY_STATE_FLAT and ((i <seg_len-15 and img_seg[i+15] >= 0.61) or (i >= 15 and img_seg[i-15] >= 0.61)):
            update_idx = seg_triple_pixel_monotonic_info[gs_gen.TRIPLE_PIXEL_UPDATE_IDX]
            edge_renderer(img_seg, edge_result_seg, i, update_idx, seg_mark_intensity)
            prev_top_idx = i
                #current_state = edge_check_info[1]
            #if seg_mark_intensity == SEG_EDGE_INTENSITY_ORIGINAL:
                #edge_result_seg[update_idx] = img_seg[update_idx]
                #edge_result_seg[i] = img_seg[i]
                #edge_result_seg[i-1] = img_seg[i-1]
                #edge_result_seg[i-2] = img_seg[i-2]
            #else:
                #edge_result_seg[update_idx] = seg_mark_intensity
        i += 1
    return edge_result_seg


def edge_check(seg_triple_pixel_monotonic_info, abs_delta_threshold, current_state, prev_top_idx, current_top_idx):
    updated_state = seg_triple_pixel_monotonic_info[gs_gen.TRIPLE_PIXEL_TOTAL_DELTA_VAL_SIGN_IDX]
    is_monotonic_texture = seg_triple_pixel_monotonic_info[gs_gen.TRIPLE_PIXEL_IS_MONOTONIC_IDX] and seg_triple_pixel_monotonic_info[gs_gen.TRIPLE_PIXEL_ABS_TOTAL_DELTA_VAL_IDX] >= abs_delta_threshold #and seg_triple_pixel_monotonic_info[gs_gen.TRIPLE_PIXEL_ABS_TOTAL_DELTA_VAL_IDX] <= 0.07 and seg_triple_pixel_monotonic_info[gs_gen.TRIPLE_PIXEL_GRAD_MONOTONIC_TYPE_IDX] == gs_gen.TRIPLE_PIXEL_GRAD_MONOTONIC_BULGE
    #is_edge_state = (SEG_EDGE_CHECK_STRATEGY == SEG_EDGE_THIN_LINE and (updated_state*current_state < 0 or current_state == SEG_EDGE_STATE_INITIAL)) or (SEG_EDGE_CHECK_STRATEGY == SEG_EDGE_TEXTURE)
    state_checker = updated_state * current_state
    is_edge_state = ((SEG_EDGE_CHECK_STRATEGY == SEG_EDGE_THIN_LINE or SEG_EDGE_CHECK_STRATEGY == SEG_EDGE_THIN_LINE_3D) and (state_checker < 0 or (updated_state != 0 and current_state == 0) or (updated_state == 0 and prev_top_idx != current_state-1 and (current_state == SEG_EDGE_STATE_INITIAL or (current_top_idx-prev_top_idx) > 1)))) or (SEG_EDGE_CHECK_STRATEGY == SEG_EDGE_TEXTURE)
    if state_checker < 0:
        updated_state = 0
    is_edge = is_monotonic_texture and is_edge_state
    if current_state == SEG_EDGE_STATE_INITIAL:
        is_left_edge = 1
    else:
        is_left_edge = 0
    return [is_edge, updated_state, is_left_edge]


def edge_renderer_single_pixel(img_seg, edge_result_seg, top_idx, update_idx, edge_render_intensity):
    if edge_render_intensity == SEG_EDGE_INTENSITY_ORIGINAL:
        edge_result_seg[update_idx] = img_seg[update_idx]
    else:
        edge_result_seg[update_idx] = edge_render_intensity
    return edge_result_seg


def edge_renderer_triple_pixel(img_seg, edge_result_seg, top_idx, update_idx, edge_render_intensity):
    if edge_render_intensity == SEG_EDGE_INTENSITY_ORIGINAL:
        edge_result_seg[top_idx-2:top_idx+1] = img_seg[top_idx-2:top_idx+1]
    else:
        edge_result_seg[top_idx-2:top_idx+1] = edge_render_intensity
    return edge_result_seg


def get_edge_renderer():
    if SEG_EDGE_CHECK_STRATEGY == SEG_EDGE_THIN_LINE:
        edge_renderer = edge_renderer_single_pixel
    else:
        edge_renderer = edge_renderer_triple_pixel
    return edge_renderer


def get_traditional_edge_detector(edge_detector_type):
    if edge_detector_type == TRADITIONAL_EDGE_DETECTOR_SOBEL:
        edge_detector = img_blur_edge_detection_sobel
    else:
        edge_detector = img_blur_edge_detection_canny
    return edge_detector


def img_edge_detection_traditional(file_path_name, edge_detector_type, input_type=TRADITIONAL_EDGE_DETECTOR_INPUT_GAUSS_BLUR):
    img = cv2.imread(file_path_name)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edge_info = img_gray_edge_detection_traditional(img_gray, edge_detector_type, input_type)
    return edge_info


def img_gray_edge_detection_traditional(img_gray, edge_detector_type, input_type=TRADITIONAL_EDGE_DETECTOR_INPUT_GAUSS_BLUR):
    if np.max(img_gray) <= 1:
        img_gray = img_mask_extract(img_gray, 255)
    if input_type == TRADITIONAL_EDGE_DETECTOR_INPUT_GAUSS_BLUR:
        img_blur = np.int_(np.round(get_img_gray_gauss_blur(img_gray))).astype('u1')
    elif input_type == TRADITIONAL_EDGE_DETECTOR_INPUT_POISSON_DISTANCE:
        img_blur = np.int_(np.round(get_img_gray_poisson_seg(img_gray/255)*255)).astype('u1')
    else:
        img_blur = np.int_(img_gray).astype('u1')
    edge_detector = get_traditional_edge_detector(edge_detector_type)
    edge_info = edge_detector(img_blur)
    return edge_info


def get_img_gray_gauss_blur(img_gray):
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    return img_blur


def get_img_gray_simple_blur(img_gray):
    img_blur = cv2.blur(img_gray, (1, 20), 0)
    return img_blur


def img_blur_edge_detection_sobel(img_blur):
    time_start = tm.time()
    sobel_edge_x = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)#5
    time_end = tm.time()
    time_x = time_end - time_start
    time_start = tm.time()
    sobel_edge_y = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)
    time_end = tm.time()
    time_y = time_end - time_start
    time_start = tm.time()
    sobel_edge_x_y = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
    time_end = tm.time()
    time_x_y = time_end - time_start
    return [sobel_edge_x_y, sobel_edge_x, sobel_edge_y, time_x, time_y, time_x_y]


def img_blur_edge_detection_canny(img_blur):
    time_start = tm.time()
    candy_edge = cv2.Canny(image=img_blur, threshold1=TRADITIONAL_EDGE_DETECTOR_CANNY_THRESHOLD_1, threshold2=TRADITIONAL_EDGE_DETECTOR_CANNY_THRESHOLD_2) #, apertureSize=5)#100 200
    time_end = tm.time()
    time_edge = time_end - time_start
    return [candy_edge, time_edge]


def analysis_img_gray_seg_z_aix_angle(img_gray, seg_direction, seg_start, start_point, end_point):
    img_seg = get_img_gray_seg(img_gray, seg_direction, seg_start, start_point, end_point)
    seg_len = len(img_seg)
    monotonic_continuous_pixel_sequence_list = get_continuous_monotonic_pixel_sequence_list(img_seg, 2, seg_len-1)
    continuous_pixel_list_len = len(monotonic_continuous_pixel_sequence_list)
    continuous_pixel_angle_transfer_list = [None]*continuous_pixel_list_len
    i = 0
    while i < continuous_pixel_list_len:
        angle_transfer_info = get_img_gray_z_aix_transfer_angle_info(monotonic_continuous_pixel_sequence_list[i])
        #continuous_pixel_angle_transfer_list.append(angle_transfer_info)
        continuous_pixel_angle_transfer_list[i] = angle_transfer_info
        i += 1
    return continuous_pixel_angle_transfer_list


def get_continuous_monotonic_pixel_sequence_list(img_seg, start_idx, seg_end_idx):
    monotonic_pixel_sequence = []
    monotonic_pixel_sequence_list = []
    i = start_idx
    is_filled = True
    while i <= seg_end_idx:
        seg_triple_pixel_info = gs_gen.seg_triple_pixel_monotonic_analysis(img_seg, i)
        if seg_triple_pixel_info[gs_gen.TRIPLE_PIXEL_IS_STRICT_MONOTONIC_IDX]:
            fill_continuous_monotonic_pixel_sequence(seg_triple_pixel_info, monotonic_pixel_sequence)
            is_filled = False
        elif len(monotonic_pixel_sequence) > 0:
            #monotonic_pixel_sequence_nd = np.array(monotonic_pixel_sequence)
            #monotonic_pixel_sequence_list.append(monotonic_pixel_sequence_nd)
            #monotonic_pixel_sequence = []
            monotonic_pixel_sequence = fill_monotonic_pixel_sequence_list(monotonic_pixel_sequence, monotonic_pixel_sequence_list)
            is_filled = True
        i += 1
    if not is_filled:
        fill_monotonic_pixel_sequence_list(monotonic_pixel_sequence, monotonic_pixel_sequence_list)
    return monotonic_pixel_sequence_list


def fill_monotonic_pixel_sequence_list(monotonic_pixel_sequence, monotonic_pixel_sequence_list):
    monotonic_pixel_sequence_nd = np.array(monotonic_pixel_sequence)
    monotonic_pixel_sequence_list.append(monotonic_pixel_sequence_nd)
    empty_monotonic_pixel_list = []
    return empty_monotonic_pixel_list


def fill_continuous_monotonic_pixel_sequence(seg_triple_pixel_info, monotonic_pixel_sequence):
    if len(monotonic_pixel_sequence) == 0:
        monotonic_pixel_sequence.append(seg_triple_pixel_info[gs_gen.TRIPLE_PIXEL_DELTA_1_IDX])
    monotonic_pixel_sequence.append(seg_triple_pixel_info[gs_gen.TRIPLE_PIXEL_DELTA_2_IDX])
    return monotonic_pixel_sequence


def get_img_gray_z_aix_transfer_angle_info(continuous_monotonic_triple_pixel_grad_list):
    list_len = len(continuous_monotonic_triple_pixel_grad_list)
    z_aix_angle_list = np.zeros(list_len-1)
    total_angle_sum = 0
    i = 1
    while i < list_len:
        z_aix_angle = (continuous_monotonic_triple_pixel_grad_list[i] - continuous_monotonic_triple_pixel_grad_list[i-1])/TRIPLE_PIXEL_SECOND_ORDER_GRAD_Z_ANGLE_PI
        total_angle_sum += z_aix_angle
        z_aix_angle_list[i-1] = z_aix_angle
        i += 1
    return [total_angle_sum, z_aix_angle_list]


def analysis_img_gray_seg_z_aix_angle_seg_info(img_gray, seg_direction, seg_start, start_point, end_point):
    img_seg = get_img_gray_seg(img_gray, seg_direction, seg_start, start_point, end_point)
    seg_len = len(img_seg)
    monotonic_continuous_pixel_seg_info_sequence_list = get_continuous_monotonic_pixel_seg_info_sequence_list(img_seg, 2, seg_len-1)
    continuous_pixel_angle_transfer_list = get_z_aix_angle_transfer_vector_list_from_monotonic_seg_info_sequence_list(monotonic_continuous_pixel_seg_info_sequence_list)
    return continuous_pixel_angle_transfer_list


def get_continuous_monotonic_pixel_seg_info_sequence_list(img_seg, start_idx, seg_end_idx):
    monotonic_seg_info_sequence = []
    monotonic_seg_info_sequence_list = []
    i = start_idx
    is_filled = True
    while i <= seg_end_idx:
        seg_triple_pixel_info = gs_gen.seg_triple_pixel_monotonic_analysis(img_seg, i)
        if seg_triple_pixel_info[gs_gen.TRIPLE_PIXEL_IS_STRICT_MONOTONIC_IDX]:
            monotonic_seg_info_sequence.append(seg_triple_pixel_info)
            is_filled = False
        elif len(monotonic_seg_info_sequence) > 0:
            monotonic_seg_info_sequence_list.append(monotonic_seg_info_sequence)
            monotonic_seg_info_sequence = []
            is_filled = True
        i += 1
    if not is_filled:
        monotonic_seg_info_sequence_list.append(monotonic_seg_info_sequence)
    return monotonic_seg_info_sequence_list


def get_z_aix_angle_transfer_vector_list_from_monotonic_seg_info_sequence_list(seg_info_sequence_list):
    seg_info_sequence_list_len = len(seg_info_sequence_list)
    z_aix_transfer_vector_list = [None]*seg_info_sequence_list_len
    i = 0
    while i < seg_info_sequence_list_len:
        z_aix_transfer_vectors = get_z_aix_angle_transfer_vectors_from_monotonic_seg_info_sequence(seg_info_sequence_list[i])
        z_aix_transfer_vector_list[i] = z_aix_transfer_vectors
        i += 1
    return z_aix_transfer_vector_list


def get_z_aix_angle_transfer_vectors_from_monotonic_seg_info_sequence(seg_info_sequence):
    sequence_len = len(seg_info_sequence)
    z_aix_transfer_vector_list = [None]*sequence_len
    i = 0
    current_angle = 0
    while i < sequence_len:
        z_aix_transfer_info = get_img_gray_z_aix_angle_transfer_vector(seg_info_sequence[i], current_angle)
        z_aix_transfer_vector_list[i] = z_aix_transfer_info[0]
        current_angle = z_aix_transfer_info[1]
        i += 1
    return z_aix_transfer_vector_list


def get_img_gray_z_aix_angle_transfer_vector(seg_triple_pixel_info, previous_angle, is_previous_seg_strict_monotonic=False, prevoius_rho=1, previous_total_abs_delta=0, previous_grad_direction=ANGLE_GRAD_STEADY, previous_curve_type=gs_gen.TRIPLE_PIXEL_GRAD_MONOTONIC_SIMPLE_FLAT):
    abs_delta_1 = seg_triple_pixel_info[gs_gen.TRIPLE_PIXEL_ABS_DELTA_1_IDX]
    abs_delta_2 = seg_triple_pixel_info[gs_gen.TRIPLE_PIXEL_ABS_DELTA_2_IDX]
    total_abs_delta = seg_triple_pixel_info[gs_gen.TRIPLE_PIXEL_ABS_DELTA_SUM_IDX]
    curve_side = seg_triple_pixel_info[gs_gen.TRIPLE_PIXEL_CURVE_SIDE_IDX]
    curve_type = seg_triple_pixel_info[gs_gen.TRIPLE_PIXEL_GRAD_MONOTONIC_TYPE_IDX]
    current_strict_monotonic_state = seg_triple_pixel_info[gs_gen.TRIPLE_PIXEL_IS_STRICT_MONOTONIC_IDX]
    pixel_monotonic_direction_type = seg_triple_pixel_info[gs_gen.TRIPLE_PIXEL_MONOTONIC_DIRECTION_IDX]
    vibrate_direction = seg_triple_pixel_info[gs_gen.TRIPLE_PIXEL_VIBRATE_DIRECTION_IDX]
    extra_output = []
    curve_angle_calculator = get_curve_z_aix_transfer_angle_calculator(curve_side)
    z_aix_transfer_angle = curve_angle_calculator(abs_delta_1, abs_delta_2, total_abs_delta, curve_type, previous_angle, is_previous_seg_strict_monotonic, pixel_monotonic_direction_type, prevoius_rho, extra_output, previous_total_abs_delta, vibrate_direction, previous_grad_direction, previous_curve_type)
    current_angle = z_aix_transfer_angle #+previous_angle
    rho = extra_output[0]
    current_grad_direction = extra_output[1]
    is_filter_out = extra_output[2]
    z_aix_transfer_angle_vector = np.array([rho*np.cos(current_angle), rho*np.sin(current_angle)])
    if curve_type == gs_gen.TRIPLE_PIXEL_GRAD_MONOTONIC_SIMPLE_VIBRATE:
        new_total_abs_delta = np.fmax(abs_delta_1, abs_delta_2)
    else:
        new_total_abs_delta =total_abs_delta
    #if z_aix_transfer_angle_vector[1] == -1:
        #z_aix_transfer_angle_vector[1] = 1
    return [z_aix_transfer_angle_vector, current_angle, current_strict_monotonic_state, rho, new_total_abs_delta, current_grad_direction, is_filter_out, curve_type]


def get_curve_z_aix_transfer_angle_calculator(curve_side):
    #curve_angle_calculator = calculate_left_side_z_aix_transfer_angle
    #curve_angle_calculator = calculate_z_aix_transfer_angle_unified_method

    #curve_angle_calculator = calculate_z_aix_transfer_angle_direction_method
    curve_angle_calculator = calculate_z_aix_transfer_angle_unified_method_rho
    '''
    if curve_side == gs_gen.TRIPLE_PIXEL_CURVE_SIDE_LEFT:
        curve_angle_calculator = calculate_left_side_z_aix_transfer_angle
    else:
        curve_angle_calculator = calculate_right_side_z_aix_transfer_angle
    '''
    return curve_angle_calculator


def calculate_left_side_z_aix_transfer_angle(abs_delta_1, abs_delta_2, total_abs_delta, curve_type, previous_angle):
    gamma = np.fmax(abs_delta_1, abs_delta_2)/total_abs_delta
    if curve_type == gs_gen.TRIPLE_PIXEL_GRAD_MONOTONIC_BULGE:
        aix_transfer_angle_coefficient = 1 - gamma
    elif curve_type == gs_gen.TRIPLE_PIXEL_GRAD_MONOTONIC_CONCAVE:
        aix_transfer_angle_coefficient = gamma - 1
    else:
        aix_transfer_angle_coefficient = 0.5#-0.5
    aix_transfer_angle = get_angle_by_coefficient(aix_transfer_angle_coefficient)
    return aix_transfer_angle


def calculate_right_side_z_aix_transfer_angle(abs_delta_1, abs_delta_2, total_abs_delta, curve_type, previous_angle):
    gamma = np.fmax(abs_delta_1, abs_delta_2)/total_abs_delta
    if curve_type == gs_gen.TRIPLE_PIXEL_GRAD_MONOTONIC_BULGE:
        aix_transfer_angle_coefficient = 1 - gamma #1.5 - gamma
    elif curve_type == gs_gen.TRIPLE_PIXEL_GRAD_MONOTONIC_CONCAVE:
        aix_transfer_angle_coefficient = gamma - 1 # gamma - 1.5
    else:
        aix_transfer_angle_coefficient = 0.5
    aix_transfer_angle = get_angle_by_coefficient(aix_transfer_angle_coefficient)
    return aix_transfer_angle


def calculate_z_aix_transfer_angle_unified_method(abs_delta_1, abs_delta_2, total_abs_delta, curve_type, previous_angle, is_previous_seg_strict_monotonic=False):
    #abs_delta_sum = abs_delta_1 + abs_delta_2
    if total_abs_delta > 0:
        gamma = np.fmax(abs_delta_1, abs_delta_2)/total_abs_delta
    if curve_type == gs_gen.TRIPLE_PIXEL_GRAD_MONOTONIC_BULGE or curve_type == gs_gen.TRIPLE_PIXEL_GRAD_MONOTONIC_LINEAR_BULGE: #or curve_type == gs_gen.TRIPLE_PIXEL_GRAD_MONOTONIC_LINEAR_CONCAVE:
        pi_coefficient = 1 - gamma
        alpha_coefficient = 2*gamma - 1
    elif curve_type == gs_gen.TRIPLE_PIXEL_GRAD_MONOTONIC_CONCAVE or curve_type == gs_gen.TRIPLE_PIXEL_GRAD_MONOTONIC_LINEAR_CONCAVE:
        pi_coefficient = gamma - 1
        alpha_coefficient = 2*gamma - 1
    elif curve_type != gs_gen.TRIPLE_PIXEL_GRAD_MONOTONIC_SIMPLE_FLAT and is_previous_seg_strict_monotonic:
        pi_coefficient = 0
        alpha_coefficient = 2*(1-gamma)
    else:
        pi_coefficient = 0
        alpha_coefficient = 1
    pi_part = get_angle_by_coefficient(pi_coefficient)
    alpha_part = alpha_coefficient*previous_angle
    aix_transfer_angle = pi_part + alpha_part
    return aix_transfer_angle


def calculate_z_aix_transfer_angle_direction_method(abs_delta_1, abs_delta_2, total_abs_delta, curve_type, previous_angle, is_previous_seg_strict_monotonic=False, pixel_monotonic_direction_type=gs_gen.TRIPLE_PIXEL_MONOTONIC_DIRECTION_VIBRATE, previous_rho=1, extra_output=None):
    rho = -1
    if total_abs_delta > 0:
        gamma = np.fmax(abs_delta_1, abs_delta_2)/total_abs_delta
    if pixel_monotonic_direction_type == gs_gen.TRIPLE_PIXEL_MONOTONIC_DIRECTION_UPWARD:
        pi_coefficient = 1 - gamma
        alpha_coefficient = 2 * gamma - 1
    elif pixel_monotonic_direction_type == gs_gen.TRIPLE_PIXEL_MONOTONIC_DIRECTION_DOWNWARD:
        pi_coefficient = gamma - 1
        alpha_coefficient = 2 * gamma - 1
    elif curve_type != gs_gen.TRIPLE_PIXEL_GRAD_MONOTONIC_SIMPLE_FLAT: #and is_previous_seg_strict_monotonic:
        pi_coefficient = 0
        alpha_coefficient = 2 * (1 - gamma)
        rho = previous_rho
    else:
        pi_coefficient = 0
        alpha_coefficient = 1
        gamma = 1
    pi_part = get_angle_by_coefficient(pi_coefficient)
    alpha_part = alpha_coefficient * previous_angle
    aix_transfer_angle = pi_part + alpha_part
    if extra_output is not None:
        if rho < 0:
            rho = 2*(1-gamma)*total_abs_delta + (2*gamma - 1)*0.01
        extra_output.append(rho)
    return aix_transfer_angle


def calculate_z_aix_transfer_angle_unified_method_rho(abs_delta_1, abs_delta_2, total_abs_delta, curve_type, previous_angle, is_previous_seg_strict_monotonic=False, pixel_monotonic_direction_type=gs_gen.TRIPLE_PIXEL_MONOTONIC_DIRECTION_VIBRATE, previous_rho=1, extra_output=None, previous_total_abs_delta=0, vibrate_direction=gs_gen.TRIPLE_PIXEL_VIBRATE_DIRECTION_NO, previous_grad_direction=ANGLE_GRAD_STEADY, previous_curve_type=gs_gen.TRIPLE_PIXEL_GRAD_MONOTONIC_SIMPLE_FLAT):
    #rho = -1
    #is_vibration_significance_info = pixel_vibration_significance_filter_local_ratio(previous_total_abs_delta, total_abs_delta)
    #is_vibration_significance = is_vibration_significance_info[0]
    #if not is_vibration_significance:
        #curve_type = gs_gen.TRIPLE_PIXEL_GRAD_MONOTONIC_SIMPLE_FLAT
    #if total_abs_delta < 0.5:
        #curve_type = gs_gen.TRIPLE_PIXEL_GRAD_MONOTONIC_SIMPLE_FLAT
    if total_abs_delta > 0:
        gamma = np.fmax(abs_delta_1, abs_delta_2)/total_abs_delta
    if curve_type == gs_gen.TRIPLE_PIXEL_GRAD_MONOTONIC_BULGE: # or curve_type == gs_gen.TRIPLE_PIXEL_GRAD_MONOTONIC_LINEAR_BULGE: #or curve_type == gs_gen.TRIPLE_PIXEL_GRAD_MONOTONIC_LINEAR_CONCAVE:
        angle_coefficient = get_bulge_angle_coefficient(gamma)
        pi_coefficient = angle_coefficient[0]#1 - gamma
        alpha_coefficient = angle_coefficient[1]#2*gamma - 1
    elif curve_type == gs_gen.TRIPLE_PIXEL_GRAD_MONOTONIC_LINEAR_BULGE or curve_type == gs_gen.TRIPLE_PIXEL_GRAD_MONOTONIC_LINEAR_CONCAVE:
        if previous_grad_direction == ANGLE_GRAD_UPWARD:
            angle_coefficient = get_bulge_angle_coefficient(gamma)
            pi_coefficient = angle_coefficient[0]  # 1 - gamma
            alpha_coefficient = angle_coefficient[1]  # 2*gamma - 1
        elif previous_grad_direction == ANGLE_GRAD_DOWNWARD:
            angle_coefficient = get_concave_angle_coefficient(gamma)
            pi_coefficient = angle_coefficient[0]  # gamma - 1
            alpha_coefficient = angle_coefficient[1]  # 2*gamma - 1
        else:
            angle_coefficient = get_simple_flat_coefficient()
            pi_coefficient = angle_coefficient[0]  # 0
            alpha_coefficient = angle_coefficient[1]  # 1
            gamma = angle_coefficient[2]  # 1
    elif curve_type == gs_gen.TRIPLE_PIXEL_GRAD_MONOTONIC_CONCAVE: #or curve_type == gs_gen.TRIPLE_PIXEL_GRAD_MONOTONIC_LINEAR_CONCAVE:
        #if pixel_monotonic_direction_type == gs_gen.TRIPLE_PIXEL_MONOTONIC_DIRECTION_DOWNWARD:
        angle_coefficient = get_concave_angle_coefficient(gamma)
        pi_coefficient = angle_coefficient[0]#gamma - 1
        alpha_coefficient = angle_coefficient[1]#2*gamma - 1
        #else:
            #angle_coefficient = get_bulge_angle_coefficient(gamma)
            #pi_coefficient = angle_coefficient[0]#1 - gamma
            #alpha_coefficient = angle_coefficient[1]#2 * gamma - 1
    elif curve_type != gs_gen.TRIPLE_PIXEL_GRAD_MONOTONIC_SIMPLE_FLAT: #and is_previous_seg_strict_monotonic:
        if vibrate_direction == gs_gen.TRIPLE_PIXEL_VIBRATE_DIRECTION_DOWNWARD:
            angle_coefficient = get_vibrate_coefficient(gamma)
        else:
            angle_coefficient = get_bulge_angle_coefficient(gamma)
        pi_coefficient = angle_coefficient[0]#0
        alpha_coefficient = angle_coefficient[1]#2 * (1 - gamma)
        #rho = previous_rho
    else:
        angle_coefficient = get_simple_flat_coefficient()
        pi_coefficient = angle_coefficient[0]#0
        alpha_coefficient = angle_coefficient[1]#1
        gamma = angle_coefficient[2]#1
    pi_part = get_angle_by_coefficient(pi_coefficient)
    alpha_part = alpha_coefficient * previous_angle
    aix_transfer_angle = pi_part + alpha_part

    filtered_angle_info = angle_grad_based_filter(previous_angle, aix_transfer_angle, previous_grad_direction, previous_curve_type, curve_type)
    aix_transfer_angle = filtered_angle_info[0]
    current_grad_direction = filtered_angle_info[1]
    is_filter_out = filtered_angle_info[2]
    if extra_output is not None:
        #if rho < 0:
            #rho = 2*(1-gamma)*total_abs_delta + (2*gamma - 1)*0.01
            #if rho/previous_rho < 0:
                #rho = previous_rho
                #aix_transfer_angle = previous_angle
        rho_info = get_step_rho(gamma, total_abs_delta)
        rho = rho_info[0]
        extra_output.append(rho)
        extra_output.append(current_grad_direction)
        extra_output.append(is_filter_out)
    return aix_transfer_angle


def get_bulge_angle_coefficient(gamma):
    pi_coefficient = 1 - gamma
    alpha_coefficient = 2 * gamma - 1
    return [pi_coefficient, alpha_coefficient]


def get_concave_angle_coefficient(gamma):
    pi_coefficient = gamma - 1
    alpha_coefficient = 2 * gamma - 1
    return [pi_coefficient, alpha_coefficient]


def get_vibrate_coefficient(gamma):
    pi_coefficient = 0
    alpha_coefficient = 2 * (1 - gamma)
    return [pi_coefficient, alpha_coefficient]


def get_simple_flat_coefficient():
    pi_coefficient = 0
    alpha_coefficient = 1
    gamma = 1
    return [pi_coefficient, alpha_coefficient, gamma]


def get_step_rho(gamma, total_abs_delta):
    rho = (2 * (1 - gamma) * total_abs_delta + (2 * gamma - 1) * COORDINATE_BASIC_STEP)
    return [rho]


def angle_to_vector_transfer(scan_angle):
    return np.array([np.cos(scan_angle), np.sin(scan_angle)])


def get_angle_by_coefficient(angle_coefficient):
    return angle_coefficient*np.pi


def pixel_vibration_significance_filter_local_ratio(previous_total_abs_delta, current_total_abs_delta):
    min_total_abs_delta = np.fmin(previous_total_abs_delta, current_total_abs_delta)
    if min_total_abs_delta > 0:
        vibration_significance_checker = current_total_abs_delta/previous_total_abs_delta#np.fmax(previous_total_abs_delta, current_total_abs_delta)/min_total_abs_delta
        is_vibration_significance = vibration_significance_checker >= COORDINATE_VIBRATION_SIGNIFICANCE_THRESHOLD
    else:
        is_vibration_significance = True
    return [is_vibration_significance, current_total_abs_delta]


def angle_grad_based_filter(previous_angle, current_angle, previous_grad_direction, previous_curve_type, current_curve_type):
    angle_filter = get_angle_filter(previous_angle)
    current_grad_direction = np.sign(current_angle - previous_angle)
    is_filter_out_info = angle_filter(previous_grad_direction, current_grad_direction)
    is_filter_out = is_filter_out_info[0]
    if current_curve_type == gs_gen.TRIPLE_PIXEL_GRAD_MONOTONIC_SIMPLE_VIBRATE_FLAT and previous_curve_type == gs_gen.TRIPLE_PIXEL_GRAD_MONOTONIC_SIMPLE_FLAT:
        is_filter_out = False
    if is_filter_out:
        final_angle = previous_angle
        final_grad_direction = previous_grad_direction
    else:
        final_angle = current_angle
        final_grad_direction = current_grad_direction
    return [final_angle, final_grad_direction, is_filter_out]


def get_angle_filter(previous_angle):
    if previous_angle > 0:
        angle_filter = positive_angle_filter
    elif previous_angle < 0:
        angle_filter = negative_angle_filter
    else:
        angle_filter = flat_angle_filter
    return angle_filter


def positive_angle_filter(previous_grad_direction, current_grad_direction):
    if current_grad_direction == ANGLE_GRAD_STEADY or ((previous_grad_direction == ANGLE_GRAD_UPWARD or previous_grad_direction == ANGLE_GRAD_STEADY) and current_grad_direction == ANGLE_GRAD_UPWARD):
    #if previous_grad_direction == ANGLE_GRAD_UPWARD or previous_grad_direction == ANGLE_GRAD_STEADY and current_grad_direction == ANGLE_GRAD_UPWARD:
        is_filter_out = True
    else:
        is_filter_out = False
    return[is_filter_out]


def negative_angle_filter(previous_grad_direction, current_grad_direction):
    if current_grad_direction == ANGLE_GRAD_STEADY or ((previous_grad_direction == ANGLE_GRAD_DOWNWARD or previous_grad_direction == ANGLE_GRAD_STEADY) and current_grad_direction == ANGLE_GRAD_DOWNWARD):
    #if (previous_grad_direction == ANGLE_GRAD_DOWNWARD or previous_grad_direction == ANGLE_GRAD_STEADY) and current_grad_direction == ANGLE_GRAD_DOWNWARD:
        is_filter_out = True
    else:
        is_filter_out = False
    return [is_filter_out]


def flat_angle_filter(previous_grad_direction, current_grad_direction):
    if current_grad_direction == ANGLE_GRAD_STEADY:
        is_filter_out = True
    else:
        is_filter_out = False
    return [is_filter_out]


def generate_vector_orbit_display(vectors):
    vector_num = len(vectors)
    x_coordinate = [None]*vector_num
    y_coordinate = [None]*vector_num
    v = vectors[0]
    x_coordinate[0] = v[0]
    y_coordinate[0] = v[1]
    z_aix_total_modification = v[1]
    i = 1
    while i < vector_num:
        v = vectors[i]
        x_coordinate[i] = v[0]
        y_coordinate[i] = v[1]
        z_aix_total_modification += v[1]
        i += 1
    return [x_coordinate, y_coordinate, z_aix_total_modification]


def generate_coordinate_modify_vectors(vectors):
    vector_num = len(vectors)
    coordinate_modify_vector_list = [None]*(vector_num-1)
    v = vectors[0]
    i = 1
    while i < vector_num:
        v = vectors[i] - v
        coordinate_modify_vector_list[i-1] = v
        i += 1
    return coordinate_modify_vector_list


def flat_list_1d_decrement(md_list):
    list_len = len(md_list)
    flat_list = []
    i = 0
    while i < list_len:
        list_item = md_list[i]
        fill_flat_list(list_item, flat_list)
        i += 1
    return flat_list


def fill_flat_list(list_item, flat_list):
    if isinstance(list_item, list):
        item_len = len(list_item)
    else:
        item_len = 0
        flat_list.append(list_item)
    i = 0
    while i < item_len:
        flat_list.append(list_item[i])
        i += 1
    return flat_list


def generate_complete_coordinate_x_z_scan(scan_coordinate):
    complete_coordinate = np.zeros(3)
    complete_coordinate[0] = scan_coordinate[0]
    complete_coordinate[2] = scan_coordinate[1]
    return complete_coordinate


def generate_complete_coordinate_y_z_scan(scan_coordinate):
    complete_coordinate = np.zeros(3)
    complete_coordinate[1] = scan_coordinate[0]
    complete_coordinate[2] = scan_coordinate[1]
    return complete_coordinate


def get_complete_coordinate_generator(seg_direction):
    if seg_direction == SEG_DIRECTION_ROW:
        complete_coordinate_generator = generate_complete_coordinate_x_z_scan
    else:
        complete_coordinate_generator = generate_complete_coordinate_y_z_scan
    return complete_coordinate_generator


def get_img_gray_seg_num(img_gray, seg_direction):
    img_shape = img_gray.shape
    if seg_direction == SEG_DIRECTION_ROW:
        img_seg_num = img_shape[0]
        img_seg_len = img_shape[1]
    else:
        img_seg_num = img_shape[1]
        img_seg_len = img_shape[0]
    return [img_seg_num, img_seg_len]


def increment_scan_coordinate(previous_coordinate, scan_vector):
    return previous_coordinate + scan_vector


def extract_img_seg_col_3d(img_gray, seg_start):
    img_seg = img_gray[:, seg_start, :]
    return img_seg


def extract_img_seg_row_3d(img_gray, seg_start):
    img_seg = img_gray[seg_start, :, :]
    return img_seg


def get_img_gray_seg_extract_fun_3d(seg_direction):
    if seg_direction == SEG_DIRECTION_ROW:
        img_seg_extract_fun = extract_img_seg_row_3d
    else:
        img_seg_extract_fun = extract_img_seg_col_3d
    return img_seg_extract_fun


def img_gray_3_dimensional_coordinate_reconstruction_scan(img_gray):
    coordinate_result_map_row_info = img_gray_single_direction_3_dimensional_coordinate_reconstruction_scan(img_gray, SEG_DIRECTION_ROW)
    coordinate_result_map_col_info = img_gray_single_direction_3_dimensional_coordinate_reconstruction_scan(img_gray, SEG_DIRECTION_COL)
    coordinate_result_map_col = coordinate_result_map_col_info[0]
    coordinate_result_map_row = coordinate_result_map_row_info[0]
    angle_transfer_map_col = coordinate_result_map_col_info[1]
    angle_transfer_map_row = coordinate_result_map_row_info[1]
    #coordinate_result_map = coordinate_result_map_row + coordinate_result_map_col
    coordinate_result_map_info = coordinate_col_row_combine(coordinate_result_map_row, coordinate_result_map_col)
    coordinate_result_map = coordinate_result_map_info[0]
    angle_transfer_map_info = angle_transfer_map_col_row_combine(angle_transfer_map_row, angle_transfer_map_col, coordinate_result_map, img_gray)
    angle_transfer_map = angle_transfer_map_info[0]
    return [coordinate_result_map, angle_transfer_map]


def coordinate_col_row_combine(coordinate_result_map_row, coordinate_result_map_col):
    #coordinate_result_map = np.fmin(coordinate_result_map_col, coordinate_result_map_row) #coordinate_result_map_row + coordinate_result_map_col
    #coordinate_result_map = np.fmax(coordinate_result_map_col, coordinate_result_map_row)
    coordinate_result_map = coordinate_result_map_col + coordinate_result_map_row
    #coordinate_result_map = coordinate_result_map_col
    #coordinate_result_map = coordinate_result_map_row
    return [coordinate_result_map]


def angle_transfer_map_col_row_combine(angle_transfer_map_row, angle_transfer_map_col, coordinate_result_map, img_gray):
    coordinate_result_mask = np.abs(np.sign(coordinate_result_map[:, :, 2]))
    #angle_transfer_map = np.fmax(angle_transfer_map_col, angle_transfer_map_row)
    #angle_transfer_map = angle_transfer_map_col#angle_transfer_map_row
    angle_transfer_map = np.sign(angle_transfer_map_col+angle_transfer_map_row)#*img_gray
    #angle_transfer_map = np.round(((coordinate_result_mask * angle_transfer_map_col)+(coordinate_result_mask*angle_transfer_map_row))/2)
    #angle_transfer_map = angle_transfer_map_col#angle_transfer_map_col
    return [angle_transfer_map]


def img_gray_single_direction_3_dimensional_coordinate_reconstruction_scan(img_gray, seg_direction):
    img_seg_extractor = get_img_gray_seg_extract_fun(seg_direction)
    complete_coordinate_generator = get_complete_coordinate_generator(seg_direction)
    img_seg_num_info = get_img_gray_seg_num(img_gray, seg_direction)
    img_seg_num = img_seg_num_info[0]
    img_seg_len = img_seg_num_info[1]
    img_shape = img_gray.shape
    coordinate_result_map = np.zeros((img_shape[0], img_shape[1], 3))
    angle_transfer_map = np.zeros(img_shape)
    coordinate_result_map_seg_extractor = get_img_gray_seg_extract_fun_3d(seg_direction)
    i = 0
    while i < img_seg_num:
        if i == 405:
            i = 405
        img_seg = img_seg_extractor(img_gray, i)
        coordinate_result_seg = coordinate_result_map_seg_extractor(coordinate_result_map, i)
        angle_transfer_seg = img_seg_extractor(angle_transfer_map, i)
        img_gray_single_seg_3_dimensional_coordinate_reconstruction_scan(img_seg, img_seg_len, complete_coordinate_generator, coordinate_result_seg, angle_transfer_seg)
        i += 1
    return [coordinate_result_map, angle_transfer_map]


def img_gray_single_seg_3_dimensional_coordinate_reconstruction_scan(img_seg, seg_len, complete_coordinate_generator, coordinate_result_seg, angle_transfer_pixel_seg):
    scan_angle = 0
    scan_vector = np.array([COORDINATE_BASIC_STEP, 0])
    scan_coordinate = np.array([0, 0])
    scan_complete_coordinate = complete_coordinate_generator(scan_coordinate)
    coordinate_result_seg[0, :] = scan_complete_coordinate
    scan_coordinate = increment_scan_coordinate(scan_coordinate, scan_vector)
    coordinate_result_seg[1, :] = complete_coordinate_generator(scan_coordinate)
    scan_coordinate = increment_scan_coordinate(scan_coordinate, scan_vector)
    #coordinate_result_seg[2, :] = complete_coordinate_generator(scan_coordinate)
    #seg_triple_pixel_info = gs_gen.seg_triple_pixel_monotonic_analysis(img_seg, 2)
    current_strict_monotonic_state = False
    current_rho = COORDINATE_BASIC_STEP#0.01
    current_total_abs_delta = COORDINATE_INITIAL_TOTAL_ABS_DELTA#seg_triple_pixel_info[gs_gen.TRIPLE_PIXEL_ABS_DELTA_SUM_IDX]
    current_grad_direction = ANGLE_GRAD_STEADY
    is_filter_out = False
    is_previous_seg_simple_vibrate_flat = False
    is_previous_filter_out = True
    previous_curve_type = gs_gen.TRIPLE_PIXEL_GRAD_MONOTONIC_SIMPLE_FLAT
    filtered_out_pixel_intensity_val = 0
    i = 2
    while i < seg_len:
        if i == 1060:
            i = 1060
        seg_triple_pixel_info = gs_gen.seg_triple_pixel_monotonic_analysis(img_seg, i)
        scan_vector_info = get_img_gray_z_aix_angle_transfer_vector(seg_triple_pixel_info, scan_angle, current_strict_monotonic_state, current_rho, current_total_abs_delta, current_grad_direction, previous_curve_type)
        scan_vector = scan_vector_info[0]
        scan_angle = scan_vector_info[1]
        current_strict_monotonic_state = scan_vector_info[2]
        current_rho = scan_vector_info[3]
        current_total_abs_delta = scan_vector_info[4]
        current_grad_direction = scan_vector_info[5]
        is_filter_out = scan_vector_info[6]
        previous_curve_type = scan_vector_info[7]
        scan_coordinate = increment_scan_coordinate(scan_coordinate, scan_vector)
        coordinate_result_seg[i, :] = complete_coordinate_generator(scan_coordinate)
        if not is_filter_out and (previous_curve_type != gs_gen.TRIPLE_PIXEL_GRAD_MONOTONIC_SIMPLE_VIBRATE_FLAT or is_previous_filter_out): #or (seg_triple_pixel_info[gs_gen.TRIPLE_PIXEL_GRAD_MONOTONIC_TYPE_IDX] == gs_gen.TRIPLE_PIXEL_GRAD_MONOTONIC_SIMPLE_VIBRATE_FLAT and not is_previous_seg_simple_vibrate_flat): #== gs_gen.TRIPLE_PIXEL_GRAD_MONOTONIC_SIMPLE:
            fill_angle_transfer_pixel_seg(seg_triple_pixel_info, angle_transfer_pixel_seg, i)
            #filtered_out_pixel_intensity_val = (filtered_out_pixel_intensity_val + 1) % 2
        else:
            angle_transfer_pixel_seg[i] = filtered_out_pixel_intensity_val
            #angle_transfer_pixel_seg[i-1] = 1
            #is_previous_seg_simple_vibrate_flat = True
        #elif seg_triple_pixel_info[gs_gen.TRIPLE_PIXEL_GRAD_MONOTONIC_TYPE_IDX] == gs_gen.TRIPLE_PIXEL_GRAD_MONOTONIC_SIMPLE_VIBRATE_FLAT and not is_previous_seg_simple_vibrate_flat:
            #angle_transfer_pixel_seg[i - 1] = 1
            #is_previous_seg_simple_vibrate_flat = True
        #elif seg_triple_pixel_info[gs_gen.TRIPLE_PIXEL_GRAD_MONOTONIC_TYPE_IDX] == gs_gen.TRIPLE_PIXEL_GRAD_MONOTONIC_SIMPLE_VIBRATE_FLAT:
            #is_previous_seg_simple_vibrate_flat = False
        is_previous_filter_out = is_filter_out
        i += 1
    return coordinate_result_seg


def img_gray_intensity_val_extract(img_gray, intensity_val, intensity_map):
    extracted_img_gray = img_gray - intensity_val
    extracted_img_gray = np.abs(np.sign(extracted_img_gray))
    extracted_img_gray = np.abs(extracted_img_gray - 1)
    extracted_img_gray = extracted_img_gray#*intensity_map
    return extracted_img_gray


def img_gray_extract_sign_value(img_gray, img_value_map=None, extract_sign=IMG_EXTRACT_POS):
    img_extracted = np.sign(img_gray)
    if extract_sign == IMG_EXTRACT_NEG:
        img_extracted = img_extracted*-1
    img_extracted = (img_extracted + 1)/2
    if img_value_map is not None:
        img_value = img_value_map
    else:
        img_value = img_gray
    img_extracted = img_extracted #* img_value
    return img_extracted


def fill_angle_transfer_pixel_seg(seg_triple_pixel_info, angle_transfer_pixel_seg, top_idx):
    abs_delta_1 = seg_triple_pixel_info[gs_gen.TRIPLE_PIXEL_ABS_DELTA_1_IDX]
    abs_delta_2 = seg_triple_pixel_info[gs_gen.TRIPLE_PIXEL_ABS_DELTA_2_IDX]
    if abs_delta_1 > abs_delta_2:
        angle_transfer_pixel_seg[top_idx-1] = 1
    else:
        angle_transfer_pixel_seg[top_idx] = 1
    return angle_transfer_pixel_seg


def get_img_gray_seg_normalized_std(img_gray_seg):
    img_seg_shape = img_gray_seg.shape
    img_seg_avg = np.average(img_gray_seg)
    img_seg_normalized = img_gray_seg - img_seg_avg
    img_seg_std_normalized = np.std(img_seg_normalized)
    img_gray_seg_normalized_std = np.zeros(img_seg_shape)
    img_gray_seg_normalized_std[np.int_(img_seg_shape[0]/2), np.int_(img_seg_shape[1]/2)] = img_seg_std_normalized
    return img_gray_seg_normalized_std


def get_img_gray_seg_dist_avg_metric(img_gray_seg):
    img_seg_shape = img_gray_seg.shape
    img_gray_seg_intensity_distance_info = get_img_gray_average_intensity_distance(img_gray_seg)
    avg_distance = img_gray_seg_intensity_distance_info[IMG_SEG_DISTANCE_AVG_DISTANCE_IDX]
    std_distance = img_gray_seg_intensity_distance_info[IMG_SEG_DISTANCE_STD_DISTANCE_IDX]
    max_distance = img_gray_seg_intensity_distance_info[IMG_SEG_DISTANCE_MAX_DISTANCE_IDX]
    seg_distance_metric_info = img_gray_distance_metric_avg_std_ratio(avg_distance, std_distance, max_distance)
    seg_distance_metric = seg_distance_metric_info[0]
    img_gray_seg_distance = np.zeros(img_seg_shape)
    img_gray_seg_distance[np.int_(img_seg_shape[0] / 2), np.int_(img_seg_shape[1] / 2)] = seg_distance_metric
    return img_gray_seg_distance


def get_img_gray_binary_one_zero_ratio_metric(img_gray_binary_seg):
    img_seg_shape = img_gray_binary_seg.shape
    one_zero_ratio_info = get_img_gray_binary_one_zero_ratio(img_gray_binary_seg)
    one_zero_ratio = one_zero_ratio_info[0]
    img_gray_binary_seg_one_zero_ratio = np.zeros(img_seg_shape)
    img_gray_binary_seg_one_zero_ratio[np.int_(img_seg_shape[0] / 2), np.int_(img_seg_shape[1] / 2)] = one_zero_ratio
    if one_zero_ratio > 0:
        one_zero_ratio = 1
    return img_gray_binary_seg_one_zero_ratio


def img_gray_distance_metric_avg_std_ratio(avg_distance, std_distance, max_distance):
    if std_distance > 0:
        seg_distance_metric = (max_distance - avg_distance)/std_distance
    else:
        seg_distance_metric = 0
    return [seg_distance_metric]


def get_img_seg_distance_metric_array(img_gray_distance_metric_seg, fragment_shape=(10, 10)):
    distance_metric_array = img_gray_distance_metric_seg[img_gray_distance_metric_seg > 0]
    distance_metric_nonzero_coordinate_info = np.nonzero(img_gray_distance_metric_seg)
    distance_nonzero_fragment_seg_info = nonzero_coordinate_to_fragment_seg_info(distance_metric_nonzero_coordinate_info, fragment_shape)
    distance_nonzero_fragment_seg_list = distance_nonzero_fragment_seg_info[0]
    if len(distance_metric_array) == 0:
        is_trivial_seg = True
    else:
        is_trivial_seg = False
    return [distance_metric_array, is_trivial_seg, distance_nonzero_fragment_seg_list]


def nonzero_coordinate_to_fragment_seg_info(distance_metric_nonzero_coordinate_info, fragment_shape):
    row_aix_array = distance_metric_nonzero_coordinate_info[0]
    col_aix_array = distance_metric_nonzero_coordinate_info[1]
    aix_array_len = len(row_aix_array)
    fragment_row_num = fragment_shape[0]
    fragment_col_num = fragment_shape[1]
    row_start_offset = np.int_(np.floor(fragment_row_num/2))*-1
    col_start_offset =  np.int_(np.floor(fragment_col_num/2))*-1
    row_end_offset = np.int_(np.ceil(fragment_row_num/2)) + 1
    col_end_offset = np.int_(np.ceil(fragment_col_num/2)) + 1
    fragment_seg_info_list = []
    i = 0
    while i < aix_array_len:
        seg_row_center = row_aix_array[i]
        seg_col_center = col_aix_array[i]
        seg_row_start = seg_row_center + row_start_offset
        seg_col_start = seg_col_center + col_start_offset
        seg_row_end = seg_row_center + row_end_offset
        seg_col_end = seg_col_center + col_end_offset
        fragment_seg_info_list.append([seg_row_start, seg_row_end, seg_col_start, seg_col_end])
        i += 1
    return [fragment_seg_info_list]


def get_img_gray_intensity_plane(img_gray, intenity_level, is_binary_plane=IMG_PLANE_BINARY, plane_kept_nonzero_ratio_threshold=PLANE_KEPT_ANY):
    img_gray_intensity_plane_list = []
    img_gray_nonzero_ratio_list = []
    img_gray_0 = img_gray
    img_size = img_gray_0.size
    i = 0
    img_max = np.max(img_gray_0)
    while i < intenity_level and img_max > 0:
        img_gray_intensity_plane = np.sign(img_gray_0 - img_max)*img_gray_0 + img_gray_0
        #img_gray_intensity_plane = np.ceil(img_gray_intensity_plane)
        img_nonzero_ratio = np.count_nonzero(img_gray_intensity_plane)/img_size
        if plane_kept_nonzero_ratio_threshold <= 0 or img_nonzero_ratio >= plane_kept_nonzero_ratio_threshold:
            if is_binary_plane == IMG_PLANE_BINARY:
                img_gray_intensity_plane_list.append(np.ceil(img_gray_intensity_plane))
            else:
                img_gray_intensity_plane_list.append(img_gray_intensity_plane)
        img_gray_nonzero_ratio_list.append(img_nonzero_ratio)
        img_gray_0 = img_gray_0 - img_gray_intensity_plane
        img_max = np.max(img_gray_0)
        i += 1
    return [img_gray_intensity_plane_list, img_gray_nonzero_ratio_list]

'''
def get_img_gray_nonzero_count(img_gray):
    img_shape = img_gray.shape
    img_pixel_num = img_shape[0]*img_shape[1]
    pixel_count_mask = (np.ones(img_pixel_num)).transpose()
    img_gray_array = img_gray.reshape(img_pixel_num)
    img_nonzero_count = img_gray_array@pixel_count_mask
    return img_nonzero_count
'''


def extract_img_gray_info_plane(img_gray, img_gray_plane, distance_seg_num, img_gray_plane_list=None):
    global DISTANCE_SEG_NUM
    if img_gray_plane_list is None:
        img_gray_plane_list = []
    old_distance_seg_num = DISTANCE_SEG_NUM
    DISTANCE_SEG_NUM = distance_seg_num
    img_gray_plane_seg = get_img_gray_poisson_seg(img_gray_plane)
    img_gray_intensity_plane_list_info = get_img_gray_intensity_plane(img_gray_plane_seg, distance_seg_num, IMG_PLANE_BINARY)
    local_img_gray_intensity_plane_list = img_gray_intensity_plane_list_info[0]
    local_img_gray_intensity_plane_nonzero_ratio_list = img_gray_intensity_plane_list_info[1]
    local_intensity_plane_list_len = len(local_img_gray_intensity_plane_list)
    i = 0
    while i < local_intensity_plane_list_len:
        local_img_gray_plane_nonzero_ratio = local_img_gray_intensity_plane_nonzero_ratio_list[i]
        local_img_gray_intensity_plane = local_img_gray_intensity_plane_list[i]
        if True or local_img_gray_plane_nonzero_ratio <= SEG_IMG_PLANE_EXTRACT_NONZERO_RATIO_THRESHOLD or local_intensity_plane_list_len == 1 or distance_seg_num < 2:
            #if local_img_gray_plane_nonzero_ratio >= 0.001:#0.038: #and local_img_gray_plane_nonzero_ratio <= 0.2:
            img_gray_plane_list.append(local_img_gray_intensity_plane)
        else:
            extract_img_gray_info_plane(img_gray, local_img_gray_intensity_plane*img_gray, distance_seg_num/2, img_gray_plane_list)
        i += 1
    DISTANCE_SEG_NUM = old_distance_seg_num
    return [img_gray_plane_list]


def display_img_gray_plane_list(img_gray_plane_list, clip=None, is_single_frame=False):
    #if clip is None:
        #list_len = len(img_gray_plane_list)
    #else:
        #list_len = len(clip)
    list_len = get_clip_controlled_list_len(img_gray_plane_list, clip)
    display_col_num = np.fmin(list_len, 4)
    display_row_num = np.int_(np.ceil(list_len/display_col_num))
    i = 1
    while i <= list_len:
        #if clip is None:
            #plane_idx = i-1
        #else:
            #plane_idx = clip[i-1]
        plane_idx = get_clip_controlled_data_idx(clip, i-1)
        if not is_single_frame:
            plt.subplot(display_row_num, display_col_num, i)
        plt.imshow(img_gray_plane_list[plane_idx], cmap='gray', vmax=1, vmin=0)
        if is_single_frame:
            plt.show()
        i += 1
    if not is_single_frame:
        plt.show()
    return


def extract_edge_from_img_gray_plane_list(img_gray_plane_list, img_shape, is_edge=True, clip=None):
    list_len = get_clip_controlled_list_len(img_gray_plane_list, clip)
    i = 0
    img_gray_edge = np.zeros(img_shape)
    while i < list_len:
        plane_idx = get_clip_controlled_data_idx(clip, i)
        img_gray_plane = img_gray_plane_list[plane_idx]
        if is_edge:
            coordinate_result_map_info = img_gray_3_dimensional_coordinate_reconstruction_scan(img_gray_plane)
            img_gray_plane = coordinate_result_map_info[1]
        img_gray_edge_plane = img_gray_plane#img_gray_plane#coordinate_result_map_info[1]
        img_gray_edge = np.sign(img_gray_edge + img_gray_edge_plane)
        i += 1
    #coordinate_result_map_info = img_gray_3_dimensional_coordinate_reconstruction_scan(img_gray_edge)
    #img_gray_edge = coordinate_result_map_info[1]
    return img_gray_edge


def extract_edge_from_img_gray_plane_list_by_one_zero_ratio_map(img_gray_plane_list, img_shape, clip=None):
    global FRAGMENT_FILTER_ALGORITHM
    list_len = get_clip_controlled_list_len(img_gray_plane_list, clip)
    i = 0
    img_gray_edge = np.zeros(img_shape)
    fragment_shape = (10, 10)
    while i < list_len:
        plane_idx = get_clip_controlled_data_idx(clip, i)
        img_gray_plane = img_gray_plane_list[plane_idx]
        old_fragment_filter_algorithm = FRAGMENT_FILTER_ALGORITHM
        FRAGMENT_FILTER_ALGORITHM = FRAGMENT_FILTER_ALGORITHM_BINARY_ONE_ZERO_RATIO_METRIC
        img_gray_fragment_distance_metric_seg = get_img_gray_poisson_seg_fragmented_overlap(img_gray_plane, fragment_shape, 1, 1)
        img_gray_filtered_mask_info = img_gray_extract_interval_threshold_operator(img_gray_fragment_distance_metric_seg, 1/5, 4/5, CMP_VAL_MASK_ONLY, CMP_EQU_EXCLUDE, CMP_EQU_EXCLUDE, CMP_VAL_ZERO_MASK)
        img_gray_metric_edge_result = img_gray_filtered_mask_info[0]
        distance_metric_info = get_img_seg_distance_metric_array(img_gray_metric_edge_result, fragment_shape)
        metric_seg_list = distance_metric_info[2]
        img_gray_metric_edge_result = get_img_gray_metric_filtered_val(img_gray_plane, metric_seg_list)
        img_gray_edge = np.sign(img_gray_edge + img_gray_metric_edge_result)
        i += 1
    #coordinate_result_map_info = img_gray_3_dimensional_coordinate_reconstruction_scan(img_gray_edge)
    #img_gray_edge = coordinate_result_map_info[1]
    FRAGMENT_FILTER_ALGORITHM = old_fragment_filter_algorithm
    return img_gray_edge


def get_img_gray_metric_filtered_val(img_gray, metric_seg_list):
    list_len = len(metric_seg_list)
    img_shape = img_gray.shape
    img_result = np.zeros(img_shape)
    i = 0
    while i < list_len:
        metric_seg = metric_seg_list[i]
        seg_row_start = metric_seg[0]
        seg_row_end = metric_seg[1]
        seg_col_start = metric_seg[2]
        seg_col_end = metric_seg[3]
        img_result[seg_row_start:seg_row_end, seg_col_start:seg_col_end] = img_gray[seg_row_start:seg_row_end, seg_col_start:seg_col_end]
        i += 1
    return img_result


def get_clip_controlled_list_len(data_list, clip):
    if clip is None:
        list_len = len(data_list)
    else:
        list_len = len(clip)
    return list_len


def get_clip_controlled_data_idx(clip, raw_idx):
    if clip is not None:
        data_idx = clip[raw_idx]
    else:
        data_idx = raw_idx
    return data_idx


def img_gray_binary_simple_inverse(img_gray_binary):
    img_gray_binary_inverse = np.abs(img_gray_binary - 1)
    return img_gray_binary_inverse


def get_img_gray_binary_one_zero_ratio(img_gray_binary):
    non_zero_count = np.count_nonzero(img_gray_binary)
    one_zero_ratio = non_zero_count/img_gray_binary.size
    if one_zero_ratio > 1:
        print("error one zero ratio larger than 1!")
    return [one_zero_ratio]


def img_gray_rectangle_mark(img_gray, rect_dim, line_intensity=1, is_copy=SEG_IMG_COPY):
    if is_copy:
        img_cp = img_copy(img_gray)
    else:
        img_cp = img_gray
    row_start = rect_dim[0]
    row_end = rect_dim[1]
    col_start = rect_dim[2]
    col_end = rect_dim[3]
    img_cp[row_start, col_start:col_end] = line_intensity
    img_cp[row_end-1, col_start:col_end] = line_intensity
    img_cp[row_start:row_end, col_start] = line_intensity
    img_cp[row_start:row_end, col_end-1] = line_intensity
    return img_cp


def img_gray_extract_interval_threshold_operator(img_gray, cmp_lower_bound_val, cmp_upper_bound_val, is_mask_only, is_lower_bound_include, is_upper_bound_include, is_zero_masked):
    img_gray_lower_bound_mask = img_gray_extract_threshold_condition_mask(img_gray, cmp_lower_bound_val, CMP_OPERATOR_LARGE, is_lower_bound_include, is_zero_masked)
    img_gray_upper_bound_mask = img_gray_extract_threshold_condition_mask(img_gray_lower_bound_mask*img_gray, cmp_upper_bound_val, CMP_OPERATOR_SMALL, is_upper_bound_include, is_zero_masked)
    img_gray_result_mask = img_gray_upper_bound_mask * img_gray_lower_bound_mask
    if not is_mask_only:
        img_gray_result = img_gray_result_mask*img_gray
    else:
        img_gray_result = img_gray_result_mask
    return [img_gray_result]


def img_gray_extract_threshold_condition_mask(img_gray, threshold_val, cmp_operator=CMP_OPERATOR_LARGE, is_equ_include=CMP_EQU_INCLUDE, is_zero_masked=CMP_VAL_ZERO_MASK):
    if is_zero_masked:
        zero_mask_inv = np.sign(img_gray)
    img_gray_result = img_gray - threshold_val
    img_gray_result = np.sign(img_gray_result)
    if cmp_operator == CMP_OPERATOR_LARGE:
        img_gray_result = img_gray_result + 1
    else:
        img_gray_result = np.abs(img_gray_result - 1)
    img_gray_result = img_gray_result/2
    if is_equ_include:
        img_gray_result = np.ceil(img_gray_result)
    else:
        img_gray_result = np.floor(img_gray_result)
    if is_zero_masked:
        img_gray_result = zero_mask_inv*img_gray_result
    return img_gray_result


def img_gray_otsu_threshold_segmentation(img_gray):
    if np.max(img_gray) <= 1:
        img_gray_cp = (img_gray*255).astype('u1')
    else:
        img_gray_cp = img_gray
    thresh = cv2.threshold(img_gray_cp, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #img_otsu = thresh[1]
    return thresh[1], thresh[0]


def img_gray_find_contour_count(img_gray):
    if np.max(img_gray) <= 1:
        img_gray_cp = (img_gray * 255).astype('u1')
    else:
        img_gray_cp = img_gray
    img_contour_info = cv2.findContours(img_gray_cp.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    img_contour = imutils.grab_contours(img_contour_info)
    img_contour_len = len(img_contour)
    return [img_contour_len, img_contour]


def img_draw_contour(img, contour):
    j = 1
    for (i, c) in enumerate(contour):
        ((x, y), _) = cv2.minEnclosingCircle(c)
        area = cv2.contourArea(c)
        print(area)
        if area > 1000:
            cv2.putText(img, "#{}".format(j), (int(x) - 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255),2)
            cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
            j += 1
    return img


def img_gray_range_extract(img_gray):
    img_cp = img_copy(img_gray)
    img_shape = img_cp.shape
    img_size = img_shape[0]*img_shape[1]
    img_array = img_cp.reshape(img_size)
    img_array = np.abs(np.floor(np.sort(img_array*-255)))
    range_list = []
    range_start = 0
    range_end = 0
    while range_end < img_size:
        range_end = img_array_extract_single_range(img_array, range_start, img_size, range_list)
        range_start = range_end
    return range_list


def img_array_extract_single_range(sorted_img_array, range_start, img_size, range_list):
    i = range_start + 1
    range_val = sorted_img_array[range_start]
    r = 1/255
    while i < img_size:
        if sorted_img_array[i] - sorted_img_array[i-1] != 0:
            break
        i += 1
    range_count = i - range_start
    range_list.append([range_val, range_count])
    range_end = i
    return range_end


def img_generate_range_stair_list(img_range_list):
    range_stair_list = []
    range_list_len = len(img_range_list)
    range_stair_prev = img_calculate_range_stair(img_range_list[0], img_range_list[1])
    range_stair_list.append(range_stair_prev)
    i = 2
    while i < range_list_len:
        range_stair = img_calculate_range_stair(img_range_list[i-1], img_range_list[i])
        range_stair_prev[IMG_RANGE_STAIR_RANGE_NEXT_REF_IDX] = range_stair
        range_stair[IMG_RANGE_STAIR_RANGE_PREV_REF_IDX] = range_stair_prev
        range_stair_list.append(range_stair)
        range_stair_prev = range_stair
        i += 1
    range_stair = img_generate_range_stair(img_range_list[range_list_len-1])
    range_stair[IMG_RANGE_STAIR_RANGE_PREV_REF_IDX] = range_stair_prev
    range_stair[IMG_RANGE_STAIR_STAIR_VAL_IDX] = 0
    range_stair_list.append(range_stair)
    return range_stair_list


def img_calculate_range_stair(img_range_high, img_range_low):
    img_range_stair = img_generate_range_stair(img_range_high)
    range_stair_val = img_range_high[IMG_RANGE_INT_VAL_IDX] - img_range_low[IMG_RANGE_INT_VAL_IDX]
    img_range_high_count = img_range_high[IMG_RANGE_COUNT_IDX]
    img_range_low_count = img_range_low[IMG_RANGE_COUNT_IDX]
    range_count_ratio = np.abs(img_range_high_count-img_range_low_count)/(img_range_high_count+img_range_low_count)
    img_range_stair[IMG_RANGE_STAIR_STAIR_VAL_IDX] = range_stair_val
    img_range_stair[IMG_RANGE_STAIR_STAIR_COUNT_RATIO_IDX] = range_count_ratio
    return img_range_stair


def img_generate_range_stair(img_range):
    img_range_stair = [None]*6
    img_range_stair[IMG_RANGE_STAIR_RANGE_VAL_IDX] = img_range[IMG_RANGE_INT_VAL_IDX]
    img_range_stair[IMG_RANGE_STAIR_STATUS_IDX] = IMG_RANGE_STAIR_STATUS_VALID
    return img_range_stair


def img_hash_range_stair_list(img_range_stair_list):
    list_len = len(img_range_stair_list)
    range_stair_dict = get_empty_dict()
    i = 0
    while i < list_len:
        range_stair_item = img_range_stair_list[i]
        hash_key = range_stair_item[IMG_RANGE_STAIR_STAIR_VAL_IDX]
        dict_item = get_dict_item_list(range_stair_dict, hash_key)
        dict_item.append(range_stair_item)
        i += 1
    return range_stair_dict


def get_empty_dict():
    empty_d = {}
    return empty_d


def get_dict_item_list(input_dict, key):
    dict_item = input_dict.get(key)
    if dict_item is None:
        dict_item = []
        input_dict[key] = dict_item
    return dict_item


def img_gray_fill_rect(img_gray, rect_row_start, rect_row_end, rect_col_start, rec_col_end, intensity_val, is_copy=SEG_IMG_COPY):
    if is_copy:
        img_cp = img_copy(img_gray)
    else:
        img_cp = img_gray
    img_cp[rect_row_start:rect_row_end, rect_col_start:rec_col_end] = intensity_val
    return img_cp


def create_relative_gray_level_image(img_dim, gray_level_division_ratio_x, gray_level_division_ratio_y, compare_gray_level, standard_gray_level=GRAY_LEVEL_COMPARE_STANDARD):
    img_res = get_img_gray_identical_intensity(img_dim, standard_gray_level)
    img_width = img_dim[0]
    img_hight = img_dim[1]
    gray_level_division_start_ratio_x = 1 - gray_level_division_ratio_x
    gray_level_division_start_ratio_y = 1 - gray_level_division_ratio_y
    compare_gray_level_start_x = np.int_(img_width * gray_level_division_start_ratio_x)
    compare_gray_level_start_y = np.int_(img_hight * gray_level_division_start_ratio_y)
    img_res[compare_gray_level_start_x:img_width, compare_gray_level_start_y:img_hight] = compare_gray_level
    return img_res


def generate_sinusodial_grating_img(img_line_num, cycle_num, x_unit, f, theta, l_0, m):
    scan_line_info = generate_sinusodial_scan_line_cycle(cycle_num, x_unit, f, theta, l_0, m)
    scan_line = scan_line_info[0]
    img_col_num = scan_line_info[1]
    img_shape = (img_line_num, img_col_num)
    img_sinusodial_grating = np.zeros(img_shape)
    i = 0
    while i < img_line_num:
        img_sinusodial_grating[i, 0:img_col_num] = scan_line
        i += 1
    return[img_sinusodial_grating, img_shape]


def generate_sinusodial_scan_line_cycle(cycle_num, x_unit, f, theta, l_0, m):
    tpi_f_len = 2*np.pi/f
    all_cycle_len = tpi_f_len*cycle_num
    x_len = np.int_(np.ceil(all_cycle_len/x_unit))
    scan_line_info = generate_sinusodial_scan_line(x_len, x_unit, f, theta, l_0, m)
    return [scan_line_info[0], x_len]


def generate_sinusodial_scan_line(x_len, x_unit, f, theta, l_0, m, tirangular_fun=np.cos):
    scan_line = np.zeros(x_len)
    i = 0
    x = 0
    tpi_f = f#2*np.pi*f
    while i < x_len:
        cos_val = tirangular_fun(tpi_f*x+theta)
        weighted_cos_val = (m/l_0)*cos_val
        scan_line[i] = l_0*(1+weighted_cos_val)
        i += 1
        x += x_unit
    #if l_0 > 0:
        #scan_line = (scan_line-np.min(scan_line))/(np.max(scan_line)-np.min(scan_line))
    return [scan_line]


def and_binary_tensor(tensor1, tensor2):
    res_tensor = tensor2*tensor1
    return res_tensor


def img_plane_inverse_aggregate(img_plane_list, plane_num, aggregate_plane_num):
    img_gray_plane_np = np.array(img_plane_list[plane_num - aggregate_plane_num:plane_num])
    img_gray_aggregated = np.sum(img_gray_plane_np, axis=0)
    return img_gray_aggregated


def img_plane_dist_avg(img_dist_list, plane_pixel_num_list, plane_num):
    plane_step = plane_num - 1
    if plane_step >= 0:
        plane_pixel_idx_start = np.sum(plane_pixel_num_list[0:plane_step])
    else:
        plane_pixel_idx_start = 0
    plane_pixel_num = plane_pixel_num_list[plane_num]
    img_plane_dist_avg_val = np.sum(img_dist_list[plane_pixel_idx_start:plane_pixel_num])/plane_pixel_num
    return img_plane_dist_avg_val


def abs_one_dim_array_div(source_array, is_abs=True):
    source_array_size = source_array.size
    array_dim = source_array[1:source_array_size] - source_array[0:source_array_size-1]
    if is_abs:
        array_dim = np.abs(array_dim)
    return array_dim


def one_dim_array_bin_mean(source_array):
    source_array_size = source_array.size
    bin_mean = (source_array[0:source_array_size-1]+source_array[1:source_array_size])/2
    return bin_mean


def one_dim_min_max_normalization(source_array):
    source_array_max = np.max(source_array)
    source_array_min = np.min(source_array)
    norm_source_array = (source_array - source_array_min)/(source_array_max-source_array_min)
    return norm_source_array


def one_dim_mean_low_high_count_ratio(source_array):
    array_mean = np.median(source_array)
    array_mean_low_count = np.sum(np.where(source_array <= array_mean, source_array, 0))
    array_mean_high_count = np.sum(np.where(source_array > array_mean, source_array, 0))
    array_mean_low_high_count_ratio = array_mean_low_count/(array_mean_high_count + array_mean_low_count)
    return array_mean_low_high_count_ratio


def one_dim_none_zero_count(source_array):
    array_none_zero_indicator = np.where(source_array > 0, 1, 0)
    array_none_zero_count = np.sum(array_none_zero_indicator)
    return array_none_zero_count


def one_dim_single_threshold_count(source_array, threshold, comparator, is_equal=False):
    res_count = np.array([-1])
    if comparator == '>':
        if is_equal:
            res_count = np.where(source_array >= threshold, 1, 0)
        else:
            res_count = np.where(source_array > threshold, 1, 0)
    elif comparator == '<':
        if is_equal:
            res_count = np.where(source_array <= threshold, 1, 0)
        else:
            res_count = np.where(source_array < threshold, 1, 0)
    res_count = np.sum(res_count)
    return res_count


def generate_img_none_zero_section_ratio_second_order_div_sorted_log_curve(img_dist_none_zero_section_size_list):
    abs_div_img_dist_none_zero_section_size_list = abs_one_dim_array_div(img_dist_none_zero_section_size_list)
    abs_div_img_dist_none_zero_section_ratio_list = abs_div_img_dist_none_zero_section_size_list / np.max(img_dist_none_zero_section_size_list)

    sorted_div_ratio_list = np.sort(abs_div_img_dist_none_zero_section_ratio_list)
    second_order_div_ratio_list = abs_one_dim_array_div(sorted_div_ratio_list)
    sorted_second_order_div_ratio_list = np.sort(second_order_div_ratio_list)
    sorted_second_order_div_ratio_list = one_dim_get_none_zero_value_list(sorted_second_order_div_ratio_list)

    div_ratio_list_size = sorted_second_order_div_ratio_list.size
    curve_x = np.array(range(div_ratio_list_size)) / div_ratio_list_size

    sorted_second_order_div_ratio_list = np.where(sorted_second_order_div_ratio_list == 0, 1e-10, sorted_second_order_div_ratio_list)
    curve_y_log = np.log(sorted_second_order_div_ratio_list)
    return [curve_x, curve_y_log, sorted_second_order_div_ratio_list]


def img_intensity_precision_proc(img_intensity):
    precision_controlled_intensity = np.round(img_intensity, IMAGE_ROUND_PRECISION)
    return precision_controlled_intensity


def one_dim_get_none_zero_value_list(source_array):
    none_zero_idx = np.argwhere(source_array >= 1e-7)
    none_zero_source_array = source_array[none_zero_idx]
    return none_zero_source_array


def img_intensity_plane_num(img):
    intensity_level_range = np.max(img) - np.min(img)
    intensity_level_range = np.around(intensity_level_range * INT_GRAY_LEVEL_BAR)
    intensity_plane_num = np.int_(intensity_level_range + 1)
    return intensity_plane_num


def img_none_zero_segmentation_plane_section_num(img_unique_value, seg_num):
    img_unique_value_size = img_unique_value.size
    section_num = np.int_(img_unique_value_size/seg_num)
    return section_num


def img_plane_whole_range_section_num(img, seg_num):
    intensity_plane_num = img_intensity_plane_num(img)
    section_num = np.int_(np.around(intensity_plane_num / seg_num))
    if section_num == 0:
        section_num = 1
    return section_num, intensity_plane_num


def img_sorted_array(img):
    img_shape = img.shape
    img_size = img_shape[0] * img_shape[1]
    img_sorted = img.reshape(img_size)
    img_sorted = np.sort(img_sorted)
    return img_sorted


def img_unique_intensity_value_sorted(img):
    img_unique_value = np.unique(img)
    img_unique_value = np.sort(img_unique_value)[::-1]
    img_unique_value_size = img_unique_value.size
    unique_value_index_dict = dict()
    for i in range(img_unique_value_size):
        key = img_unique_value[i]
        key = str(img_intensity_precision_proc(key))
        unique_value_index_dict[str(key)] = i
    return img_unique_value, unique_value_index_dict


def img_unique_value_sorted_threshold_first_low_search(img_unique_value_sorted, search_pre_start_idx, pre_threshold_base):
    i = search_pre_start_idx + 1
    img_unique_value_sorted_size = img_unique_value_sorted.size
    #first_low_value = None #The cost of parameter passing in Python is high.
    first_low_idx = -1
    rounded_threshold = img_intensity_precision_proc(pre_threshold_base)
    while i < img_unique_value_sorted_size and img_intensity_precision_proc(img_unique_value_sorted[i]) > rounded_threshold:
        i += 1
    if i < img_unique_value_sorted_size:
        #first_low_value = img_unique_value_sorted[i]
        first_low_idx = i
    return first_low_idx


def img_get_segment_threshold_base(img_unique_value_sorted, search_pre_start_idx, pre_threshold_base, img_threshold_step_value):
    threshold_base_res = None
    threshold_first_low_idx = img_unique_value_sorted_threshold_first_low_search(img_unique_value_sorted, search_pre_start_idx, pre_threshold_base)
    if threshold_first_low_idx != -1:
        threshold_base_res = img_unique_value_sorted[threshold_first_low_idx]
        threshold_base_res = np.int_(np.abs(threshold_base_res - pre_threshold_base)/np.abs(img_threshold_step_value))
        threshold_base_res = pre_threshold_base + threshold_base_res*img_threshold_step_value
    return threshold_base_res, threshold_first_low_idx


def img_get_segment_threshold(img_unique_value_sorted, unique_value_index_dict, img_search_pre_start_idx, img_pre_threshold_base, img_seg_step_value, img_single_step_value):
    img_threshold_value = img_pre_threshold_base + img_single_step_value
    img_threshold_step_value = img_seg_step_value + img_single_step_value
    img_current_threshold_base, img_current_pre_start_index = img_get_segment_threshold_base(img_unique_value_sorted, img_search_pre_start_idx, img_threshold_value, img_threshold_step_value)
    if img_current_threshold_base is not None:
        current_threshold = img_current_threshold_base + img_seg_step_value
        unique_value_key = str(img_intensity_precision_proc(current_threshold))
        unique_value_idx = unique_value_index_dict.get(unique_value_key)
        #if current_threshold < 0 and np.max():
            #current_threshold = None
        #else:
        if unique_value_idx is not None:
            img_current_pre_start_index = unique_value_idx
    else:
        current_threshold = None
    return current_threshold, img_current_pre_start_index


def img_segmentation_plane_extraction(seg_residual_img, threshold):
    precision_proc_img = img_intensity_precision_proc(seg_residual_img)
    precision_proc_threshold = img_intensity_precision_proc(threshold)
    segmentation_img_plane = np.where(precision_proc_img >= precision_proc_threshold, 1, 0)
    segmentation_img_plane_arg = np.argwhere(precision_proc_img >= precision_proc_threshold)
    current_seg_residual = seg_residual_img - seg_residual_img * segmentation_img_plane * 100
    return segmentation_img_plane, current_seg_residual, segmentation_img_plane_arg


def img_single_channel_integer_threshold_segmentation(img, seg_num):
    img_unique_value_sorted, unique_value_index_dict = img_unique_intensity_value_sorted(img)
    img_single_step_value = -IMG_SINGLE_STEP_VALUE_ABS
    plane_section_num, intensity_plane_num = img_plane_whole_range_section_num(img, seg_num)  #img_none_zero_segmentation_plane_section_num(img_unique_value_sorted, seg_num)
    img_seg_step_value = (plane_section_num-1) * img_single_step_value
    img_current_intensity_idx = 0
    img_current_intensity = img_unique_value_sorted[img_current_intensity_idx]
    img_current_threshold = img_current_intensity + img_seg_step_value
    segmentation_img_plane_list = []
    segmentation_img_plane_arg_list = []
    img_threshold_list = []
    current_seg_residual = img

    segmentation_img_plane, current_seg_residual, segmentation_img_plane_arg = img_segmentation_plane_extraction(current_seg_residual, img_current_threshold)
    segmentation_img_plane_list.append(segmentation_img_plane)
    segmentation_img_plane_arg_list.append(segmentation_img_plane_arg)
    img_threshold_list.append(img_current_threshold)
    img_current_threshold, img_current_intensity_idx = img_get_segment_threshold(img_unique_value_sorted, unique_value_index_dict, img_current_intensity_idx, img_current_threshold, img_seg_step_value, img_single_step_value)
    seg_count = seg_num - 2

    while img_current_threshold is not None:
        img_threshold_list.append(img_current_threshold)
        segmentation_img_plane, current_seg_residual, segmentation_img_plane_arg = img_segmentation_plane_extraction(current_seg_residual, img_current_threshold)
        if seg_count >= 0:
            segmentation_img_plane_list.append(segmentation_img_plane)
            segmentation_img_plane_arg_list.append(segmentation_img_plane_arg)
        else:
            segmentation_img_plane_list[seg_num-1] = segmentation_img_plane_list[seg_num-1] + segmentation_img_plane
        if np.max(current_seg_residual) < 0:#This condition will be used temporarily. It will be changed to section number count later to increase performance.
            break
        img_current_threshold, img_current_intensity_idx = img_get_segment_threshold(img_unique_value_sorted, unique_value_index_dict, img_current_intensity_idx, img_current_threshold, img_seg_step_value, img_single_step_value)
        seg_count -= 1
    img_threshold_list = np.array(img_threshold_list)
    img_abs_div_threshold_list = abs_one_dim_array_div(img_threshold_list, False)
    return segmentation_img_plane_list, img_threshold_list, img_abs_div_threshold_list, plane_section_num, intensity_plane_num, segmentation_img_plane_arg_list


def img_segmentation_reconstruction(segmentation_img_plane_list, img_abs_div_threshold_list, img_intensity_max, reconstruction_plane_mask=None):
    if reconstruction_plane_mask is None:
        reconstruction_plane_mask = RECONSTRUCTION_PLANE_MASK_EMPTY
    segmentation_img_plane_list_size = len(segmentation_img_plane_list)
    img_intensity = img_intensity_max
    if reconstruction_plane_mask.get(str(0)) is None:
        img_reconstructed = segmentation_img_plane_list[0] * img_intensity
    else:
        img_reconstructed = np.zeros(segmentation_img_plane_list[0].shape)
    i = 1
    img_intensity_update = img_intensity
    while i < segmentation_img_plane_list_size:
        #if reconstruction_plane_mask.get(str(i)) is None:
        img_intensity = img_intensity + img_abs_div_threshold_list[i-1]
        if img_intensity < 0:
            img_intensity = 0
        if reconstruction_plane_mask.get(str(i)) is None:
            img_intensity_update = img_intensity
        img_reconstructed = img_reconstructed + segmentation_img_plane_list[i]*img_intensity_update
        i += 1
    return img_reconstructed


def img_segmentation_reconstruction(segmentation_img_plane_list, img_abs_div_threshold_list, img_intensity_max, reconstruction_plane_mask=None):
    if reconstruction_plane_mask is None:
        reconstruction_plane_mask = RECONSTRUCTION_PLANE_MASK_EMPTY
    segmentation_img_plane_list_size = len(segmentation_img_plane_list)
    img_intensity = img_intensity_max
    if reconstruction_plane_mask.get(str(0)) is None:
        img_reconstructed = segmentation_img_plane_list[0] * img_intensity
    else:
        img_reconstructed = np.zeros(segmentation_img_plane_list[0].shape)
    i = 1
    img_intensity_update = img_intensity
    while i < segmentation_img_plane_list_size:
        #if reconstruction_plane_mask.get(str(i)) is None:
        img_intensity = img_intensity + img_abs_div_threshold_list[i-1]
        if img_intensity < 0:
            img_intensity = 0
        if reconstruction_plane_mask.get(str(i)) is None:
            img_intensity_update = img_intensity
        img_reconstructed = img_reconstructed + segmentation_img_plane_list[i]*img_intensity_update
        i += 1
    return img_reconstructed


def img_segmentation_reconstruction_threshold(segmentation_img_plane_list, img_threshold_list):
    segmentation_img_plane_list_size = len(segmentation_img_plane_list)
    img_intensity = img_threshold_list[0]
    img_reconstructed = segmentation_img_plane_list[0] * img_intensity
    i = 1
    while i < segmentation_img_plane_list_size:
        img_intensity = img_threshold_list[i]
        img_reconstructed = img_reconstructed + segmentation_img_plane_list[i] * img_intensity
        i += 1
    return img_reconstructed


def img_segmentation_reconstruction_seg_mean(segmentation_img_plane_list, img_original):
    segmentation_img_plane_list_size = len(segmentation_img_plane_list)
    img_original = img_original * 255
    img_reconstructed = segmentation_img_plane_list[0] * np.round(np.mean(segmentation_img_plane_list[0] * img_original), 0)/255
    i = 1
    while i < segmentation_img_plane_list_size:
        img_intensity = np.round(np.mean(segmentation_img_plane_list[i] * img_original), 0)/255
        img_reconstructed = img_reconstructed + segmentation_img_plane_list[i] * img_intensity
        i += 1
    return img_reconstructed


def img_segmentation_plane_intensity_max(segmentation_img_plane, img_original):
    img_plane_intensity = segmentation_img_plane*img_original
    plane_intensity_mean = np.max(img_plane_intensity)
    return plane_intensity_mean


def img_segmentation_reconstruction_img_plane_max(segmentation_img_plane_list, segmentation_img_plane_arg_list, img_original):
    segmentation_img_plane_list_size = len(segmentation_img_plane_list)
    segmentation_img_plane_arg = segmentation_img_plane_arg_list[0]
    plane_intensity_mean = img_segmentation_plane_intensity_max(segmentation_img_plane_list[0], img_original)
    img_reconstructed = segmentation_img_plane_list[0]*plane_intensity_mean
    i = 1
    while i < segmentation_img_plane_list_size:
        segmentation_img_plane_arg = segmentation_img_plane_arg_list[i]
        plane_intensity_mean = img_segmentation_plane_intensity_max(segmentation_img_plane_list[i], img_original)
        img_reconstructed = img_reconstructed + segmentation_img_plane_list[i] * plane_intensity_mean
        i += 1
    return img_reconstructed


def test_img_single_channel_integer_threshold_segmentation(img, seg_num):
    img_unique_value_sorted, unique_value_index_dict = img_unique_intensity_value_sorted(img)
    img_single_step_value = -IMG_SINGLE_STEP_VALUE_ABS
    #test start
    plane_section_num = img_none_zero_segmentation_plane_section_num(img_unique_value_sorted, seg_num)
    img_seg_step_value = 2*img_single_step_value
    img_current_intensity_idx = 0
    img_current_intensity = img_unique_value_sorted[img_current_intensity_idx]
    img_current_threshold = img_current_intensity + img_seg_step_value
    img_current_threshold, img_current_intensity_idx = img_get_segment_threshold(img_unique_value_sorted, unique_value_index_dict, img_current_intensity_idx, img_current_threshold, img_seg_step_value, img_single_step_value)
    test_num = np.where(img_unique_value_sorted >= img_intensity_precision_proc(img_current_threshold), 1, 0)
    img_current_threshold, img_current_intensity_idx = img_get_segment_threshold(img_unique_value_sorted, unique_value_index_dict, img_current_intensity_idx, img_current_threshold, img_seg_step_value, img_single_step_value)
    img_current_threshold, img_current_intensity_idx = img_get_segment_threshold(img_unique_value_sorted, unique_value_index_dict, img_current_intensity_idx, img_current_threshold, img_seg_step_value, img_single_step_value)
    #test end
    return img_unique_value_sorted


def primefactors(n):
    factor_list = []
    # even number divisible
    factor_exp = 0
    while n % 2 == 0:
        factor_exp += 1
        n = n / 2
    if factor_exp > 0:
        factor_list.append((2, factor_exp))
    # n became odd
    for i in range(3, int(mat.sqrt(n)) + 1, 2):
        factor_exp = 0
        while n % i == 0:
            factor_exp += 1
            n = n / i
        if factor_exp > 0:
            factor_list.append((i, factor_exp))
    if n > 1:
        factor_list.append((np.int_(n), 1))
    return factor_list


def img_segmentation_threshold_list(img, threshold_list):
    segmentation_img_plane_list = []
    threshold_num = threshold_list.size
    current_seg_residual = img
    i = 0
    while i < threshold_num:
        img_current_threshold = threshold_list[i]
        segmentation_img_plane, current_seg_residual, segmentation_img_plane_arg = img_segmentation_plane_extraction(current_seg_residual, img_current_threshold)
        segmentation_img_plane_list.append(segmentation_img_plane)
        i += 1
    return segmentation_img_plane_list


def img_segmentation_threshold_list_light(img, threshold_list):
    threshold_num = threshold_list.size
    segmentation_img_plane_list = [] #[0#] * threshold_num
    #current_seg_residual = img
    s_time_seg = tm.time()
    current_seg_residual = np.round(img, IMAGE_ROUND_PRECISION)
    threshold_list = np.round(np.array(threshold_list), IMAGE_ROUND_PRECISION)
    e_time_seg = tm.time()
    #print("round:", e_time_seg - s_time_seg, "s")
    i = 0
    while i < threshold_num:
        #img_current_threshold
        precision_proc_threshold = threshold_list[i]

        #precision_proc_img = np.round(current_seg_residual, IMAGE_ROUND_PRECISION)

        #precision_proc_threshold = np.round(img_current_threshold, IMAGE_ROUND_PRECISION)

        segmentation_img_plane = np.where(current_seg_residual >= precision_proc_threshold, 1, 0)
        current_seg_residual = np.where(current_seg_residual < precision_proc_threshold, current_seg_residual, -1)

        #segmentation_img_plane_arg = np.argwhere(precision_proc_img >= precision_proc_threshold)
        #s_time_seg = tm.time()
        #current_seg_residual = current_seg_residual - current_seg_residual * segmentation_img_plane * 100
        #e_time_seg = tm.time()
        #segmentation_img_plane, current_seg_residual, segmentation_img_plane_arg = img_segmentation_plane_extraction(current_seg_residual, img_current_threshold)

        segmentation_img_plane_list.append(segmentation_img_plane)

        i += 1

        #print("one cycle:", e_time_seg-s_time_seg, "s")
    return segmentation_img_plane_list


def img_unique_value_count(img_int, unique_value):
    img_int_binary = np.where(img_int == unique_value, 1, 0)
    unique_value_count = np.sum(img_int_binary)
    return unique_value_count


def img_whole_range_unique_value_count(img, is_normalize=False):
    img_int = np.int_(np.around(img * INT_GRAY_LEVEL_BAR))
    img_intensity_max = np.max(img_int)
    img_intensity_min = np.min(img_int)
    img_whole_range_unique_value_count_list = []
    img_whole_range_unique_value_scale = range(img_intensity_min, img_intensity_max+1)
    for i in img_whole_range_unique_value_scale:
        if i >= 0:
            unique_value_count = img_unique_value_count(img_int, i)
            img_whole_range_unique_value_count_list.append(unique_value_count)
    img_whole_range_unique_value_count_list = np.array(img_whole_range_unique_value_count_list)
    if is_normalize:
        img_whole_range_unique_value_count_list = img_whole_range_unique_value_count_list/np.sum(img_whole_range_unique_value_count_list)
    if img_whole_range_unique_value_scale[0] < 0:
        value_len = len(img_whole_range_unique_value_scale)
        img_whole_range_unique_value_scale = img_whole_range_unique_value_scale[1:value_len]
    return np.int_(np.array(list(img_whole_range_unique_value_scale))), img_whole_range_unique_value_count_list#/np.max(img_whole_range_unique_value_count_list)


def img_intensity_zero_filter(img, intensity_to_neutralize, filter_intensity=0):
    img_filtered = np.where(img_intensity_precision_proc(img) != img_intensity_precision_proc(intensity_to_neutralize), img, filter_intensity)
    return img_filtered


def kapur_threshold(image):
    """ Runs the Kapur's threshold algorithm.

    Reference:
    Kapur, J. N., P. K. Sahoo, and A. K. C.Wong. ‘‘A New Method for Gray-Level
    Picture Thresholding Using the Entropy of the Histogram,’’ Computer Vision,
    Graphics, and Image Processing 29, no. 3 (1985): 273–285.

    @param image: The input image
    @type image: ndarray

    @return: The estimated threshold
    @rtype: int
    """
    hist, _ = np.histogram(image, bins=range(256), density=True)
    c_hist = hist.cumsum()
    c_hist_i = 1.0 - c_hist

    # To avoid invalid operations regarding 0 and negative values.
    c_hist[c_hist <= 0] = 1
    c_hist_i[c_hist_i <= 0] = 1

    c_entropy = (hist * np.log(hist + (hist <= 0))).cumsum()
    b_entropy = -c_entropy / c_hist + np.log(c_hist)

    c_entropy_i = c_entropy[-1] - c_entropy
    f_entropy = -c_entropy_i / c_hist_i + np.log(c_hist_i)

    return np.argmax(b_entropy + f_entropy)


def otsu_threshold(image=None, hist=None):
    """ Runs the Otsu threshold algorithm.

    Reference:
    Otsu, Nobuyuki. "A threshold selection method from gray-level
    histograms." IEEE transactions on systems, man, and cybernetics
    9.1 (1979): 62-66.

    @param image: The input image
    @type image: ndarray
    @param hist: The input image histogram
    @type hist: ndarray

    @return: The Otsu threshold
    @rtype int
    """
    if image is None and hist is None:
        raise ValueError('You must pass as a parameter either'
                         'the input image or its histogram')

    # Calculating histogram
    if not hist:
        hist = np.float_(np.histogram(image, bins=range(256))[0])

    cdf_backg = np.cumsum(np.arange(len(hist)) * hist)
    w_backg = np.cumsum(hist)  # The number of background pixels
    w_backg[w_backg == 0] = 1  # To avoid divisions by zero
    m_backg = cdf_backg / w_backg  # The means

    cdf_foreg = cdf_backg[-1] - cdf_backg
    w_foreg = w_backg[-1] - w_backg  # The number of foreground pixels
    w_foreg[w_foreg == 0] = 1  # To avoid divisions by zero
    m_foreg = cdf_foreg / w_foreg  # The means

    var_between_classes = w_backg * w_foreg * (m_backg - m_foreg) ** 2

    return np.argmax(var_between_classes)


def _get_variance(hist, c_hist, cdf, thresholds):
    """Get the total entropy of regions for a given set of thresholds"""

    variance = 0

    for i in range(len(thresholds) - 1):
        # Thresholds
        t1 = thresholds[i] + 1
        t2 = thresholds[i + 1]

        # Cumulative histogram
        weight = c_hist[t2] - c_hist[t1 - 1]

        # Region CDF
        r_cdf = cdf[t2] - cdf[t1 - 1]

        # Region mean
        r_mean = r_cdf / weight if weight != 0 else 0

        variance += weight * r_mean ** 2

    return variance


def _get_thresholds(hist, c_hist, cdf, nthrs):
    """Get the thresholds that maximize the variance between regions

    @param hist: The normalized histogram of the image
    @type hist: ndarray
    @param c_hist: The normalized histogram of the image
    @type c_hist: ndarray
    @param cdf: The cummulative distribution function of the histogram
    @type cdf: ndarray
    @param nthrs: The number of thresholds
    @type nthrs: int
    """
    # Thresholds combinations
    thr_combinations = combinations(range(255), nthrs)

    max_var = 0
    opt_thresholds = None

    # Extending histograms for convenience
    c_hist = np.append(c_hist, [0])
    cdf = np.append(cdf, [0])

    for thresholds in thr_combinations:
        # Extending thresholds for convenience
        e_thresholds = [-1]
        e_thresholds.extend(thresholds)
        e_thresholds.extend([len(hist) - 1])

        # Computing variance for the current combination of thresholds
        regions_var = _get_variance(hist, c_hist, cdf, e_thresholds)

        if regions_var > max_var:
            max_var = regions_var
            opt_thresholds = thresholds

    return np.array(opt_thresholds)


def otsu_multithreshold(image=None, hist=None, nthrs=2):
    """ Runs the Otsu's multi-threshold algorithm.

    Reference:
    Otsu, Nobuyuki. "A threshold selection method from gray-level
    histograms." IEEE transactions on systems, man, and cybernetics
    9.1 (1979): 62-66.

    Liao, Ping-Sung, Tse-Sheng Chen, and Pau-Choo Chung. "A fast algorithm
    for multilevel thresholding." J. Inf. Sci. Eng. 17.5 (2001): 713-727.

    @param image: The input image
    @type image: ndarray
    @param hist: The input image histogram
    @type hist: ndarray
    @param nthrs: The number of thresholds
    @type nthrs: int

    @return: The estimated thresholds
    @rtype: int
    """
    # Histogran
    if image is None and hist is None:
        raise ValueError('You must pass as a parameter either'
                         'the input image or its histogram')

    # Calculating histogram
    if not hist:
        hist = np.float_(np.histogram(image, bins=range(256))[0])

    # Cumulative histograms
    c_hist = np.cumsum(hist)
    cdf = np.cumsum(np.arange(len(hist)) * hist)

    return _get_thresholds(hist, c_hist, cdf, nthrs)


def _get_regions_entropy(hist, c_hist, thresholds):
    """Get the total entropy of regions for a given set of thresholds"""

    total_entropy = 0
    for i in range(len(thresholds) - 1):
        # Thresholds
        t1 = thresholds[i] + 1
        t2 = thresholds[i + 1]

        # print(thresholds, t1, t2)

        # Cumulative histogram
        hc_val = c_hist[t2] - c_hist[t1 - 1]

        # Normalized histogram
        h_val = hist[t1:t2 + 1] / hc_val if hc_val > 0 else 1

        # entropy
        entropy = -(h_val * np.log(h_val + (h_val <= 0))).sum()

        # Updating total entropy
        total_entropy += entropy

    return total_entropy


def _get_thresholds_kapur(hist, c_hist, nthrs):
    """Get the thresholds that maximize the entropy of the regions

    @param hist: The normalized histogram of the image
    @type hist: ndarray
    @param c_hist: The cummuative normalized histogram of the image
    @type c_hist: ndarray
    @param nthrs: The number of thresholds
    @type nthrs: int
    """
    # Thresholds combinations
    thr_combinations = combinations(range(255), nthrs)

    max_entropy = 0
    opt_thresholds = None

    # Extending histograms for convenience
    # hist = np.append([0], hist)
    c_hist = np.append(c_hist, [0])

    for thresholds in thr_combinations:
        # Extending thresholds for convenience
        e_thresholds = [-1]
        e_thresholds.extend(thresholds)
        e_thresholds.extend([len(hist) - 1])

        # Computing regions entropy for the current combination of thresholds
        regions_entropy = _get_regions_entropy(hist, c_hist, e_thresholds)

        if regions_entropy > max_entropy:
            max_entropy = regions_entropy
            opt_thresholds = thresholds

    return np.array(opt_thresholds)


def kapur_multithreshold(image, nthrs):
    """ Runs the Kapur's multi-threshold algorithm.

    Reference:
    Kapur, J. N., P. K. Sahoo, and A. K. C.Wong. ‘‘A New Method for Gray-Level
    Picture Thresholding Using the Entropy of the Histogram,’’ Computer Vision,
    Graphics, and Image Processing 29, no. 3 (1985): 273–285.

    @param image: The input image
    @type image: ndarray
    @param nthrs: The number of thresholds
    @type nthrs: int

    @return: The estimated threshold
    @rtype: int
    """
    # Histogran
    hist, _ = np.histogram(image, bins=range(256), density=True)

    # Cumulative histogram
    c_hist = hist.cumsum()

    return _get_thresholds_kapur(hist, c_hist, nthrs)


def img_sim_measure(img_original, img_reconstructed, max_p_ssim=MAX_P_SSIM, max_p_psnr=MAX_P_PSNR, max_p_rmse=MAX_P_RMSE):
    fsim = img_qm.fsim(img_original, img_reconstructed)
    ssim = img_qm.ssim(img_original, img_reconstructed, max_p=65)
    psnr = img_qm.psnr(img_original, img_reconstructed, max_p=5)
    rmse = img_qm.rmse(img_original, img_reconstructed, max_p=1)
    return fsim, ssim, psnr, rmse


def img_triple_measure_max_idx(measure_0, measure_1, measure_2):
    measure_list = np.array([measure_0, measure_1, measure_2])
    measure_max_idx = np.argmax(measure_list)
    return measure_max_idx, measure_list


def img_triple_target_measure_max_measure_diff(target_measure_idx, measure_list):
    measure_max_idx = np.argmax(measure_list)
    measure_min_idx = np.argmin(measure_list)
    target_measure = measure_list[target_measure_idx]
    measure_diff_from_max = target_measure - measure_list[measure_max_idx]
    measure_diff_from_min = target_measure - measure_list[measure_min_idx]
    return measure_diff_from_max, measure_diff_from_min


def img_triple_target_measure_diff_scenario(measure_diff_from_max, measure_diff_from_min):
    max_diff_sign = np.sign(measure_diff_from_max)
    min_diff_sign = np.sign(measure_diff_from_min)
    mixed_sign = max_diff_sign * min_diff_sign
    diff_scenario = DIFF_SCENARIO_MIDDLE
    if mixed_sign >= 0:
        if max_diff_sign >= 0:
            diff_scenario = DIFF_SCENARIO_MAX
        else:
            diff_scenario = DIFF_SCENARIO_MINI
    return diff_scenario


SEGMENTATION_OPTIMIZATION_METHOD_OTSU = "otsu"
SEGMENTATION_OPTIMIZATION_METHOD_KAPUR = "kapur"
SEGMENTATION_OPTIMIZATION_FUN_DICT = {SEGMENTATION_OPTIMIZATION_METHOD_OTSU:otsu_multithreshold, SEGMENTATION_OPTIMIZATION_METHOD_KAPUR:kapur_multithreshold}


def img_integer_transfer(img):
    res_img = img*INT_GRAY_LEVEL_BAR
    return res_img


def img_segmentation_optimization_threshold_list(img, seg_num, opt_target, is_img_integer=False):
    if not is_img_integer:
        img = img_integer_transfer(img)
    threshold_num = seg_num - 1
    opt_target_fun = SEGMENTATION_OPTIMIZATION_FUN_DICT[opt_target]
    threshold_list = opt_target_fun(img, nthrs=threshold_num) + 1
    threshold_list = np.sort(threshold_list)
    threshold_list = threshold_list[::-1]
    threshold_list = np.append(threshold_list, 0)
    return threshold_list


def img_integer_segmentation_equal_range_thresholds(img, seg_num, is_segment_mod=False):
    img_max = np.max(img)
    img_min = np.min(img)
    img_mini_idx_ref = np.int_(img_min*INT_GRAY_LEVEL_BAR)
    img_single_step_value = img_intensity_precision_proc(-IMG_SINGLE_STEP_VALUE_ABS)
    plane_section_num, intensity_plane_num = img_plane_whole_range_section_num(img, seg_num)
    intensity_plane_num = np.int_(intensity_plane_num)
    img_hist = np.histogram(img, bins=intensity_plane_num)[0]

    img_seg_step = (plane_section_num - 1) * img_single_step_value
    threshold_list = []
    threshold_list_with_zero_section = []
    i = 0
    threshold_num = seg_num # - 1
    threshold = img_max
    threshold_idx = 1
    while (i < threshold_num or is_segment_mod) and threshold_idx > 0:
        #threshold = threshold + img_seg_step
        threshold = img_intensity_precision_proc(threshold + img_seg_step)
        # test
        threshold_int = threshold * INT_GRAY_LEVEL_BAR
        # test
        threshold_list_with_zero_section.append(threshold)
        threshold_idx = np.int_(threshold * INT_GRAY_LEVEL_BAR) - img_mini_idx_ref
        if np.sum(img_hist[threshold_idx:threshold_idx+plane_section_num+1]) > 0:
            threshold_list.append(threshold)
            i += 1
        threshold += img_single_step_value
    effective_threshold_last = 0
    if i == seg_num and not is_segment_mod:
        threshold_last_idx = seg_num - 1
        effective_threshold_last = threshold_list[threshold_last_idx]
        threshold_list[threshold_last_idx] = 0
        threshold_list_with_zero_section[threshold_last_idx] = 0
    elif threshold_list[i-1] > 0:
        effective_threshold_last = threshold_list[len(threshold_list)-1] + img_seg_step + img_single_step_value
        threshold_list.append(0)
        threshold_list_with_zero_section.append(0)
    threshold_list = np.array(threshold_list)
    threshold_list_with_zero_section = np.array(threshold_list_with_zero_section)
    return threshold_list, np.int_(np.around(threshold_list*INT_GRAY_LEVEL_BAR)), threshold_list_with_zero_section, np.int_(threshold_list_with_zero_section*INT_GRAY_LEVEL_BAR), plane_section_num, effective_threshold_last


def img_integer_segmentation_equal_range_thresholds_light(img, seg_num, is_segment_mod=False):
    img_max = np.amax(img)
    img_min = np.min(img)
    img_mini_idx_ref = np.int_(img_min*INT_GRAY_LEVEL_BAR)
    img_single_step_value = img_intensity_precision_proc(-IMG_SINGLE_STEP_VALUE_ABS)
    plane_section_num, intensity_plane_num = img_plane_whole_range_section_num(img, seg_num)
    intensity_plane_num = np.int_(intensity_plane_num)
    #img_hist = np.histogram(img, bins=intensity_plane_num)[0]
    img_seg_step = (plane_section_num - 1) * img_single_step_value
    threshold_list = []
    threshold_list_with_zero_section = []
    i = 0
    threshold_num = seg_num # - 1
    threshold = img_max
    threshold_idx = 1

    while (i < threshold_num or is_segment_mod) and threshold_idx > 0:
        #threshold = threshold + img_seg_step
        threshold = img_intensity_precision_proc(threshold + img_seg_step)
        # test
        threshold_int = threshold * INT_GRAY_LEVEL_BAR
        # test
        threshold_list_with_zero_section.append(threshold)
        threshold_idx = np.int_(threshold * INT_GRAY_LEVEL_BAR) - img_mini_idx_ref
        #if np.sum(img_hist[threshold_idx:threshold_idx+plane_section_num+1]) > 0:
        threshold_list.append(threshold)
        i += 1
        threshold += img_single_step_value
    effective_threshold_last = 0
    if i == seg_num and not is_segment_mod:
        threshold_last_idx = seg_num - 1
        effective_threshold_last = threshold_list[threshold_last_idx]
        threshold_list[threshold_last_idx] = img_min #0
        threshold_list_with_zero_section[threshold_last_idx] = img_min #0
    elif threshold_list[i-1] > img_min:
        effective_threshold_last = threshold_list[len(threshold_list)-1] + img_seg_step + img_single_step_value
        threshold_list.append(img_min) #0)
        threshold_list_with_zero_section.append(img_min) #0)
    elif threshold_list[i-1] < 0:
        threshold_list[i-1] = img_min
    threshold_list = np.array(threshold_list)
    threshold_list_with_zero_section = np.array(threshold_list_with_zero_section)
    return threshold_list, np.int_(np.around(threshold_list*INT_GRAY_LEVEL_BAR)), threshold_list_with_zero_section, np.int_(threshold_list_with_zero_section*INT_GRAY_LEVEL_BAR), plane_section_num, effective_threshold_last


def img_threshold_array_div(img_threshold_list, effective_threshold_last):
    list_last_idx = img_threshold_list.size - 1
    img_threshold_list[list_last_idx] = img_intensity_precision_proc(effective_threshold_last)
    img_threshold_list_div = abs_one_dim_array_div(img_threshold_list, False)
    return img_threshold_list_div


def img_gray_integer_segmentation_equal_range_reconstruction(img_gray, seg_num, is_segment_mod=False):
   multi_threshold_integer_float,  multi_threshold_integer_int, threshold_list_with_zero, threshold_list_with_zero_int, plane_section_num, effective_threshold_last = img_integer_segmentation_equal_range_thresholds_light(img_gray, seg_num, is_segment_mod)
   segmentation_img_plane_list = img_segmentation_threshold_list_light(img_gray, multi_threshold_integer_int / 255)
   img_reconstructed = img_segmentation_reconstruction_threshold(segmentation_img_plane_list, multi_threshold_integer_int / 255)
   return img_reconstructed

def img_rgb_integer_segmentation_equal_range_thresholds(img_rgb, seg_num, is_segment_mod=False):
    img_red = img_rgb[:, :, IMG_MAP_RED_IDX]
    img_blue = img_rgb[:, :, IMG_MAP_BLUE_IDX]
    img_green = img_rgb[:, :, IMG_MAP_GREEN_IDX]

    multi_threshold_integer_float_red, multi_threshold_integer_int_red, threshold_list_with_zero_red, threshold_list_with_zero_int_red, plane_section_num_red, effective_threshold_last_red = img_integer_segmentation_equal_range_thresholds(img_red, seg_num, is_segment_mod=is_segment_mod)
    multi_threshold_integer_float_blue, multi_threshold_integer_int_blue, threshold_list_with_zero_blue, threshold_list_with_zero_int_blue, plane_section_num_blue, effective_threshold_last_blue = img_integer_segmentation_equal_range_thresholds(img_blue, seg_num, is_segment_mod=is_segment_mod)
    multi_threshold_integer_float_green, multi_threshold_integer_int_green, threshold_list_with_zero_green, threshold_list_with_zero_int_green, plane_section_num_green, effective_threshold_last_green = img_integer_segmentation_equal_range_thresholds(img_green, seg_num, is_segment_mod=is_segment_mod)

    segmentation_img_plane_list_red = img_segmentation_threshold_list(img_red, multi_threshold_integer_int_red/255)
    segmentation_img_plane_list_blue = img_segmentation_threshold_list(img_blue, multi_threshold_integer_int_blue/255)
    segmentation_img_plane_list_green = img_segmentation_threshold_list(img_green, multi_threshold_integer_int_green/255)

    #img_div_threshold_list_red = img_threshold_array_div(multi_threshold_integer_int_red/255, effective_threshold_last_red)
    #img_div_threshold_list_blue = img_threshold_array_div(multi_threshold_integer_int_blue/255, effective_threshold_last_blue)
    #img_div_threshold_list_green = img_threshold_array_div(multi_threshold_integer_int_green/255, effective_threshold_last_green)

    img_max_red = np.max(img_red)
    img_max_blue = np.max(img_blue)
    img_max_green = np.max(img_green)

    #img_red_reconstructed = img_segmentation_reconstruction(segmentation_img_plane_list_red, img_div_threshold_list_red, img_max_red)
    #img_blue_reconstructed = img_segmentation_reconstruction(segmentation_img_plane_list_blue, img_div_threshold_list_blue, img_max_blue)
    #img_green_reconstructed = img_segmentation_reconstruction(segmentation_img_plane_list_green, img_div_threshold_list_green, img_max_green)

    img_red_reconstructed = img_segmentation_reconstruction_threshold(segmentation_img_plane_list_red, multi_threshold_integer_int_red/255)
    img_blue_reconstructed = img_segmentation_reconstruction_threshold(segmentation_img_plane_list_blue, multi_threshold_integer_int_blue/255)
    img_green_reconstructed = img_segmentation_reconstruction_threshold(segmentation_img_plane_list_green, multi_threshold_integer_int_green/255)

    img_red_reconstructed = np.expand_dims(img_red_reconstructed, 2)
    img_blue_reconstructed = np.expand_dims(img_blue_reconstructed, 2)
    img_green_reconstructed = np.expand_dims(img_green_reconstructed, 2)
    #img_rgb_reconstructed = np.concatenate((img_red_reconstructed, img_green_reconstructed, img_blue_reconstructed), axis=2)
    img_rgb_reconstructed = np.concatenate((img_red_reconstructed, img_green_reconstructed, img_blue_reconstructed), axis=2)
    return img_rgb_reconstructed


def sparse_binary_array_compression(sparse_array, non_sparse_element, sparse_element):
    array_len = np.size(sparse_array)
    compressed_array = []
    scan_idx = 0
    start_element = sparse_array[scan_idx]
    if start_element == non_sparse_element:
        scan_idx = fill_compressed_array(sparse_array, non_sparse_element, scan_idx, array_len, compressed_array)
    else:
        scan_idx = omit_sparse_element(sparse_array, sparse_element, scan_idx, array_len)
        if scan_idx < array_len:
            scan_idx = fill_compressed_array(sparse_array, non_sparse_element, scan_idx, array_len, compressed_array)
    while scan_idx < array_len:
        scan_idx = omit_sparse_element(sparse_array, sparse_element, scan_idx, array_len)
        scan_idx = fill_compressed_array(sparse_array, non_sparse_element, scan_idx, array_len, compressed_array)
    return compressed_array


def extract_non_spars_element_seg(sparse_array, non_sparse_element, idx_start, array_len):
    i = idx_start
    left_bound = idx_start
    i += 1
    while i < array_len:
        if sparse_array[i] != non_sparse_element:
            break
        i += 1
    i -= 1
    right_bound = i
    element_seg = (left_bound, right_bound)
    return element_seg


def fill_compressed_array(sparse_array, non_sparse_element, idx_start, array_len, compressed_array):
    non_sparse_seg = extract_non_spars_element_seg(sparse_array, non_sparse_element, idx_start, array_len)
    if non_sparse_seg[NON_SPARSE_SEG_RIGHT_IDX] < array_len:
        compressed_array.append(non_sparse_seg)
    scan_idx = non_sparse_seg[NON_SPARSE_SEG_RIGHT_IDX] + 1
    return scan_idx


def omit_sparse_element(sparse_array, sparse_element, idx_start, array_len):
    i = idx_start + 1
    while i < array_len:
        if sparse_array[i] != sparse_element:
            break
        i += 1
    return i


def image_gray_rect_region_intensity_increment(img, rect_x_start, rect_y_start, rect_x_end, rect_y_end, increment_ratio, modify_intensity, is_img_copy=True, is_direct_value=False):
    img_mod = get_img_instance_copy_choice(img, is_img_copy)
    x_rnd_len = np.int_((rect_x_end - rect_x_start + 1) * increment_ratio)
    y_rnd_len = np.int_((rect_y_end - rect_y_start + 1) * increment_ratio)
    if increment_ratio < 1:
        rand_x_list = np.random.randint(rect_x_start, rect_x_end, x_rnd_len)
        rand_y_list = np.random.randint(rect_y_start, rect_y_end, y_rnd_len)
    else:
        rand_x_list = range(rect_x_start, rect_x_end)
        rand_y_list = range(rect_y_start, rect_y_end)

    for i in rand_x_list:
        for j in rand_y_list:
            if is_direct_value:
                img_mod[i, j] = modify_intensity
            else:
                img_mod[i, j] += modify_intensity
    return img_mod


def generate_inner_rect_gray_img(img_size, inner_rect_size, inner_rect_intensity, outer_rect_intensity):
    img_len = img_size[0]
    img_width = img_size[1]
    inner_rect_len = inner_rect_size[0]
    inner_rect_width = inner_rect_size[1]
    inner_rect_len_offset = np.int_((img_len - inner_rect_len)/2)
    inner_rect_width_offset = np.int_((img_width - inner_rect_width)/2)
    inner_rect_x_start = inner_rect_len_offset
    inner_rect_y_start = inner_rect_width_offset
    inner_rect_x_end_len = inner_rect_x_start + inner_rect_len
    inner_rect_y_end_len = inner_rect_y_start + inner_rect_width
    img_gray = np.zeros(img_size)
    img_gray[:, :] = outer_rect_intensity
    img_gray[inner_rect_x_start:inner_rect_x_end_len, inner_rect_y_start:inner_rect_y_end_len] = inner_rect_intensity
    return img_gray


def img_gray_rect_part_copy(img_source, img_target, source_rect, target_rect, is_img_copy=True):
    img_mod = get_img_instance_copy_choice(img_target, is_img_copy)
    source_x_start = source_rect[IMG_GRAY_SOURCE_TARGET_RECT_X_TL_IDX]
    source_x_end = source_rect[IMG_GRAY_SOURCE_TARGET_RECT_X_BR_IDX] + 1
    source_y_start = source_rect[IMG_GRAY_SOURCE_TARGET_RECT_Y_TL_IDX]
    source_y_end = source_rect[IMG_GRAY_SOURCE_TARGET_RECT_Y_BR_IDX] + 1
    target_x_start = target_rect[IMG_GRAY_SOURCE_TARGET_RECT_X_TL_IDX]
    target_x_end = target_rect[IMG_GRAY_SOURCE_TARGET_RECT_X_BR_IDX] + 1
    target_y_start = target_rect[IMG_GRAY_SOURCE_TARGET_RECT_Y_TL_IDX]
    target_y_end = target_rect[IMG_GRAY_SOURCE_TARGET_RECT_Y_BR_IDX] + 1
    img_mod[target_x_start:target_x_end, target_y_start:target_y_end] = img_source[source_x_start:source_x_end, source_y_start:source_y_end]
    return img_mod


def display_img_gray_hist_statistics(img_gray):
    img_whole_range_unique_value_scale, img_whole_range_unique_value_count_list = img_whole_range_unique_value_count(img_gray)
    plt.bar(img_whole_range_unique_value_scale, img_whole_range_unique_value_count_list)
    plt.show()


def display_img_gray_hist_statistics_comp(img_original, img_reconstructed, is_normalize=False):
    img_whole_range_unique_value_scale_org, img_whole_range_unique_value_count_list_org = img_whole_range_unique_value_count(img_original, is_normalize)
    img_whole_range_unique_value_scale_rec, img_whole_range_unique_value_count_list_rec = img_whole_range_unique_value_count(img_reconstructed, is_normalize)
    plt.subplot(1, 2, 1)
    plt.bar(img_whole_range_unique_value_scale_org, img_whole_range_unique_value_count_list_org)
    plt.subplot(1, 2, 2)
    plt.bar(img_whole_range_unique_value_scale_rec, img_whole_range_unique_value_count_list_rec)
    plt.show()
    return img_whole_range_unique_value_scale_org, img_whole_range_unique_value_count_list_org, img_whole_range_unique_value_scale_rec, img_whole_range_unique_value_count_list_rec


def convert_gray_dicom_to_normal_img(dicom_file_path):
    ds = pydicom.dcmread(dicom_file_path)
    img = ds.pixel_array / DICOM_NORM_FACTOR
    img = one_dim_min_max_normalization(img)
    return img, ds


def create_point_cloud_plane(plane_dim, plane_idx, plane_dict):
    plane = np.zeros(plane_dim)
    plane_key = str(plane_idx)
    plane_dict[plane_key] = plane
    return plane_dict


def add_edge_point_delta_metric(edge_point_delta_metric, positive_metric, negative_metric):
    if edge_point_delta_metric >= 0:
        add_res = positive_metric + edge_point_delta_metric
        add_res = (add_res, negative_metric)
    else:
        add_res = negative_metric + edge_point_delta_metric
        add_res = (positive_metric, add_res)
    return add_res


def get_edge_point_direction(left_delta, right_delta):
    left_delta_sign = np.sign(left_delta)
    right_delta_sign = np.sign(right_delta)
    delta_sign_mul = np.abs(left_delta_sign * right_delta_sign)
    if delta_sign_mul == 0:
        edge_point_direction = left_delta_sign + right_delta_sign
        contain_zero = True
    else:
        edge_point_direction = left_delta_sign
        contain_zero = False
    return edge_point_direction, contain_zero


def extract_image_segment_monocular_single_3d_structure(image_seg_delta, image_seg_delta_len, scan_start, edge_delta_metric):
    edge_point_delta_metric = edge_delta_metric
    #edge_point_direction = np.sign(image_seg_delta[scan_start-1])
    if scan_start < image_seg_delta_len:
        edge_point_direction, contain_zero = get_edge_point_direction(image_seg_delta[scan_start-1], image_seg_delta[scan_start])
    else:
        edge_point_direction = np.sign(image_seg_delta[scan_start-1])
    edge_delta_direction = np.sign(edge_delta_metric)
    structure_metric = edge_point_delta_metric
    delta_count = 1
    scan_continue = True
    scan_idx = scan_start
    while scan_continue and scan_idx < image_seg_delta_len:
        edge_point_delta_metric = image_seg_delta[scan_idx] - image_seg_delta[scan_idx - 1]
        edge_point_direction_metric, contain_zero = get_edge_point_direction(image_seg_delta[scan_idx-1], image_seg_delta[scan_idx])  #np.sign(image_seg_delta[scan_idx])
        edge_delta_direction_metric = np.sign(edge_point_delta_metric)
        if edge_point_direction == edge_point_direction_metric and edge_point_direction_metric != 0 and (edge_delta_direction_metric != 0 or contain_zero):  # ((edge_delta_direction == edge_delta_direction_metric and edge_delta_direction_metric != 0) or contain_zero):
            structure_metric += edge_point_delta_metric
            delta_count += 1
        else:
            #edge_point_direction = np.sign(image_seg_delta[scan_start-2])
            scan_continue = False
        scan_idx += 1
    #if delta_count == 1:
        #structure_metric = 0
    point_count = delta_count + 1
    structure_shape = np.sign(structure_metric) * -1 # edge_point_direction * -1  # 1 BULGE, 0 FLAT, -1 CONCAVE
    if structure_shape == MONOCULAR_3D_SHAPE_CONCAVE_DIALECT:
        structure_shape = MONOCULAR_3D_SHAPE_CONCAVE
    structure_metric = np.abs(structure_metric)
    return structure_metric, structure_shape, point_count


def image_segment_monocular_3d_structure_scan(image_seg_delta):
    structure_metric_list = []
    structure_shape_list = []
    point_count_list = []
    image_seg_delta_len = image_seg_delta.size
    scan_idx = 1
    while scan_idx < image_seg_delta_len:
        delta_right = image_seg_delta[scan_idx]
        delta_left = image_seg_delta[scan_idx-1]
        delta_metric_conserve = np.abs(np.sign(delta_left * delta_right))
        if delta_metric_conserve != 0:
            edge_delta_metric = delta_right - delta_left
        else:
            edge_delta_metric = 0
        delta_left_sign = np.sign(delta_left)
        delta_right_sign = np.sign(delta_right)
        delta_sign_diff_s = delta_left_sign - delta_right_sign
        delta_sign_diff_m = delta_left_sign * delta_right_sign
        scan_idx += 1
        if edge_delta_metric == 0 or (delta_sign_diff_s != 0 and delta_sign_diff_m != 0):
            structure_metric_list.append(0)
            structure_shape_list.append(MONOCULAR_3D_SHAPE_FLAT)
            point_count_list.append(MONOCULAR_3D_SCAN_STATIC_POINT_COUNT)
            #point_count = 2
        else:
            structure_metric, structure_shape, point_count = extract_image_segment_monocular_single_3d_structure(image_seg_delta, image_seg_delta_len, scan_idx, edge_delta_metric)
            structure_metric_list.append(structure_metric)
            structure_shape_list.append(structure_shape)
            point_count_list.append(point_count)
            scan_idx = scan_idx + point_count - 2
    return np.array(structure_metric_list), np.array(structure_shape_list), np.array(point_count_list)


def extract_image_segment_monocular_single_3d_structure_curve_linear_diff_test(image_seg_delta, image_seg_delta_len, scan_start, edge_delta_metric):
    scan_idx = scan_start
    edge_point_delta_metric_left = edge_delta_metric
    edge_point_direction_left_base = np.sign(edge_point_delta_metric_left)
    edge_point_direction_left = edge_point_direction_left_base

    if scan_idx < image_seg_delta_len:
        delta_grad = image_seg_delta[scan_idx] - edge_point_delta_metric_left
        grad_direction_left = np.sign(delta_grad)
    baseline_delta = edge_point_delta_metric_left
    grad_sum = edge_point_delta_metric_left

    structure_metric_curve = np.abs(edge_point_delta_metric_left)
    structure_metric_curve_total = np.abs(edge_point_delta_metric_left)
    delta_count = 1
    scan_continue = True
    while scan_continue and scan_idx < image_seg_delta_len:
        edge_point_delta_metric_right = image_seg_delta[scan_idx]
        edge_point_direction_right = np.sign(edge_point_delta_metric_right)
        grad_sum += edge_point_delta_metric_right

        if edge_point_direction_left_base == edge_point_direction_right or (edge_point_direction_right == 0 and edge_point_direction_left != 0 and delta_count == 1):
            structure_metric_curve += np.abs(edge_point_delta_metric_right) * delta_grad  # grad_direction_left
            structure_metric_curve_total += np.abs(edge_point_delta_metric_right)
            delta_count += 1
            if edge_point_direction_right == 0:
                scan_continue = False
                if delta_count == 2:
                    structure_metric_curve *= -1  # grad_direction_left
        else:
            scan_continue = False
        edge_point_direction_left = edge_point_direction_right
        scan_idx += 1
        if scan_idx < image_seg_delta_len and scan_continue:
            delta_grad = image_seg_delta[scan_idx] - edge_point_delta_metric_right
            grad_direction_right = np.sign(delta_grad)
            if grad_direction_left != grad_direction_right and image_seg_delta[scan_idx] != 0:
                baseline_delta = grad_sum/delta_count
                grad_direction_left = grad_direction_right
        #edge_point_delta_metric_left = edge_point_delta_metric_right
    point_count = delta_count + 1
    if delta_count > 1:
        structure_metric_base = baseline_delta * delta_count
    else:
        structure_metric_base = 0
        structure_metric_curve = 0
    structure_metric = structure_metric_curve #- structure_metric_base
    structure_shape = np.sign(structure_metric)  # 1 BULGE, 0 FLAT, -1 CONCAVE
    structure_metric = np.abs(structure_metric)
    structure_metric = structure_metric #/structure_metric_curve_total#/(structure_metric + np.abs(baseline_delta))
    return structure_metric, structure_shape, point_count


def extract_image_segment_monocular_single_3d_structure_curve_linear_diff(image_seg_delta, image_seg_delta_len, scan_start, edge_delta_metric):
    scan_idx = scan_start
    edge_point_delta_metric_left = edge_delta_metric
    edge_point_direction_left_base = np.sign(edge_point_delta_metric_left)
    edge_point_direction_left = edge_point_direction_left_base

    if scan_idx < image_seg_delta_len:
        delta_grad = image_seg_delta[scan_idx] - edge_point_delta_metric_left
        grad_direction_left = np.sign(delta_grad)
    baseline_delta = edge_point_delta_metric_left
    grad_sum = edge_point_delta_metric_left

    structure_metric_curve = np.abs(edge_point_delta_metric_left)
    structure_metric_curve_total = np.abs(edge_point_delta_metric_left)
    delta_count = 1
    scan_continue = True
    while scan_continue and scan_idx < image_seg_delta_len:
        edge_point_delta_metric_right = image_seg_delta[scan_idx]
        edge_point_direction_right = np.sign(edge_point_delta_metric_right)
        grad_sum += edge_point_delta_metric_right

        if edge_point_direction_left_base == edge_point_direction_right or (edge_point_direction_right == 0 and edge_point_direction_left != 0 and delta_count == 1):
            structure_metric_curve += np.abs(edge_point_delta_metric_right) * delta_grad  # grad_direction_left
            structure_metric_curve_total += np.abs(edge_point_delta_metric_right)
            delta_count += 1
            if edge_point_direction_right == 0:
                scan_continue = False
                if delta_count == 2:
                    structure_metric_curve *= -1  # grad_direction_left
        else:
            scan_continue = False
        edge_point_direction_left = edge_point_direction_right
        scan_idx += 1
        if scan_idx < image_seg_delta_len and scan_continue:
            delta_grad = image_seg_delta[scan_idx] - edge_point_delta_metric_right
            grad_direction_right = np.sign(delta_grad)
            if grad_direction_left != grad_direction_right and image_seg_delta[scan_idx] != 0:
                baseline_delta = grad_sum/delta_count
                grad_direction_left = grad_direction_right
        #edge_point_delta_metric_left = edge_point_delta_metric_right
    point_count = delta_count + 1
    if delta_count > 1:
        structure_metric_base = baseline_delta * delta_count
    else:
        structure_metric_base = 0
        structure_metric_curve = 0
    structure_metric = structure_metric_curve #- structure_metric_base
    structure_shape = np.sign(structure_metric)  # 1 BULGE, 0 FLAT, -1 CONCAVE
    structure_metric = np.abs(structure_metric)
    structure_metric = structure_metric #/structure_metric_curve_total#/(structure_metric + np.abs(baseline_delta))
    return structure_metric, structure_shape, point_count


def image_segment_monocular_3d_structure_scan_curve_linear_diff(image_seg_delta):
    structure_metric_list = []
    structure_shape_list = []
    point_count_list = []
    image_seg_delta_len = image_seg_delta.size
    scan_idx = 0
    while scan_idx < image_seg_delta_len:
        edge_delta_metric = image_seg_delta[scan_idx]
        scan_idx += 1
        if edge_delta_metric == 0:
            structure_metric_list.append(0)
            structure_shape_list.append(MONOCULAR_3D_SHAPE_FLAT)
            point_count_list.append(MONOCULAR_3D_SCAN_STATIC_POINT_COUNT)
        else:
            structure_metric, structure_shape, point_count = extract_image_segment_monocular_single_3d_structure_curve_linear_diff(image_seg_delta, image_seg_delta_len, scan_idx, edge_delta_metric)
            structure_metric_list.append(structure_metric)
            structure_shape_list.append(structure_shape)
            point_count_list.append(point_count)
            scan_idx = scan_idx + point_count - 2
    return np.array(structure_metric_list), np.array(structure_shape_list), np.array(point_count_list)


def monocular_3d_code_word_extraction(image_seg_delta, image_seg_delta_len, start_idx):
    start_element_val = image_seg_delta[start_idx]
    start_code = np.sign(start_element_val)
    element_len = 1
    i = start_idx + 1
    next_element_code = MONOCULAR_3D_CODE_SINGLE_ELEMENT_SE
    element_val_delta_list = [start_element_val]
    while i < image_seg_delta_len:
        code_element_val = image_seg_delta[i]
        code_element = np.sign(code_element_val)
        if code_element == start_code:
            element_val_delta_list.append(code_element_val)
            element_len += 1
            i += 1
        else:
            next_element_code = code_element
            break
    element_val_delta_list = np.array(element_val_delta_list)
    monocular_3d_code_word_delta_diff_direction = MONOCULAR_3D_CODE_WORD_DELTA_DIFF_DIRECTION_F
    code_word_res = (start_code, element_len, element_val_delta_list, monocular_3d_code_word_delta_diff_direction)
    return code_word_res, i, next_element_code


def monocular_3d_code_ocular_is_scan_continue(code_element_left, code_element_right, next_element_code, ocular_code_element_base):
    if code_element_right == MONOCULAR_3D_CODE_SINGLE_ELEMENT_F:
        scan_continue = next_element_code == ocular_code_element_base
    else:
        scan_continue = next_element_code == MONOCULAR_3D_CODE_SINGLE_ELEMENT_F
    return scan_continue


def monocular_3d_code_ocular_seg_proc(image_seg_delta, image_seg_delta_len, code_word_res, start_idx, next_element_code):
    scan_idx = start_idx
    code_element_left = code_word_res[MONOCULAR_3D_CODE_WORD_TUPLE_CODE_ELEMENT_IDX]
    if code_element_left != MONOCULAR_3D_CODE_SINGLE_ELEMENT_F:
        code_delta_segment = code_word_res[MONOCULAR_3D_CODE_WORD_TUPLE_ELEMENT_VAL_DELTA_LIST_IDX]
        ocular_code_element_base = code_element_left
    else:
        ocular_code_element_base = next_element_code
        code_delta_segment = np.zeros(0)
    code_word_unresolved_param = None
    while scan_idx < image_seg_delta_len:
        code_word_res, scan_idx, next_element_code = monocular_3d_code_word_extraction(image_seg_delta, image_seg_delta_len, scan_idx)
        code_element_right = code_word_res[MONOCULAR_3D_CODE_WORD_TUPLE_CODE_ELEMENT_IDX]
        scan_continue = monocular_3d_code_ocular_is_scan_continue(code_element_left, code_element_right, next_element_code, ocular_code_element_base)
        code_delta_segment_right = code_word_res[MONOCULAR_3D_CODE_WORD_TUPLE_ELEMENT_VAL_DELTA_LIST_IDX]
        if scan_continue:
            code_delta_segment = np.concatenate((code_delta_segment, code_delta_segment_right))
            code_element_left = code_element_right
        else:
            if code_element_right == MONOCULAR_3D_CODE_SINGLE_ELEMENT_F:
                code_word_unresolved_param = (code_word_res, scan_idx, next_element_code)
            else:
                code_delta_segment = np.concatenate((code_delta_segment, code_delta_segment_right))
            break
    return code_delta_segment, scan_idx, code_word_unresolved_param, ocular_code_element_base


def monocular_3d_code_parse(image_seg_delta):
    scan_idx = 0
    image_seg_delta_len = image_seg_delta.size
    monocular_3d_code_seg_list = []
    monocular_3d_code_element_list = []
    code_word_unresolved_param = None
    while scan_idx < image_seg_delta_len:
        if code_word_unresolved_param is None:
            code_word_res, scan_idx, next_element_code = monocular_3d_code_word_extraction(image_seg_delta, image_seg_delta_len, scan_idx)
        else:
            code_word_res = code_word_unresolved_param[MONOCULAR_3D_CODE_WORD_PARAM_WORD_TUPLE_IDX]
            scan_idx = code_word_unresolved_param[MONOCULAR_3D_CODE_WORD_PARAM_WORD_SCAN_INDEX_IDX]
            next_element_code = code_word_unresolved_param[MONOCULAR_3D_CODE_WORD_PARAM_NEXT_CODE_ELEMENT_IDX]
        code_element = code_word_res[MONOCULAR_3D_CODE_WORD_TUPLE_CODE_ELEMENT_IDX]
        if code_element == MONOCULAR_3D_CODE_SINGLE_ELEMENT_F or next_element_code == MONOCULAR_3D_CODE_SINGLE_ELEMENT_F:
            if code_element == MONOCULAR_3D_CODE_SINGLE_ELEMENT_F:
                code_delta_segment = code_word_res[MONOCULAR_3D_CODE_WORD_TUPLE_ELEMENT_VAL_DELTA_LIST_IDX]
                code_seg_element = code_word_res[MONOCULAR_3D_CODE_WORD_TUPLE_CODE_ELEMENT_IDX]
                monocular_3d_code_seg_list.append(code_delta_segment)
                monocular_3d_code_element_list.append(code_seg_element)
            code_delta_segment, scan_idx, code_word_unresolved_param, ocular_code_element_base = monocular_3d_code_ocular_seg_proc(image_seg_delta, image_seg_delta_len, code_word_res, scan_idx, next_element_code)
            monocular_3d_code_seg_list.append(code_delta_segment)
            monocular_3d_code_element_list.append(ocular_code_element_base)
        else:
            code_delta_segment = code_word_res[MONOCULAR_3D_CODE_WORD_TUPLE_ELEMENT_VAL_DELTA_LIST_IDX]
            code_seg_element = code_word_res[MONOCULAR_3D_CODE_WORD_TUPLE_CODE_ELEMENT_IDX]
            monocular_3d_code_seg_list.append(code_delta_segment)
            monocular_3d_code_element_list.append(code_seg_element)
            code_word_unresolved_param = None
    if code_word_unresolved_param is not None:
        code_word_res = code_word_unresolved_param[MONOCULAR_3D_CODE_WORD_PARAM_WORD_TUPLE_IDX]
        code_delta_segment = code_word_res[MONOCULAR_3D_CODE_WORD_TUPLE_ELEMENT_VAL_DELTA_LIST_IDX]
        code_seg_element = code_word_res[MONOCULAR_3D_CODE_WORD_TUPLE_CODE_ELEMENT_IDX]
        monocular_3d_code_seg_list.append(code_delta_segment)
        monocular_3d_code_element_list.append(code_seg_element)
    return monocular_3d_code_seg_list, monocular_3d_code_element_list


def monocular_3d_code_curve_linear_diff(code_delta_segment, pixel_num):
    code_delta_segment_len = np.sum(code_delta_segment)
    baseline_delta = code_delta_segment_len / pixel_num
    i = 0
    curve_val = 0
    baseline_val = 0
    linear_diff_metric = 0
    while i < pixel_num:
        curve_val += code_delta_segment[i]
        baseline_val += baseline_delta
        linear_diff = curve_val - baseline_val
        linear_diff_metric += linear_diff
        i += 1
    return linear_diff_metric


def monocular_3d_code_shape_metric(code_delta_segment, code_seg_element=-1):
    pixel_num = code_delta_segment.size
    if pixel_num <= 1:
        code_shape_metric = 0
    else:
        code_shape_metric = monocular_3d_code_curve_linear_diff(code_delta_segment, pixel_num)
        code_shape_metric = code_shape_metric * code_seg_element
    if code_shape_metric == -0:
        code_shape_metric = 0
    code_shape = np.sign(code_shape_metric)
    return code_shape_metric, code_shape, pixel_num + 1


def monocular_3d_code_z_curve_shape_metric(code_word_res_left, code_word_res_ocular, code_word_res_right):
    ocular_code_delta_seg = code_word_res_ocular[MONOCULAR_3D_CODE_WORD_TUPLE_ELEMENT_VAL_DELTA_LIST_IDX]
    ocular_code_delta_element = code_word_res_ocular[MONOCULAR_3D_CODE_WORD_TUPLE_CODE_ELEMENT_IDX]
    code_word_res_left_delta_seg = code_word_res_left[MONOCULAR_3D_CODE_WORD_TUPLE_ELEMENT_VAL_DELTA_LIST_IDX]
    code_word_res_right_delta_seg = code_word_res_right[MONOCULAR_3D_CODE_WORD_TUPLE_ELEMENT_VAL_DELTA_LIST_IDX]
    code_word_res_left_delta_seg_size = code_word_res_left_delta_seg.size
    code_word_res_right_delta_seg_size = code_word_res_right_delta_seg.size
    code_flat_diff = code_word_res_right_delta_seg_size - code_word_res_left_delta_seg_size
    code_flat_diff *= ocular_code_delta_element
    code_flat_shape = np.sign(code_flat_diff)
    if code_flat_shape != MONOCULAR_3D_SHAPE_FLAT:
        code_flat_shape *= -1
    code_shape_metric, code_shape, pixel_num = monocular_3d_code_shape_metric(ocular_code_delta_seg)
    if code_shape == code_flat_shape or code_shape == MONOCULAR_3D_SHAPE_FLAT:
        #code_shape_metric = np.fmax(code_word_res_left_delta_seg_size, code_word_res_right_delta_seg_size)/np.fmin(code_word_res_left_delta_seg_size, code_word_res_right_delta_seg_size)
        #code_shape_metric *= code_flat_shape
        z_curve_code_seg = np.concatenate((code_word_res_left_delta_seg, ocular_code_delta_seg, code_word_res_right_delta_seg))
        code_shape_metric, code_shape, pixel_num = monocular_3d_code_shape_metric(z_curve_code_seg)
        #pixel_num = pixel_num + code_word_res_left_delta_seg_size + code_word_res_right_delta_seg_size
    return code_shape_metric, code_shape, pixel_num


def monocular_3d_code_shape_analysis(code_delta_segment_list, code_seg_element_list):
    code_shape_metric_list = []
    code_shape_list = []
    pixel_num_list = []
    seg_num = len(code_seg_element_list)
    i = 0
    while i < seg_num:
        code_delta_segment = code_delta_segment_list[i]
        code_seg_element = code_seg_element_list[i]
        code_shape_metric, code_shape, pixel_num = monocular_3d_code_shape_metric(code_delta_segment, code_seg_element)
        code_shape_metric_list.append(code_shape_metric)
        code_shape_list.append(code_shape)
        pixel_num_list.append(pixel_num)
        i += 1
    code_shape_metric_list = np.array(code_shape_metric_list)
    code_shape_list = np.array(code_shape_list)
    pixel_num_list = np.array(pixel_num_list)
    return code_shape_metric_list, code_shape_list, pixel_num_list


def monocular_3d_code_preamble_scan(image_seg_delta, image_seg_delta_len, scan_start):
    scan_idx = scan_start
    while scan_idx < image_seg_delta_len and image_seg_delta[scan_idx] != 0:
        scan_idx += 1
    code_shape_metric = 0
    code_shape = MONOCULAR_3D_SHAPE_FLAT
    pixel_num = scan_idx - scan_start + 1
    return scan_idx, code_shape_metric, code_shape, pixel_num


def monocular_3d_code_visible_preamble_scan(image_seg_delta, image_seg_delta_len, scan_start):
    scan_idx = scan_start
    code_word_res = (0, 0, [], -2)
    code_word_len = 0
    while scan_idx < image_seg_delta_len:
        code_word_res, scan_idx, next_element_code = monocular_3d_code_word_extraction(image_seg_delta, image_seg_delta_len, scan_idx)
        code_word_len += code_word_res[MONOCULAR_3D_CODE_WORD_TUPLE_ELEMENT_LEN_IDX]
        code_word_shape = code_word_res[MONOCULAR_3D_CODE_WORD_TUPLE_CODE_ELEMENT_IDX]
        if code_word_shape == MONOCULAR_3D_CODE_SINGLE_ELEMENT_F or next_element_code == MONOCULAR_3D_SHAPE_FLAT:
            break
    code_shape_metric = 0
    code_shape = MONOCULAR_3D_SHAPE_FLAT
    code_word_res_len_last = code_word_res[MONOCULAR_3D_CODE_WORD_TUPLE_ELEMENT_LEN_IDX]
    scan_idx -= code_word_res_len_last
    pixel_num = code_word_len - code_word_res_len_last + 1  # scan_idx - scan_start + 1
    if pixel_num == 1:
        pixel_num = 0
    return scan_idx, code_shape_metric, code_shape, pixel_num, code_word_res


def fill_shape_info(code_shape_metric_list, code_shape_list, pixel_num_list, code_shape_metric, code_shape, pixel_num):
    code_shape_metric_list.append(code_shape_metric)
    code_shape_list.append(code_shape)
    pixel_num_list.append(pixel_num)
    return code_shape_metric_list, code_shape_list, pixel_num_list


def monocular_3d_code_ocular_word_coherency_check(code_delta_segment_diff, code_delta_segment_diff_len):
    scan_idx = 0
    code_seg_diff = code_delta_segment_diff[scan_idx]
    base_seg_diff_direction = np.sign(code_seg_diff)
    scan_idx += 1
    is_base_zero_check = True
    while scan_idx < code_delta_segment_diff_len:
        code_seg_diff = code_delta_segment_diff[scan_idx]
        code_seg_diff_direction = np.sign(code_seg_diff)
        if is_base_zero_check and base_seg_diff_direction == 0 and code_seg_diff_direction != 0:
            base_seg_diff_direction = code_seg_diff_direction
            is_base_zero_check = False
        if code_seg_diff_direction != base_seg_diff_direction:
            break
        scan_idx += 1
    ocular_seg_coherent_sign = code_delta_segment_diff_len - scan_idx
    return ocular_seg_coherent_sign, base_seg_diff_direction


def monocular_3d_code_ocular_word_proc(code_word_res, ocular_seg_coherent_sign, base_seg_diff_direction, scan_idx):
    start_code = code_word_res[MONOCULAR_3D_CODE_WORD_TUPLE_CODE_ELEMENT_IDX]
    element_len = code_word_res[MONOCULAR_3D_CODE_WORD_TUPLE_ELEMENT_LEN_IDX]
    element_val_delta_list = code_word_res[MONOCULAR_3D_CODE_WORD_TUPLE_ELEMENT_VAL_DELTA_LIST_IDX]
    monocular_3d_code_word_delta_diff_direction = base_seg_diff_direction
    if ocular_seg_coherent_sign != MONOCULAR_3D_CODE_OCULAR_COHERENT:
        element_len -= ocular_seg_coherent_sign
        ##scan_idx = scan_idx - ocular_seg_coherent_sign
        element_val_delta_list = element_val_delta_list[0:element_len]
    code_word_res = (start_code, element_len, element_val_delta_list, monocular_3d_code_word_delta_diff_direction)
    return code_word_res, scan_idx


def monocular_3d_code_ocular_word_extraction(image_seg_delta, image_seg_delta_len, start_idx):
    code_word_res, scan_idx, next_element_code = monocular_3d_code_word_extraction(image_seg_delta, image_seg_delta_len, start_idx)
    scan_idx_base = scan_idx
    code_delta_segment = code_word_res[MONOCULAR_3D_CODE_WORD_TUPLE_ELEMENT_VAL_DELTA_LIST_IDX]
    code_delta_segment_len = code_delta_segment.size
    ocular_seg_coherent_sign = MONOCULAR_3D_CODE_OCULAR_COHERENT
    if code_delta_segment_len > 1:
        code_delta_segment_diff = np.diff(code_delta_segment)
        ocular_seg_coherent_sign, base_seg_diff_direction = monocular_3d_code_ocular_word_coherency_check(code_delta_segment_diff, code_delta_segment_len-1)
        code_word_res, scan_idx = monocular_3d_code_ocular_word_proc(code_word_res, ocular_seg_coherent_sign, base_seg_diff_direction,scan_idx)
        if scan_idx != scan_idx_base:
            next_element_code = code_word_res[MONOCULAR_3D_CODE_WORD_TUPLE_CODE_ELEMENT_IDX]
    else:
        base_seg_diff_direction = MONOCULAR_3D_CODE_WORD_DELTA_DIFF_DIRECTION_NONE
        code_word_res, scan_idx = monocular_3d_code_ocular_word_proc(code_word_res, MONOCULAR_3D_CODE_OCULAR_COHERENT, base_seg_diff_direction, scan_idx)
    return code_word_res, scan_idx, next_element_code, ocular_seg_coherent_sign


def fill_flat_shape_info(code_shape_metric_list, code_shape_list, pixel_num_list, code_word_res):
    code_shape_metric = 0
    code_shape = MONOCULAR_3D_SHAPE_FLAT
    code_delta_segment = code_word_res[MONOCULAR_3D_CODE_WORD_TUPLE_ELEMENT_VAL_DELTA_LIST_IDX]
    pixel_num = code_delta_segment.size
    pixel_num += 1
    fill_shape_info(code_shape_metric_list, code_shape_list, pixel_num_list, code_shape_metric, code_shape, pixel_num)
    return code_shape_metric_list, code_shape_list, pixel_num_list


def monocular_3d_code_parse_analysis(image_seg_delta):
    scan_idx = 0
    image_seg_delta_len = image_seg_delta.size
    image_seg_delta = img_intensity_precision_proc(image_seg_delta)
    code_shape_metric_list = []
    code_shape_list = []
    pixel_num_list = []
    #scan_idx, code_shape_metric, code_shape, pixel_num = monocular_3d_code_preamble_scan(image_seg_delta, image_seg_delta_len, scan_idx)
    #if pixel_num != 1:
        #fill_shape_info(code_shape_metric_list, code_shape_list, pixel_num_list, code_shape_metric, code_shape, pixel_num)
    scan_idx, code_shape_metric, code_shape, pixel_num, code_word_res_last = monocular_3d_code_visible_preamble_scan(image_seg_delta, image_seg_delta_len, scan_idx)
    if pixel_num > 1:
        fill_shape_info(code_shape_metric_list, code_shape_list, pixel_num_list, code_shape_metric, code_shape, pixel_num)
        code_word_res_ocular, scan_idx, next_element_code, ocular_seg_coherent_sign = monocular_3d_code_ocular_word_extraction(image_seg_delta, image_seg_delta_len, scan_idx)
        code_delta_segment = code_word_res_ocular[MONOCULAR_3D_CODE_WORD_TUPLE_ELEMENT_VAL_DELTA_LIST_IDX]
        code_shape_metric, code_shape, pixel_num = monocular_3d_code_shape_metric(code_delta_segment)
        fill_shape_info(code_shape_metric_list, code_shape_list, pixel_num_list, code_shape_metric, code_shape, pixel_num)
    current_state = MONOCULAR_3D_CODE_SCAN_STATE_FLAT_LEFT
    #  Scan will complete when scan_idx reaches the end of sequence. Therefore, it is not necessary to check scan_idx when changing current_state.
    while scan_idx < image_seg_delta_len:
        if current_state == MONOCULAR_3D_CODE_SCAN_STATE_FLAT_LEFT:
            code_word_res_left, scan_idx, next_element_code = monocular_3d_code_word_extraction(image_seg_delta, image_seg_delta_len, scan_idx)
            fill_flat_shape_info(code_shape_metric_list, code_shape_list, pixel_num_list, code_word_res_left)
            current_state = MONOCULAR_3D_CODE_SCAN_STATE_OCULAR
        elif current_state == MONOCULAR_3D_CODE_SCAN_STATE_OCULAR:
            code_word_res_ocular, scan_idx, next_element_code, ocular_seg_coherent_sign = monocular_3d_code_ocular_word_extraction(image_seg_delta, image_seg_delta_len, scan_idx)
            if ocular_seg_coherent_sign != MONOCULAR_3D_CODE_OCULAR_COHERENT or next_element_code != MONOCULAR_3D_SHAPE_FLAT:
                code_delta_segment = code_word_res_ocular[MONOCULAR_3D_CODE_WORD_TUPLE_ELEMENT_VAL_DELTA_LIST_IDX]
                code_shape_metric, code_shape, pixel_num = monocular_3d_code_shape_metric(code_delta_segment)
                fill_shape_info(code_shape_metric_list, code_shape_list, pixel_num_list, code_shape_metric, code_shape, pixel_num)
                if ocular_seg_coherent_sign != MONOCULAR_3D_CODE_OCULAR_COHERENT:
                    current_state = MONOCULAR_3D_CODE_SCAN_STATE_FLAT_LEFT
            if next_element_code != MONOCULAR_3D_SHAPE_FLAT:
                current_state = MONOCULAR_3D_CODE_SCAN_STATE_OCULAR_OPPOSITE
            elif ocular_seg_coherent_sign == MONOCULAR_3D_CODE_OCULAR_COHERENT:
                current_state = MONOCULAR_3D_CODE_SCAN_STATE_FLAT_RIGHT
        elif current_state == MONOCULAR_3D_CODE_SCAN_STATE_OCULAR_OPPOSITE:
            #scan_idx, code_shape_metric, code_shape, pixel_num = monocular_3d_code_preamble_scan(image_seg_delta, image_seg_delta_len, scan_idx)
            scan_idx, code_shape_metric, code_shape, pixel_num, code_word_res_last = monocular_3d_code_visible_preamble_scan(image_seg_delta, image_seg_delta_len, scan_idx)
            if pixel_num > 1:
                fill_shape_info(code_shape_metric_list, code_shape_list, pixel_num_list, code_shape_metric, code_shape, pixel_num)
            code_word_res_ocular, scan_idx, next_element_code, ocular_seg_coherent_sign = monocular_3d_code_ocular_word_extraction(image_seg_delta, image_seg_delta_len, scan_idx)
            code_delta_segment = code_word_res_ocular[MONOCULAR_3D_CODE_WORD_TUPLE_ELEMENT_VAL_DELTA_LIST_IDX]
            code_shape_metric, code_shape, pixel_num = monocular_3d_code_shape_metric(code_delta_segment)
            fill_shape_info(code_shape_metric_list, code_shape_list, pixel_num_list, code_shape_metric, code_shape, pixel_num)
            current_state = MONOCULAR_3D_CODE_SCAN_STATE_FLAT_LEFT
        elif current_state == MONOCULAR_3D_CODE_SCAN_STATE_FLAT_RIGHT:
            code_word_res_right, scan_idx, next_element_code = monocular_3d_code_word_extraction(image_seg_delta, image_seg_delta_len, scan_idx)
            code_shape_metric, code_shape, pixel_num = monocular_3d_code_z_curve_shape_metric(code_word_res_left, code_word_res_ocular, code_word_res_right)
            fill_shape_info(code_shape_metric_list, code_shape_list, pixel_num_list, code_shape_metric, code_shape, pixel_num)
            fill_flat_shape_info(code_shape_metric_list, code_shape_list, pixel_num_list, code_word_res_right)
            code_word_res_left = code_word_res_right
            current_state = MONOCULAR_3D_CODE_SCAN_STATE_OCULAR
    code_shape_metric_list = np.array(code_shape_metric_list)
    code_shape_list = np.array(code_shape_list)
    pixel_num_list = np.array(pixel_num_list)
    return code_shape_metric_list, code_shape_list, pixel_num_list


def fill_one_dim_array_grad_sum(grad_sum_array, grad_sum, fill_num):
    i = 0
    while i < fill_num:
        grad_sum_array.append(grad_sum)
        i += 1
    return grad_sum_array


def extract_fill_num(fill_num_array, fill_num_idx):
    if fill_num_array is not None:
        fill_num = fill_num_array[fill_num_idx]
    else:
        fill_num = 1
    return fill_num


def one_dim_array_gard_sum(one_dim_array, fill_num_array=None):
    grad_sum_array = []
    array_len = one_dim_array.size
    i = 0
    grad_sum = 0
    while i < array_len:
        grad_sum += one_dim_array[i]
        fill_num = extract_fill_num(fill_num_array, i)
        #grad_sum_array.append(grad_sum)
        fill_one_dim_array_grad_sum(grad_sum_array, grad_sum, fill_num)
        i += 1
    grad_sum_array = np.array(grad_sum_array)
    return grad_sum_array


def img_gray_avg_pool(img_gray, k_size=(10, 3), sd=(2, 2)):
    img_gray_torch = torch.from_numpy(img_gray)
    img_gray_torch = torch.unsqueeze(img_gray_torch, 0)
    pool_model = torch.nn.AvgPool2d(kernel_size=k_size, stride=sd)
    img_gray_torch_pool = pool_model(img_gray_torch)
    img_gray_torch_pool = torch.squeeze(img_gray_torch_pool)
    img_gray_pool = img_gray_torch_pool.numpy()
    return img_gray_pool


def generate_fun_grating_img(img_line_num, grating_fun, grating_fun_params):
    scan_line_info = grating_fun(grating_fun_params)
    #scan_line_info = np.round(scan_line_info, 2)
    scan_line = scan_line_info[0]
    img_col_num = scan_line_info[1]
    img_shape = (img_line_num, img_col_num)
    img_fun_grating = np.zeros(img_shape)
    i = 0
    while i < img_line_num:
        img_fun_grating[i, 0:img_col_num] = scan_line
        i += 1
    return[img_fun_grating, img_shape]


def wrap_sinusodial_grating_fun_param(f, m, theta, l_0, cycle_width, img_width, triangle_fun):
    cycle_num = np.int_(img_width/cycle_width)
    grating_fun_param = [f, m, theta, l_0, cycle_width, cycle_num, triangle_fun]
    return grating_fun_param


def generate_sinusodial_scan_line_cycle_fun(fun_param):
    f = fun_param[SINUNO_FUN_PARAM_F_IDX]
    cycle_width = fun_param[SINUNO_FUN_PARAM_CYCLE_WIDTH_IDX]
    cycle_num = fun_param[SINUNO_FUN_PARAM_CYCLE_NUM_IDX]
    theta = fun_param[SINUNO_FUN_PARAM_THETA_IDX]
    l_0 = fun_param[SINUNO_FUN_PARAM_L0_IDX]
    m = fun_param[SINUNO_FUN_PARAM_M_IDX]
    triangle_fun_idx = fun_param[SINUNO_FUN_PARAM_TRIANGLE_FUN_IDX]
    triangle_fun = SINUNO_TRIANGLE_FUN_LIST[triangle_fun_idx]
    tpi_f_len = 2*np.pi/f
    x_unit = tpi_f_len/cycle_width
    all_cycle_len = cycle_width*cycle_num
    #x_len = np.int_(np.ceil(all_cycle_len/x_unit))
    scan_line_info = generate_sinusodial_scan_line(all_cycle_len, x_unit, f, theta, l_0, m, triangle_fun)
    return [scan_line_info[0], all_cycle_len]



def wrap_square_grating_fun_param(l_0, v_abs_max, cycle_width, img_width, half_cycle_width, const_value=SQUARE_GRATING_CONST_AVG):
    cycle_num = np.int_(img_width / cycle_width)
    grating_fun_param = [l_0, v_abs_max, cycle_num, half_cycle_width, const_value]
    return grating_fun_param


def generate_square_scan_line(fun_param):
    l_0 = fun_param[SQUARE_GRATING_L0_IDX]
    v_abs_max = fun_param[SQUARE_GRATING_V_ABS_MAX_IDX]
    cycle_num = fun_param[SQUARE_GRATING_CYCLE_NUM]
    half_cycle_width = fun_param[SQUARE_GRATING_HALF_CYCLE_WIDTH]
    const_val = fun_param[SQUARE_GRATING_CONST_VAL_IDX]
    all_cycle_len = np.int_(2*cycle_num*half_cycle_width)
    scan_line = np.zeros(all_cycle_len)
    if const_val == SQUARE_GRATING_CONST_AVG:
        line_value_low = l_0 - v_abs_max
        line_value_high = line_value_low + 2*v_abs_max
    else:
        line_value_low = l_0
        line_value_high = line_value_low + v_abs_max
    line_value_list = [line_value_high, line_value_low]
    i = 0
    while i < all_cycle_len:
        val_index = np.int_(np.int_(i/half_cycle_width) % 2)
        scan_line[i] = line_value_list[val_index]
        i += 1
    scan_line = np.round(scan_line, 4)
    return [scan_line, all_cycle_len]


def opt_result_to_threshold_list(best_x, is_img_int=False):
    g_best_x = np.sort(np.round(best_x, 0))[::-1]
    if g_best_x[len(g_best_x) - 1] > 0:
        g_best_x = np.append(g_best_x, 0)
    if is_img_int is not True:
        threshold_list = g_best_x / 255
    else:
        threshold_list = g_best_x
    return threshold_list


def img_sim_hist_shape_measure_rgb(img_original, img_reconstruction, multi_threshold_integer_int_list, hist_shape_coef=0.4):
    i = 0
    hist_shape_measure = 0
    while i < 3:
        img_plane_org = img_original[:, :, i]
        img_plane_rec = img_reconstruction[:, :, i]
        threshold_list_int = multi_threshold_integer_int_list[i]
        hist_shape_measure += img_sim_hist_shape_measure(img_plane_org, img_plane_rec, threshold_list_int, hist_shape_coef)
        i += 1
    hist_shape_measure = hist_shape_measure/3
    return hist_shape_measure


def img_sim_hist_shape_measure(img_original, img_reconstruction, threshold_list_int, hist_shape_coef=0.4):
    img_whole_range_unique_value_scale_org, img_whole_range_unique_value_count_list_org = img_whole_range_unique_value_count(img_original)
    img_whole_range_unique_value_scale_rec, img_whole_range_unique_value_count_list_rec = img_whole_range_unique_value_count(img_reconstruction)
    #img_whole_range_unique_value_scale_org *= 255
    #img_whole_range_unique_value_scale_rec *= 255
    #hist_mean_org = np.sum(img_whole_range_unique_value_scale_org*img_whole_range_unique_value_count_list_org)
    #hist_mean_org = hist_mean_org/np.sum(img_whole_range_unique_value_count_list_org)
    #hist_mean_rec = np.sum(img_whole_range_unique_value_scale_rec*img_whole_range_unique_value_count_list_rec)
    #hist_mean_rec = hist_mean_rec/np.sum(img_whole_range_unique_value_count_list_rec)
    hist_dyna_org = np.max(img_whole_range_unique_value_scale_org) - np.min(img_whole_range_unique_value_scale_org)
    hist_dyna_rec = np.max(img_whole_range_unique_value_scale_rec) - np.min(img_whole_range_unique_value_scale_rec)
    img_whole_range_unique_value_count_list_rec = img_whole_range_unique_value_count_list_rec[img_whole_range_unique_value_count_list_rec!=0]
    img_org_rec_sample_index = threshold_list_int - img_whole_range_unique_value_scale_rec[0]
    img_org_rec_sample_index = img_org_rec_sample_index[::-1]
    sample_len = img_org_rec_sample_index.size - 1
    img_rec_hist_diff_vectors = np.zeros((sample_len, 2))
    img_org_hist_diff_vectors = np.zeros((sample_len, 2))
    while img_whole_range_unique_value_count_list_rec.size < sample_len + 1:
        img_whole_range_unique_value_count_list_rec = np.append(img_whole_range_unique_value_count_list_rec, [0])
    img_rec_hist_diff_vectors[:, 0] = np.diff(img_whole_range_unique_value_count_list_rec)
    rec_threshold_coef = np.max(img_whole_range_unique_value_scale_org) - np.min(img_whole_range_unique_value_scale_org)
    rec_threshold_coef = rec_threshold_coef/sample_len
    img_rec_hist_diff_vectors[:, 1] = np.diff(np.sort(threshold_list_int))*rec_threshold_coef
    img_org_hist_diff_vectors[:, 1] = img_rec_hist_diff_vectors[:, 1]
    i = 0
    while i < sample_len:
        hist_sample_l = img_org_rec_sample_index[i]
        hist_sample_u = img_org_rec_sample_index[i+1]
        if hist_sample_u >= img_whole_range_unique_value_count_list_org.size:
            hist_sample_u = img_whole_range_unique_value_count_list_org.size - 1
        if hist_sample_l >= img_whole_range_unique_value_count_list_org.size:
            hist_sample_l = img_whole_range_unique_value_count_list_org.size - 1
        img_org_hist_diff_vectors[i, 0] = img_whole_range_unique_value_count_list_org[hist_sample_u] - img_whole_range_unique_value_count_list_org[hist_sample_l]
        i += 1
    hist_shape_measure = np.diag(np.inner(img_org_hist_diff_vectors, img_rec_hist_diff_vectors))
    vector_norm_org = np.linalg.norm(img_org_hist_diff_vectors, axis=1)
    vector_norm_rec = np.linalg.norm(img_rec_hist_diff_vectors, axis=1)
    vector_norm_org = np.where(vector_norm_org == 0, 1, vector_norm_org)
    vector_norm_rec = np.where(vector_norm_rec == 0, 1, vector_norm_rec)
    #print(vector_norm_org)
    hist_shape_measure = hist_shape_measure/vector_norm_rec
    hist_shape_measure = hist_shape_measure/vector_norm_org
    #print(hist_shape_measure)
    hist_shape_measure = np.where(hist_shape_measure >= 1, 1, hist_shape_measure)
    hist_shape_measure = np.where(hist_shape_measure <= -1, -1, hist_shape_measure)
    hist_shape_measure = np.arccos(hist_shape_measure)
    hist_shape_measure = np.sum(hist_shape_measure)
    hist_shape_measure = hist_shape_measure/(sample_len*np.pi)
    hist_shape_measure = 1 - hist_shape_measure
    hist_dyna_diff_ratio = 1 - np.abs(hist_dyna_org - hist_dyna_rec) / hist_dyna_org
    hist_shape_measure = hist_shape_coef*hist_shape_measure + (1 - hist_shape_coef)*hist_dyna_diff_ratio
    return hist_shape_measure


def cal_img_skewness(img):
    m_img = np.mean(img)
    std_img = np.std(img)
    N_img = img.size
    v_cube_img = np.sum(np.power(img - m_img, 3))
    skewness_img = v_cube_img / (N_img * np.power(std_img, 3))
    return skewness_img


def cal_img_dyna(img):
    img_dyna = np.max(img) - np.min(img)
    return img_dyna


def img_sim_hist_shape_measure_skewness(img_original, img_reconstruction, threshold_list_int, hist_skewness_coef=0.4):
    img_original_int = np.int_(np.round(img_original*INT_GRAY_LEVEL_BAR))
    img_rec_int = np.int_(np.round(img_reconstruction * INT_GRAY_LEVEL_BAR))
    img_shape = img_rec_int.shape
    img_skew_right = np.ones(img_shape)*INT_GRAY_LEVEL_BAR
    img_skew_right[0] = INT_GRAY_LEVEL_BAR - 1
    img_skew_left = np.zeros(img_shape)
    img_skew_left[0] = 1
    skewness_org = cal_img_skewness(img_original_int)
    skewness_rec = cal_img_skewness(img_rec_int)
    skewness_right = cal_img_skewness(img_skew_right)
    skewness_left = cal_img_skewness(img_skew_left)
    skewness_diff_ratio = np.abs(skewness_rec - skewness_org)/np.abs(skewness_right - skewness_left)
    hist_dyna_org = cal_img_dyna(img_original)
    hist_dyna_rec = cal_img_dyna(img_reconstruction)
    hist_dyna_diff_ratio = 1 - np.abs(hist_dyna_org - hist_dyna_rec) / hist_dyna_org
    hist_shape_measure = hist_skewness_coef * (1 - skewness_diff_ratio) + (1 - hist_skewness_coef) * hist_dyna_diff_ratio
    return hist_shape_measure


def hist_feature_hybrid_sim(hist_sim, fsim, ssim, hist_sim_ratio=0.1, fsim_ratio=0.9, ssim_ratio=0):#0.3 0.7
    hfsim = hist_sim_ratio*hist_sim + fsim*fsim_ratio + ssim*(1 - fsim_ratio - hist_sim_ratio)
    #hfsim = hist_sim*fsim
    return hfsim


def cartesian_mul_mean(a, b, arr_size):
    i = 0
    ret = 0
    while i < arr_size:
        element_mul = a[i] - b[i:arr_size]
        ret += np.power(np.sum(np.abs(element_mul)), 0.5)
        i += 1
    ret = ret/arr_size
    return ret


def img_gray_one_dim_fourier_transform(img_gray, array_idx, dim_direction=IMG_GRAY_DIM_HORIZONTAL, dim_freq_scaler=1):
    if dim_direction == IMG_GRAY_DIM_HORIZONTAL:
        dim_array = img_gray[array_idx, :]
    else:
        dim_array = img_gray[:, array_idx]
    array_fft_ret = np.fft.fft(dim_array)
    array_fft_freq_ret = np.fft.fftfreq(dim_array.size) * dim_freq_scaler
    array_fft_abs_ret = np.abs(array_fft_ret)
    return array_fft_ret, array_fft_freq_ret, array_fft_abs_ret


def draw_illusion_img(central_index_hor, central_index_ver, peri_index_hor_adj, peri_index_ver_adj):
    illusion_img = np.ones((1000, 1000)) * 115#SAME_SWITCH_ILLU:#100#50#120#115#90#60#30#85 #NO_ILLU:#180#220#160 #TRI_DIFF_ILLU#140
    illusion_img[central_index_ver[0]:central_index_ver[1], central_index_hor[0]:central_index_hor[1]] = 155#154#151
    illusion_img[central_index_ver[0]:central_index_ver[1], peri_index_hor_adj[0]:peri_index_hor_adj[1]] = 150#149#147#145#147#180#140#180#152
    illusion_img[peri_index_ver_adj[0]:peri_index_ver_adj[1], central_index_hor[0]:central_index_hor[1]] = 150#149#147#145#147#180#140#180#152
    return illusion_img


def contrast_threshold_illusion_generation(horizontal_direction=CONT_THRESH_ILLUSION_HOR_RIGHT, vertical_direction=CONT_THRESH_ILLUSION_VER_TOP, illusion_direction=CONT_THRESH_ILLUSION_VER):
    central_index_hor_low = 440
    central_index_hor_high = 520#480
    central_index_ver_low = 440
    central_index_ver_high = 520#480
    if horizontal_direction == CONT_THRESH_ILLUSION_HOR_LEFT:
        hor_adj_offset = -1
    else:
        hor_adj_offset = 1
    if vertical_direction == CONT_THRESH_ILLUSION_VER_TOP:
        ver_adj_offset = -1
    else:
        ver_adj_offset = 1
    peri_index_hor_adj_low = central_index_hor_low + hor_adj_offset*80
    peri_index_hor_adj_high = central_index_hor_high + hor_adj_offset*80
    peri_index_ver_adj_low = central_index_hor_low + ver_adj_offset*80
    peri_index_ver_adj_high = central_index_hor_high + ver_adj_offset*80
    central_index_hor = (central_index_hor_low, central_index_hor_high)
    central_index_ver = (central_index_ver_low, central_index_ver_high)
    peri_index_hor_adj = (peri_index_hor_adj_low, peri_index_hor_adj_high)
    peri_index_ver_adj = (peri_index_ver_adj_low, peri_index_ver_adj_high)
    no_illusion_img = draw_illusion_img(central_index_hor, central_index_ver, peri_index_hor_adj, peri_index_ver_adj)
    if illusion_direction == CONT_THRESH_ILLUSION_HOR:
        peri_index_hor_adj_low = peri_index_hor_adj_low + hor_adj_offset*20
        peri_index_hor_adj_high = peri_index_hor_adj_high + hor_adj_offset*20
        peri_index_hor_adj = (peri_index_hor_adj_low, peri_index_hor_adj_high)
    else:
        peri_index_ver_adj_low = peri_index_ver_adj_low + ver_adj_offset*20
        peri_index_ver_adj_high = peri_index_ver_adj_high + ver_adj_offset*20
        peri_index_ver_adj = (peri_index_ver_adj_low, peri_index_ver_adj_high)
    illusion_img = draw_illusion_img(central_index_hor, central_index_ver, peri_index_hor_adj, peri_index_ver_adj)
    return no_illusion_img, illusion_img


def combine_identical_image_ver_format(img_list, img_ver_space=80):
    img_num = len(img_list)
    single_img = img_list[0]
    single_img_shape = single_img.shape
    img_width = single_img_shape[1]
    single_img_height = single_img_shape[0]
    img_height = single_img_height * img_num + img_ver_space * (img_num - 1)
    combined_img = np.zeros((img_height, img_width))
    i = 0
    img_insertion_line = 0
    while i < img_num:
        single_img = img_list[i]
        combined_img[img_insertion_line:img_insertion_line+single_img_height, :] = single_img
        img_insertion_line += single_img_height
        img_insertion_line += img_ver_space
        i += 1
    return combined_img


def generate_gradient_grating_img(img_line_num, grating_fun, grating_fun_params, gradient_step_unit, gradient_step_num, gradient_param_idx):
    i = 0
    gradient_img_list = []
    m = grating_fun_params[gradient_param_idx]
    while i < gradient_step_num:
        square_grating_img_info = generate_fun_grating_img(img_line_num, grating_fun, grating_fun_params)
        square_grating_img = square_grating_img_info[0]
        gradient_img_list.append(square_grating_img)
        m += gradient_step_unit
        grating_fun_params[gradient_param_idx] = m
        i += 1
    combined_gradient_img = combine_identical_image_ver_format(gradient_img_list)
    return gradient_img_list, combined_gradient_img


def img_rgb_hist_sample(img, hist_threshold):
    multi_threshold_integer_int_list = []
    img_sampled = []
    i = 0
    while i < 3:
        img_plane = img[:, :, i]
        img_gray_distorted_filled, multi_threshold_integer_int = img_gray_hist_sample(img_plane, hist_threshold)
        multi_threshold_integer_int_list.append(multi_threshold_integer_int)
        img_gray_distorted_filled = np.expand_dims(img_gray_distorted_filled, 2)
        img_sampled.append(img_gray_distorted_filled)
        i += 1
    #multi_threshold_integer_int_list = np.array(multi_threshold_integer_int_list)
    img_sampled = np.concatenate((img_sampled[0], img_sampled[1], img_sampled[2]), axis=2)
    return img_sampled, multi_threshold_integer_int_list


def img_gray_hist_sample(img, hist_threshold):
    img_whole_range_unique_value_scale_rec, img_whole_range_unique_value_count_list_rec = img_whole_range_unique_value_count(img)
    img_whole_range_unique_value_count_list_rec = img_whole_range_unique_value_count_list_rec / np.max(img_whole_range_unique_value_count_list_rec)
    multi_threshold_integer_int = np.where(img_whole_range_unique_value_count_list_rec > hist_threshold)[0]
    multi_threshold_index_zero = np.where(img_whole_range_unique_value_count_list_rec <= hist_threshold)[0]
    multi_threshold_index_zero = np.round(multi_threshold_index_zero + np.min(img) * 255, 0)
    img_gray_distorted_filled = img_gray_fill_zeros(np.round(img * 255, 0), multi_threshold_index_zero)
    img_gray_distorted_filled = img_gray_distorted_filled / INT_GRAY_LEVEL_BAR
    return img_gray_distorted_filled, multi_threshold_integer_int


def img_gray_fill_zeros(img, zero_values):
    img_ret = img_copy(img)
    zero_value_len = zero_values.size
    i = 0
    while i < zero_value_len:
        img_ret[img_ret == zero_values[i]] = -1
        i += 1
    return img_ret


def img_hfsim_cal(img_org, img_rec, hist_threshold, is_rgb=False):
    if is_rgb:
        img_sampled, multi_threshold_integer_int_list = img_rgb_hist_sample(img_rec, hist_threshold)
        hist_shape_measure = img_sim_hist_shape_measure_rgb(img_org, img_sampled, multi_threshold_integer_int_list)
    else:
        img_gray_distorted_filled, multi_threshold_integer_int = img_gray_hist_sample(img_rec, hist_threshold)
        hist_shape_measure = img_sim_hist_shape_measure(img_org, img_gray_distorted_filled, multi_threshold_integer_int)
    #fsim = img_qm.fsim(np.round(img_org * 255, 0).astype(np.uint8), np.round(img_rec * 255).astype(np.uint8))
    fsim = imq_spt.fsim(np.round(img_org * 255, 0).astype(np.uint8), np.round(img_rec * 255).astype(np.uint8))
    ssim = img_qm.ssim(np.round(img_org * 255, 0).astype(np.uint8), np.round(img_rec * 255, 0).astype(np.uint8), max_p=255)
    hfsim = hist_feature_hybrid_sim(hist_shape_measure, fsim, ssim)
    return hfsim, fsim, ssim, hist_shape_measure


def get_txt_file_float_lines(file_path):
    lines = get_txt_file_str_lines(file_path)
    line_num = len(lines)
    float_lines = []
    i = 0
    while i < line_num:
        float_lines.append(float(lines[i]))
        i += 1
    return np.array(float_lines)


def retrieve_mos_info(mos_file_path, name_score_spliter=' '):
    mos_lines = get_txt_file_str_lines(mos_file_path)
    mos_line_num = len(mos_lines)
    score_list = []
    name_list = []
    i = 0
    while i < mos_line_num:
        mos_line = mos_lines[i]
        mos_info = mos_line.split(name_score_spliter)
        score_list.append(float(mos_info[0]))
        name_list.append(mos_info[1])
        i += 1
    scores = np.array(score_list)
    return scores, name_list


def retrieve_file_paths(ref_file_path):
    file_name_list = os.listdir(ref_file_path)
    file_path_list = []
    for entry in file_name_list:
        file_path = os.path.join(ref_file_path, entry)
        file_path_list.append(file_path)
    return file_path_list, file_name_list


def retrieve_mos_raw_images(file_path_list, is_rgb=True, img_map_idx=IMG_MAP_GRAY_IDX, is_img_int=False):
    file_path_list_len = len(file_path_list)
    raw_image_list = []
    i = 0
    while i < file_path_list_len:
        file_path = file_path_list[i]
        if is_rgb:
            raw_image = get_rgb_img_from_file(file_path)/INT_GRAY_LEVEL_BAR
        else:
            raw_image = get_gray_img_from_file(file_path, img_map_idx=img_map_idx)
        if is_img_int:
            raw_image = np.round(raw_image*INT_GRAY_LEVEL_BAR, 0)
        raw_image_list.append(raw_image)
        i += 1
    return raw_image_list


def resize_mos_images(raw_mos_images, resize_ratio=0.5, interpolation=cv2.INTER_NEAREST):
    image_num = len(raw_mos_images)
    resized_images = []
    i = 0
    while i < image_num:
        raw_image = raw_mos_images[i]
        resized_image = img_gray_resize(np.round(raw_image*INT_GRAY_LEVEL_BAR, 0), resize_ratio, interpolation=interpolation)/INT_GRAY_LEVEL_BAR
        resized_images.append(resized_image)
        i += 1
    return resized_images


def resize_mos_images_down_sample(raw_mos_images, is_rgb=True, resize_const=256):
    image_num = len(raw_mos_images)
    resized_images = []
    i = 0
    while i < image_num:
        raw_image = raw_mos_images[i]
        resized_image = img_down_sample_resize(np.round(raw_image*INT_GRAY_LEVEL_BAR, 0), resize_const, is_rgb)
        resized_image = resized_image/INT_GRAY_LEVEL_BAR
        resized_images.append(resized_image)
        i += 1
    return resized_images


def mos_images_equal_range_segmentation_reconstruction(mos_images, seg_num, is_segment_mod, is_rgb=True):
    image_num = len(mos_images)
    reconstructed_images = []
    i = 0
    while i < image_num:
        mos_image = mos_images[i]
        if is_rgb:
            seg_rec_mos_image = img_rgb_integer_segmentation_equal_range_thresholds(mos_image, seg_num, is_segment_mod=is_segment_mod)
        else:
            seg_rec_mos_image = img_gray_integer_segmentation_equal_range_reconstruction(mos_image, seg_num, is_segment_mod=is_segment_mod)
        reconstructed_images.append(seg_rec_mos_image)
        i += 1
    return reconstructed_images


def generate_mos_ref_file_map(mos_images, mos_image_names):
    image_num = len(mos_images)
    ref_image_map = dict()
    i = 0
    while i < image_num:
        mos_image_name = mos_image_names[i]
        mos_key = mos_image_name.split('.')
        mos_key = mos_key[0].upper()
        mos_image = mos_images[i]
        ref_image_map[mos_key] = mos_image
        i += 1
    return ref_image_map


def generate_mos_file_paths(mos_file_names, root_path):
    file_name_num = len(mos_file_names)
    mos_file_paths = []
    i = 0
    while i < file_name_num:
        mos_file_name = mos_file_names[i]
        file_path = os.path.join(root_path, mos_file_name)
        mos_file_paths.append(file_path)
        i += 1
    return mos_file_paths


def cal_mos_sim_scores(ref_image_map, dist_image_list, dist_image_name_list, hist_threshold, dist_image_name_spliter='_', is_rgb=True):
    dist_image_num = len(dist_image_list)
    fsims = []
    ssims = []
    hfsims = []
    hsmsims = []
    i = 0
    while i < dist_image_num:
        dist_image_name = dist_image_name_list[i]
        ref_key = dist_image_name.split(dist_image_name_spliter)[0]
        ref_key = ref_key.upper()
        ref_image = ref_image_map[ref_key]
        dist_image = dist_image_list[i]
        hfsim, fsim, ssim, hist_shape_measure = img_hfsim_cal(ref_image, dist_image, hist_threshold, is_rgb)
        fsims.append(fsim)
        ssims.append(ssim)
        hfsims.append(hfsim)
        hsmsims.append(hist_shape_measure)
        i += 1
        print("image completed:", i, dist_image_name, ref_key, " fsim:", fsim, " hfsim:", hfsim)
    fsims = np.array(fsims)
    ssims = np.array(ssims)
    hfsims = np.array(hfsims)
    hsmsims = np.array(hsmsims)
    return fsims, ssims, hfsims, hsmsims


def cal_mos_sim_metric_performance_score(mos_scores, mos_sim_metric_scores, is_regression=True):
    mos_order = get_index_array_one_dim(np.argsort(mos_scores))
    ssim_order = get_index_array_one_dim(np.argsort(mos_sim_metric_scores))
    #test
    sim_order_sub = np.abs(mos_order - ssim_order)
    sim_order_sub_max = np.argmax(sim_order_sub)
    print("sim order sum max arg:", sim_order_sub_max)
    #test end
    spear_corr, spear_pval = spearmanr(mos_order, ssim_order)
    kend_corr, kend_pval = kendalltau(mos_order, ssim_order)
    if is_regression:
        mos_sim_regression_scores = mos_sim_score_none_linear_regression_score(mos_scores, mos_sim_metric_scores)
        pearson_corr, pearson_pval = pearsonr(mos_scores, mos_sim_regression_scores)
        mos_sim_regression_rmse = mean_squared_error(mos_scores, mos_sim_regression_scores)
        mos_sim_regression_rmse = np.sqrt(mos_sim_regression_rmse)
    else:
        pearson_corr = np.nan
        mos_sim_regression_rmse = np.nan
    return spear_corr, kend_corr, pearson_corr, mos_sim_regression_rmse


def get_txt_file_str_lines(file_path):
    txt_file = open(file_path, "r", newline='')
    line_list = []
    while True:
        line = txt_file.readline()
        if not line:
            break
        if line[len(line) - 2] == '\r':
            line = line[0:len(line) - 2]
        line_list.append(line)
    txt_file.close()
    return line_list


def get_index_array_one_dim(source_array):
    array_len = source_array.size
    index_array = np.int_(np.zeros(array_len))
    i = 0
    while i < array_len:
        index_array[source_array[i]] = i
        i += 1
    return index_array


def img_down_sample_resize(img, resize_const=256, is_rgb=True):
    if is_rgb:
        img_resized = img_rgb_down_sample_resize(img, resize_const)
    else:
        img_resized = img_gray_down_sample_resize(img, resize_const)
    return img_resized


def img_rgb_down_sample_resize(img_rgb, resize_const=256):
    img_0_resized = img_gray_down_sample_resize(img_rgb[:, :, 0], resize_const)
    img_1_resized = img_gray_down_sample_resize(img_rgb[:, :, 1], resize_const)
    img_2_resized = img_gray_down_sample_resize(img_rgb[:, :, 2], resize_const)
    img_0_resized = np.expand_dims(img_0_resized, 2)
    img_1_resized = np.expand_dims(img_1_resized, 2)
    img_2_resized = np.expand_dims(img_2_resized, 2)
    img_resized = np.concatenate((img_0_resized, img_1_resized, img_2_resized), axis=2)
    return img_resized


def img_gray_down_sample_resize(img_gray, resize_const=256):
    img_shape = img_gray.shape
    row_num = img_shape[0]
    col_num = img_shape[1]
    min_dim = np.fmin(row_num, col_num)
    resize_factor = np.int_(np.fmax(1, np.round(min_dim / resize_const)))
    core = np.full((resize_factor, resize_factor), np.power(1 / resize_factor, 2))
    resized_img_gray = convolve2d(img_gray, core, mode='same')
    resized_img_gray = resized_img_gray[0:row_num:resize_factor, 0:col_num:resize_factor]
    resized_img_gray = np.round(resized_img_gray, 0)
    return resized_img_gray


def img_planes_density_convolve_sum(img_plane_list, kernel_size=(3, 3), kernel_threshold=7):
    plane_num = len(img_plane_list)
    img_plane_shape = img_plane_list[0].shape
    img_plane_sum = np.zeros(img_plane_shape)
    simple_kernel = np.ones(kernel_size)
    i = 0
    while i < plane_num:
        img_plane = img_plane_list[i]
        img_plane = convolve2d(img_plane, simple_kernel, mode='same')
        #plt.hist(img_plane.flatten())
        #plt.show()
        hist, bin_edges = np.histogram(img_plane.flatten())
        hist_last_idx = hist.size - 1
        if np.argmax(hist) != 0:#hist_last_idx:
            kernel_threshold = 10
            img_plane = np.where(img_plane == 10, 1, 0)
        else:
            kernel_threshold = 2#bin_edges[hist_last_idx - 1]
            img_plane = np.where(img_plane >= kernel_threshold, 1, 0)
        img_plane_sum = img_plane_sum + img_plane
        i += 1
    img_plane_sum = np.where(img_plane_sum > 0, 1, 0)
    img_plane_sum_neg = np.where(img_plane_sum > 0, 0, 1)
    return img_plane_sum, img_plane_sum_neg


def extract_img_gray_plane_convolved_sum(img_gray, seg_num, is_segment_mod, kernel_size=(3, 3), kernel_threshold=7):
    multi_threshold_integer_float, multi_threshold_integer_int, threshold_list_with_zero, threshold_list_with_zero_int, plane_section_num, effective_threshold_last = img_integer_segmentation_equal_range_thresholds_light(img_gray, seg_num, is_segment_mod)
    segmentation_img_plane_list = img_segmentation_threshold_list_light(img_gray, multi_threshold_integer_int / INT_GRAY_LEVEL_BAR)
    img_plane_sum, img_plane_sum_neg = img_planes_density_convolve_sum(segmentation_img_plane_list, kernel_threshold=kernel_threshold, kernel_size=kernel_size)
    return img_plane_sum, img_plane_sum_neg


def display_img_plane_histogram(img_plane_list, kernel_size=(3, 3), seg_num=32, plane_black_white_coef=3.0, is_diplay_hist=True):
    plane_num = len(img_plane_list)
    img_plane_shape = img_plane_list[0].shape
    #img_plane_sum = np.zeros(img_plane_shape)
    simple_kernel = np.ones(kernel_size)
    img_plane_sum = np.zeros((img_plane_shape[0] - 2, img_plane_shape[1] - 2))
    img_plane_sum_w = np.zeros((img_plane_shape[0] - 2, img_plane_shape[1] - 2))
    i = 0
    black_white_split = plane_black_white_coef/seg_num
    plt_num = plane_num * 2
    while i < plane_num:
        img_plane = img_plane_list[i]
        img_plane_conv = convolve2d(img_plane, simple_kernel, mode='valid')
        #img_plane_conv_th = np.where(img_plane_conv <= 9, img_plane_conv, 0)
        #img_plane_conv_th = np.where(img_plane_conv_th >= 1, 1, 0)

        #img_plane_conv = convolve2d(img_plane_conv_th, simple_kernel, mode='valid')
        #img_plane_conv_th = np.where(img_plane_conv <= 9, img_plane_conv, 0)
        #img_plane_conv_th = np.where(img_plane_conv_th >= 5, 1, 0)
        #img_plane_conv_th = convolve2d(img_plane_conv_th, simple_kernel, mode='full')
        #img_plane_conv_th = np.where(img_plane_conv_th <= 6, img_plane_conv_th, 0)
        #img_plane_conv_th = np.where(img_plane_conv_th >= 3, 1, 0)
        hist, bin_edges = np.histogram(img_plane_conv.flatten())
        none_zero_hist_ratio, none_zero_hist_std_diff = img_gray_conv_hist_none_zero_statistics(hist)
        img_plane_conv_th = img_plane_none_zero_hist_std_diff_select_func(img_plane_conv, none_zero_hist_std_diff)
        if none_zero_hist_ratio < black_white_split:
            img_plane_sum = img_plane_sum + img_plane_conv_th
        else:
            img_plane_sum_w = img_plane_sum_w + img_plane_conv_th
        if is_diplay_hist:
            plt.subplot(2, plane_num, i+1)
            plt.imshow(img_plane_conv_th, cmap='gray', vmin=0, vmax=1)
            plt.subplot(2, plane_num, i+1+plane_num)
            plt.hist(img_plane_conv)
        i += 1
    if is_diplay_hist:
        plt.show()
    img_plane_sum_neg = np.where(img_plane_sum > 0, 0, 1)
    img_plane_sum_w_neg = np.where(img_plane_sum_w > 0, 0, 1)
    plt.subplot(2, 2, 1)
    plt.imshow(img_plane_sum, cmap='gray', vmin=0, vmax=1)
    plt.subplot(2, 2, 2)
    plt.imshow(img_plane_sum_neg, cmap='gray', vmin=0, vmax=1)
    plt.subplot(2, 2, 3)
    plt.imshow(img_plane_sum_w, cmap='gray', vmin=0, vmax=1)
    plt.subplot(2, 2, 4)
    plt.imshow(img_plane_sum_w_neg, cmap='gray', vmin=0, vmax=1)
    plt.show()
    plt.imshow(img_plane_sum*img_plane_sum_w, cmap='gray', vmin=0, vmax=1)
    plt.show()
    return


def display_img_gray_plane_histogram(img_gray, seg_num, is_segment_mod, kernel_size=(3, 3)):
    multi_threshold_integer_float, multi_threshold_integer_int, threshold_list_with_zero, threshold_list_with_zero_int, plane_section_num, effective_threshold_last = img_integer_segmentation_equal_range_thresholds_light(img_gray, seg_num, is_segment_mod)
    segmentation_img_plane_list = img_segmentation_threshold_list_light(img_gray, multi_threshold_integer_int / INT_GRAY_LEVEL_BAR)
    display_img_plane_histogram(segmentation_img_plane_list, kernel_size, seg_num=seg_num, plane_black_white_coef=3.25, is_diplay_hist=False)#plane_black_white_coef=3.25
    return


def img_gray_conv_hist_none_zero_statistics(conv_hist):
    hist_size = conv_hist.size
    none_zero_hist = conv_hist[1:hist_size]
    img_gray_none_zero_sum = np.sum(none_zero_hist)#img_gray_size - conv_hist[0]
    none_zero_hist_mean = np.mean(none_zero_hist)
    none_zero_hist_std = np.std(none_zero_hist)
    none_zero_hist_std_diff = np.abs(none_zero_hist - none_zero_hist_mean)/none_zero_hist_std
    hist_sum = np.sum(conv_hist)
    none_zero_hist_ratio = img_gray_none_zero_sum / hist_sum
    #print("std diff:", none_zero_hist_std_diff)
    #print("hist ratio:", none_zero_hist_ratio)
    return none_zero_hist_ratio, none_zero_hist_std_diff


def filter_none_zero_hist_std_diff(none_zero_hist_std_diff, std_diff_threshold=2.0):
    filtered_diff_index_list = np.argwhere(none_zero_hist_std_diff < std_diff_threshold)
    filtered_diff_index_list = filtered_diff_index_list + 1
    return filtered_diff_index_list


def img_plane_none_zero_hist_std_diff_select_func(img_plane, none_zero_hist_std_diff, std_diff_threshold=2):
    filter_list = filter_none_zero_hist_std_diff(none_zero_hist_std_diff, std_diff_threshold)
    list_size = filter_list.size
    i = 0
    while i < list_size:
        select_value = filter_list[i]
        img_plane = np.where(img_plane == select_value, -1, img_plane)
        i += 1
    img_plane = np.where(img_plane < 0, 1, 0)
    return img_plane


def get_img_gray_plane_list_sum(img_plane_list, kernel_size=(3, 3), seg_num=32, plane_black_white_coef=5.0, conv_mode='valid'):
    plane_num = len(img_plane_list)
    img_plane_shape = img_plane_list[0].shape
    simple_kernel = np.ones(kernel_size)
    img_plane_sum_b = np.zeros((img_plane_shape[0], img_plane_shape[1]))
    img_plane_sum_w = np.zeros((img_plane_shape[0], img_plane_shape[1]))
    i = 0
    black_white_split = plane_black_white_coef / seg_num
    while i < plane_num:
        img_plane = img_plane_list[i]
        img_plane_conv = convolve2d(img_plane, simple_kernel, mode=conv_mode)
        hist, bin_edges = np.histogram(img_plane_conv.flatten())
        none_zero_hist_ratio, none_zero_hist_std_diff = img_gray_conv_hist_none_zero_statistics(hist)
        img_plane_conv_th = img_plane_none_zero_hist_std_diff_select_func(img_plane_conv, none_zero_hist_std_diff)
        if none_zero_hist_ratio < black_white_split:
            img_plane_sum_b = img_plane_sum_b + img_plane_conv_th
        else:
            img_plane_sum_w = img_plane_sum_w + img_plane_conv_th
        i += 1
    img_plane_sum_b = np.where(img_plane_sum_b > 0, 1, 0)
    img_plane_sum_w = np.where(img_plane_sum_w > 1, 1, 0)
    img_plane_sum_b_neg = np.where(img_plane_sum_b > 0, 0, 1)
    img_plane_sum_w_neg = np.where(img_plane_sum_w > 0, 0, 1)
    if np.max(img_plane_sum_w) == 1:
        img_plane_sum_mul = img_plane_sum_b * img_plane_sum_w
    else:
        img_plane_sum_mul = img_plane_sum_b
    return img_plane_sum_b, img_plane_sum_b_neg, img_plane_sum_w, img_plane_sum_w_neg, img_plane_sum_mul


def img_gray_planes_sum(img_gray, seg_num, is_segment_mod, kernel_size=(3, 3), plane_black_white_coef=5.0, conv_mode='valid'):
    multi_threshold_integer_float, multi_threshold_integer_int, threshold_list_with_zero, threshold_list_with_zero_int, plane_section_num, effective_threshold_last = img_integer_segmentation_equal_range_thresholds_light(img_gray, seg_num, is_segment_mod)
    img_plane_list = img_segmentation_threshold_list_light(img_gray, multi_threshold_integer_int / INT_GRAY_LEVEL_BAR)
    img_plane_sum_b, img_plane_sum_b_neg, img_plane_sum_w, img_plane_sum_w_neg, img_plane_sum_mul = get_img_gray_plane_list_sum(img_plane_list, kernel_size=kernel_size, seg_num=seg_num, plane_black_white_coef=plane_black_white_coef, conv_mode=conv_mode)
    return img_plane_sum_b, img_plane_sum_b_neg, img_plane_sum_w, img_plane_sum_w_neg, img_plane_sum_mul


def img_gray_plane_sum_sim(img_gray_1, img_gray_2, seg_num, is_segment_mod, kernel_size=(3, 3), plane_black_white_coef=3.0, conv_mode='valid'):
    img_plane_sum_b_1, img_plane_sum_b_neg_1, img_plane_sum_w_1, img_plane_sum_w_neg_1, img_plane_sum_mul_1 = img_gray_planes_sum(img_gray_1, seg_num=seg_num, is_segment_mod=is_segment_mod, kernel_size=kernel_size, plane_black_white_coef=plane_black_white_coef, conv_mode=conv_mode)
    img_plane_sum_b_2, img_plane_sum_b_neg_2, img_plane_sum_w_2, img_plane_sum_w_neg_2, img_plane_sum_mul_2 = img_gray_planes_sum(img_gray_2, seg_num=seg_num, is_segment_mod=is_segment_mod, kernel_size=kernel_size, plane_black_white_coef=plane_black_white_coef, conv_mode=conv_mode)

    img_plane_sum_b_sub = np.abs(img_plane_sum_mul_1 - img_plane_sum_mul_2)
    plt.subplot(2, 2, 1)
    plt.imshow(img_plane_sum_b_1+img_plane_sum_w_1, cmap='gray', vmin=0, vmax=1)
    plt.subplot(2, 2, 2)
    plt.imshow(img_plane_sum_b_2+img_plane_sum_w_2, cmap='gray', vmin=0, vmax=1)
    plt.subplot(2, 2, 3)
    plt.imshow(np.where(img_plane_sum_w_1+img_plane_sum_w_2+img_plane_sum_b_1+img_plane_sum_b_2 > 0, 1, 0), cmap='gray', vmin=0, vmax=1)
    plt.subplot(2, 2, 4)
    plt.imshow(np.where(img_plane_sum_w_1+img_plane_sum_w_2>0, 1, 0), cmap='gray', vmin=0, vmax=1)
    plt.show()
    m_test = np.sum(img_plane_sum_mul_1*img_plane_sum_mul_2)/np.sum(np.where(img_plane_sum_mul_1+img_plane_sum_mul_2 > 0, 1, 0))
    print(m_test)

    img_plane_sum_b_sub = np.mean(img_plane_sum_b_sub)

    #img_plane_sum_w_sub = np.abs(img_plane_sum_w_1 - img_plane_sum_w_2)
    img_plane_sum_sub = img_plane_sum_b_sub#(np.mean(img_plane_sum_b_sub) + np.mean(img_plane_sum_w_sub))/2
    img_plane_sum_sim_measure = 1-img_plane_sum_sub
    return img_plane_sum_sim_measure


def cal_mos_plane_sum_sim_scores_gray(ref_image_map, dist_image_list, dist_image_name_list, seg_num, is_segment_mod, C=0.1, is_rgb=False, kernel_size=(3, 3), plane_black_white_coef=3.0, conv_mode='valid', dist_image_name_spliter='_'):
    dist_image_num = len(dist_image_list)
    plane_sum_sims = []
    i = 0
    while i < dist_image_num:
        dist_image_name = dist_image_name_list[i]
        ref_key = dist_image_name.split(dist_image_name_spliter)[0]
        ref_key = ref_key.upper()
        ref_image = ref_image_map[ref_key]
        dist_image = dist_image_list[i]
        #img_plane_sum_sim_measure = img_gray_plane_sum_sim(ref_image, dist_image, seg_num=seg_num, is_segment_mod=is_segment_mod, kernel_size=kernel_size, plane_black_white_coef=plane_black_white_coef, conv_mode=conv_mode)

        if is_rgb:
            ref_image_gray = img_rgb_to_gray_array_cal(np.round(ref_image * 255, 0))
            dist_image_gray = img_rgb_to_gray_array_cal(np.round(dist_image * 255, 0))
        else:
            ref_image_gray = ref_image
            dist_image_gray = dist_image

        img_plane_sum_sim_measure, img_plane_comp_metric_list = get_img_gray_planes_sim(ref_image_gray, dist_image_gray, seg_num=seg_num, is_segment_mod=is_segment_mod)
        FSIMc = imq_spt.fsim(np.round(ref_image * 255, 0), np.round(dist_image * 255, 0), is_output_fsimc=True)
        #img_plane_seg_gradient_sim = img_gray_plane_seg_gradient_sim(ref_image, dist_image, seg_num, is_segment_mod, C)
        img_plane_seg_gradient_sim = img_gray_plane_seg_gradient_multiple_scale_sim(ref_image, dist_image, is_segment_mod)
        #img_plane_sum_sim_measure = img_plane_sum_sim_measure*0 + FSIMc*0 + img_plane_seg_gradient_sim*1
        img_plane_sum_sim_measure = np.power(img_plane_seg_gradient_sim, 0.3) * np.power(FSIMc, 0.7)

        plane_sum_sims.append(img_plane_sum_sim_measure)
        i += 1
        print("image completed:", i, dist_image_name, ref_key, "plane_sum_sim:", img_plane_sum_sim_measure)
    plane_sum_sims = np.array(plane_sum_sims)
    return plane_sum_sims


def img_plane_compare(img_plane_1, img_plane_2):
    img_plane_and = img_plane_1 * img_plane_2
    img_plane_or = img_plane_1 + img_plane_2
    img_plane_or = np.where(img_plane_or > 0, 1, 0)
    img_plane_and_sum = np.sum(img_plane_and)
    img_plane_or_sum = np.sum(img_plane_or)
    img_plane_sum_ratio = img_plane_or_sum/img_plane_1.size
    if img_plane_sum_ratio >= 0.3:
        img_plane_comp_metric = img_plane_and_sum / img_plane_or_sum
    else:
        img_plane_comp_metric = -1
    return img_plane_comp_metric


def img_gray_planes_sim(img_plane_list_1, img_plane_list_2):
    img_plane_num = len(img_plane_list_1)
    img_plane_comp_metric_list = []
    i = 0
    while i < img_plane_num:
        img_plane_1 = img_plane_list_1[i]
        img_plane_2 = img_plane_list_2[i]
        img_plane_comp_metric = img_plane_compare(img_plane_1, img_plane_2)
        if img_plane_comp_metric >= 0:
            img_plane_comp_metric_list.append(img_plane_comp_metric)
        i += 1
    img_plane_comp_metric_list = np.array(img_plane_comp_metric_list)
    img_gray_planes_sim_metric = np.mean(img_plane_comp_metric_list)
    return img_gray_planes_sim_metric, img_plane_comp_metric_list


def img_gray_plane_list_extraction(img_gray, seg_num, is_segment_mod):
    multi_threshold_integer_float, multi_threshold_integer_int, threshold_list_with_zero, threshold_list_with_zero_int, plane_section_num, effective_threshold_last = img_integer_segmentation_equal_range_thresholds_light(img_gray, seg_num, is_segment_mod)
    img_plane_list = img_segmentation_threshold_list_light(img_gray, multi_threshold_integer_int / INT_GRAY_LEVEL_BAR)
    return img_plane_list, multi_threshold_integer_int


def img_gray_plane_list_extraction_rec(img_gray, seg_num, is_segment_mod, is_rec_aligned=True):
    img_plane_list, multi_threshold_integer_int = img_gray_plane_list_extraction(img_gray, seg_num, is_segment_mod)
    aligned_reconstruction_thresholds = range(seg_num)
    aligned_reconstruction_thresholds = np.array(aligned_reconstruction_thresholds) + 1
    aligned_reconstruction_thresholds = aligned_reconstruction_thresholds[::-1]
    if is_rec_aligned:
        img_gray_rec = img_segmentation_reconstruction_threshold(img_plane_list, aligned_reconstruction_thresholds/INT_GRAY_LEVEL_BAR)
    else:
        #img_gray_rec = img_segmentation_reconstruction_threshold(img_plane_list, multi_threshold_integer_int / INT_GRAY_LEVEL_BAR)
        img_gray_rec = img_gray_planes_reconstruction_avg_mask(img_gray, img_plane_list)
    img_gray_gradient_map = imq_spt.gradient_map(np.round(img_gray_rec*INT_GRAY_LEVEL_BAR, 0))
    return img_plane_list, img_gray_rec, img_gray_gradient_map


def get_img_gray_planes_sim(img_gray_1, img_gray_2, seg_num, is_segment_mod):
    img_gray_1 = imq_spt.gradient_map(np.round(img_gray_1 * 255, 0)) / 255
    img_gray_2 = imq_spt.gradient_map(np.round(img_gray_2 * 255, 0)) / 255
    img_plane_list_1, multi_threshold_integer_int_1 = img_gray_plane_list_extraction(img_gray_1, seg_num, is_segment_mod)
    img_plane_list_2, multi_threshold_integer_int_2 = img_gray_plane_list_extraction(img_gray_2, seg_num, is_segment_mod)
    img_gray_planes_sim_metric, img_plane_comp_metric_list = img_gray_planes_sim(img_plane_list_1, img_plane_list_2)
    return img_gray_planes_sim_metric, img_plane_comp_metric_list


def img_gray_gradient_map_planes_sum(img_gray, seg_num, is_segment_mod=True, resize_ratio=0.5, white_ratio=0.2):
    img_gray = img_gray_resize(np.round(img_gray * INT_GRAY_LEVEL_BAR, 0), resize_ratio) / INT_GRAY_LEVEL_BAR
    #img_gray = imq_spt.gradient_map(np.round(img_gray * INT_GRAY_LEVEL_BAR, 0)) / INT_GRAY_LEVEL_BAR
    img_plane_list, multi_threshold_integer_int = img_gray_plane_list_extraction(img_gray, seg_num, is_segment_mod)
    img_plane_num = len(img_plane_list)
    img_gray_shape = img_gray.shape
    img_gray_w = np.zeros(img_gray_shape)
    img_gray_b = np.zeros(img_gray_shape)
    img_gray_size = img_gray.size
    i = 0
    while i < img_plane_num:
        img_plane = img_plane_list[i]
        img_plane_white_ratio = np.sum(img_plane)/img_gray_size
        if img_plane_white_ratio >= white_ratio:
            img_gray_w = img_gray_w + img_plane
        else:
            img_gray_b = img_gray_b + img_plane
        i += 1
    return img_gray_b, img_gray_w


def mos_score_reg_func(x, b1, b2, b3, b4, b5):
    ret = np.exp(b2*(x-b3))
    ret = 1 + ret
    ret = 1/ret
    ret = 1/2 - ret
    ret = b1 * ret
    ret = ret + b4 * x
    ret = ret + b5
    return ret


def none_linear_regression(func, xdata, ydata, maxfev=100000):
    popt = curve_fit(func, xdata, ydata, maxfev=maxfev)
    return popt[0]


def mos_sim_score_none_linear_regression_score(mos_score, mos_sim_score):
    popt = none_linear_regression(mos_score_reg_func, mos_sim_score, mos_score)
    mos_sim_regression_score = mos_score_reg_func(mos_sim_score, popt[0], popt[1], popt[2], popt[3], popt[4])
    return mos_sim_regression_score


def hvs_sim_img_plane_correlation_cal(img_plane_1, img_plane_2, C):
    img_plane_cross_correlation = 2*img_plane_1*img_plane_2 + C
    img_plane_covariance = np.power(img_plane_1, 2) + np.power(img_plane_2, 2) + C
    img_plane_correlation_matrix = img_plane_cross_correlation / img_plane_covariance
    img_plane_correlation = np.mean(img_plane_correlation_matrix)
    return img_plane_correlation_matrix, img_plane_correlation


def img_gray_plane_seg_gradient_sim(img_gray_1, img_gray_2, seg_num, is_segment_mod, C):
    img_plane_list_1, img_gray_rec_1, img_gray_gradient_map_1 = img_gray_plane_list_extraction_rec(img_gray_1, seg_num=seg_num, is_segment_mod=is_segment_mod)
    img_plane_list_2, img_gray_rec_2, img_gray_gradient_map_2 = img_gray_plane_list_extraction_rec(img_gray_2, seg_num=seg_num, is_segment_mod=is_segment_mod)
    img_plane_correlation_matrix, img_plane_correlation = hvs_sim_img_plane_correlation_cal(np.round(img_gray_rec_1 * INT_GRAY_LEVEL_BAR, 0), np.round(img_gray_rec_2 * INT_GRAY_LEVEL_BAR, 0), C)
    img_gradient_correlation_matrix, img_gradient_correlation = hvs_sim_img_plane_correlation_cal(np.round(img_gray_gradient_map_1*255, 0), np.round(img_gray_gradient_map_2*255, 0), C)
    img_plane_seg_gradient_sim = np.power(img_plane_correlation, 1) * np.power(img_gradient_correlation, 0)
    return img_plane_seg_gradient_sim


def img_gray_dynamic_range_sim(img_gray_1, img_gray_2, C=0.001):
    img_gray_1_int = np.round(img_gray_1*INT_GRAY_LEVEL_BAR, 0)
    img_gray_2_int = np.round(img_gray_2*INT_GRAY_LEVEL_BAR, 0)
    img_gray_dynamic_range_1 = np.array([np.max(img_gray_1_int), np.min(img_gray_1_int)])
    img_gray_dynamic_range_2 = np.array([np.max(img_gray_2_int), np.min(img_gray_2_int)])
    img_dynamic_range_correlation_matrix,  img_dynamic_range_correlation = hvs_sim_img_plane_correlation_cal(img_gray_dynamic_range_1, img_gray_dynamic_range_2, C=C)
    return img_dynamic_range_correlation


#img_gray_plane_list_2 is considered as the distorted image
def img_gray_dynamic_range_positive_negative_diff_sim(img_gray_1, img_gray_2, C=0.001):
    img_gray_dynamic_range_sub, img_gray_dynamic_range_sub_sign = img_gray_dynamic_range_sub_cal(img_gray_1, img_gray_2)
    #img_gray_dynamic_range_1 = np.max(img_gray_1) - np.min(img_gray_1)
    #img_gray_dynamic_range_2 = np.max(img_gray_2) - np.min(img_gray_2)
    #img_gray_dynamic_range_sub = img_gray_dynamic_range_1 - img_gray_dynamic_range_2
    #img_gray_dynamic_range_sub_sign = -np.sign(img_gray_dynamic_range_sub)
    #img_gray_dynamic_range_sub = np.abs(img_gray_dynamic_range_sub)
    #if img_gray_dynamic_range_2 <= img_gray_dynamic_range_1:
    if img_gray_dynamic_range_sub_sign <= 0:
        img_dynamic_range_correlation = 1 - rectified_sigmoid(img_gray_dynamic_range_sub, scalar=1.2)
    else:
        img_dynamic_range_correlation = rectified_exp(img_gray_dynamic_range_sub, s_scalar=0.06)#rectified_sigmoid(img_gray_dynamic_range_sub + 2.2, scalar=5.5)
    return img_dynamic_range_correlation, img_gray_dynamic_range_sub_sign


def img_gray_dynamic_range_sub_cal(img_gray_1, img_gray_2):
    img_gray_dynamic_range_1 = np.max(img_gray_1) - np.min(img_gray_1)
    img_gray_dynamic_range_2 = np.max(img_gray_2) - np.min(img_gray_2)
    img_gray_dynamic_range_sub = img_gray_dynamic_range_1 - img_gray_dynamic_range_2
    img_gray_dynamic_range_sub_sign = -np.sign(img_gray_dynamic_range_sub)
    img_gray_dynamic_range_sub = np.abs(img_gray_dynamic_range_sub)
    return img_gray_dynamic_range_sub, img_gray_dynamic_range_sub_sign


def img_gray_plane_seg_gradient_multiple_scale_sim(img_gray_1, img_gray_2, is_segment_mod, C=100, seg_num_list=(8,)):
    scale_len = len(seg_num_list)
    img_plane_seg_gradient_multiple_scale_sim = 1
    i = 0
    while i < scale_len:
        img_plane_seg_gradient_sim = img_gray_plane_seg_gradient_sim(img_gray_1, img_gray_2, seg_num_list[i], is_segment_mod, C)
        img_plane_seg_gradient_multiple_scale_sim *= img_plane_seg_gradient_sim
        i += 1
    return img_plane_seg_gradient_multiple_scale_sim


def img_gray_none_linear_difference_of_gaussian(img_gray, kernel_size_s=(3, 3), kernel_size_l=(15, 15)):
    low_sigma = cv2.GaussianBlur(np.round(img_gray, 0).astype(np.uint8),  kernel_size_s, 0)
    high_sigma = cv2.GaussianBlur(np.round(img_gray, 0).astype(np.uint8), kernel_size_l, 0)
    dog = high_sigma - low_sigma
    dog_mean = np.mean(dog)
    dog = np.where(dog >= dog_mean, 1, 0)
    return dog


def get_img_gray_plane_w_b_ratio(img_gray_plane):
    plane_w_size = np.sum(img_gray_plane)
    plane_total = img_gray_plane.size
    plane_w_ratio = plane_w_size / plane_total
    plane_b_ratio = 1 - plane_w_ratio
    return plane_w_ratio, plane_b_ratio


def adjust_img_gray_plane(img_gray_plane, adjust_ratio=0.5):
    plane_w_ratio, plane_b_ratio = get_img_gray_plane_w_b_ratio(img_gray_plane)
    if plane_w_ratio > adjust_ratio:
        img_gray_plane = np.where(img_gray_plane >0, 0, 1)
    return img_gray_plane


def img_gray_plane_sum_binary(img_gray_plane_sum):
    plane_intensity_sum = np.sum(img_gray_plane_sum)
    img_gray_plane_sum_count = np.sum(np.where(img_gray_plane_sum > 0, 1, 0))
    plane_intensity_mean = np.round(plane_intensity_sum/img_gray_plane_sum_count, 0)
    img_gray_plane_sum = np.where(img_gray_plane_sum >= plane_intensity_mean, 1, 0)
    return img_gray_plane_sum


def img_gray_planes_simple_sum(img_gray_plane_list):
    plane_num = len(img_gray_plane_list)
    plane_shape = img_gray_plane_list[0].shape
    img_gray_plane_sum = np.zeros(plane_shape)
    i = 0
    while i < plane_num:
        img_gray_plane = img_gray_plane_list[i]
        img_gray_plane_sum = img_gray_plane_sum + img_gray_plane
        i += 1
    #img_gray_plane_sum_mean = np.round(np.mean(img_gray_plane_sum), 0)
    #img_gray_plane_sum = np.where(img_gray_plane_sum >= img_gray_plane_sum_mean, 1, 0)
    img_gray_plane_sum = img_gray_plane_sum_binary(img_gray_plane_sum)
    return img_gray_plane_sum


def img_gary_planes_black_white_ratio_inverse(img_gray_plane_list, adjust_ratio=0.5):
    plane_num = len(img_gray_plane_list)
    i = 0
    while i < plane_num:
        img_gray_plane = img_gray_plane_list[i]
        plane_w_ratio, plane_b_ratio = get_img_gray_plane_w_b_ratio(img_gray_plane)
        if plane_w_ratio > adjust_ratio:
            img_gray_plane = np.where(img_gray_plane > 0, 0, 1)
            img_gray_plane_list[i] = img_gray_plane
        i += 1
    return img_gray_plane_list


def img_gray_adjusted_planes_extraction(img_gray, seg_num, is_segment_mod, adjust_ratio=0.5):
    img_plane_list, img_gray_rec, img_gray_gradient_map = img_gray_plane_list_extraction_rec(img_gray, seg_num, is_segment_mod)
    img_plane_list = img_gary_planes_black_white_ratio_inverse(img_plane_list, adjust_ratio=adjust_ratio)
    img_gray_plane_sum = img_gray_planes_simple_sum(img_plane_list)
    img_gray_plane_sum = img_gray_none_linear_difference_of_gaussian(np.round(img_gray_plane_sum * 255, 0))
    return img_plane_list, img_gray_plane_sum


def array_k_means_cluster(array, centroid_num=2):
    whitened_array = whiten(array)
    codebook, distortion = kmeans(whitened_array, centroid_num)
    codebook = np.sort(codebook)
    cluster_result, cluster_distortion = vq(whitened_array, codebook)
    return cluster_result, cluster_distortion


def img_gray_planes_w_ratios_extraction(img_gray_plane_list):
    plane_num = len(img_gray_plane_list)
    img_gray_plane_w_ratio_list = []
    i = 0
    while i < plane_num:
        img_gray_plane = img_gray_plane_list[i]
        plane_w_ratio, plane_b_ratio = get_img_gray_plane_w_b_ratio(img_gray_plane)
        img_gray_plane_w_ratio_list.append(plane_w_ratio)
        i += 1
    img_gray_plane_w_ratio_list = np.array(img_gray_plane_w_ratio_list)
    return img_gray_plane_w_ratio_list


def img_gray_planes_w_ratio_cluster(img_gray_plane_list, centroid_num=2):
    img_gray_plane_w_ratio_list = img_gray_planes_w_ratios_extraction(img_gray_plane_list)
    w_ratio_cluster_result, w_ratio_cluster_distortion = array_k_means_cluster(img_gray_plane_w_ratio_list, centroid_num)
    return w_ratio_cluster_result, w_ratio_cluster_distortion


def img_gray_plane_list_copy(img_gray_plane_list):
    img_gray_plane_list_cpy = np.array(img_gray_plane_list)
    return img_gray_plane_list_cpy


def img_gary_planes_clustered_black_white_ratio_inverse(img_gray_plane_list, img_gray_plane_w_ratio_cluster_list):
    img_gray_plane_list_cpy = img_gray_plane_list_copy(img_gray_plane_list)
    plane_num = len(img_gray_plane_list)
    cluster_max = np.max(img_gray_plane_w_ratio_cluster_list)
    i = 0
    while i < plane_num:
        img_gray_plane = img_gray_plane_list[i]
        if img_gray_plane_w_ratio_cluster_list[i] == cluster_max:
            img_gray_plane = np.where(img_gray_plane > 0, 0, 1)
            img_gray_plane_list[i] = img_gray_plane
        i += 1
    return img_gray_plane_list, img_gray_plane_list_cpy


def img_gray_adjusted_planes_extraction_cluster(img_gray, seg_num, is_segment_mod, centroid_num=2, is_rec_aligned=True):
    img_plane_list, img_gray_rec, img_gray_gradient_map = img_gray_plane_list_extraction_rec(img_gray, seg_num, is_segment_mod, is_rec_aligned=is_rec_aligned)
    w_ratio_cluster_result, w_ratio_cluster_distortion = img_gray_planes_w_ratio_cluster(img_plane_list, centroid_num=centroid_num)
    img_plane_list_inverse, img_plane_list = img_gary_planes_clustered_black_white_ratio_inverse(img_plane_list, w_ratio_cluster_result)
    #display_img_gray_plane_list(img_plane_list)
    img_gray_plane_sum = img_gray_planes_simple_sum(img_plane_list_inverse)
    img_gray_plane_sum = img_gray_none_linear_difference_of_gaussian(np.round(img_gray_plane_sum*255, 0))
    return img_plane_list, img_gray_plane_sum, img_gray_rec


def img_binary_xor(img_binary_1, img_binary_2):
    img_binary_xor_ret = img_binary_1 - img_binary_2
    img_binary_xor_ret = np.where(img_binary_xor_ret != 0, 1, 0)
    return img_binary_xor_ret


def img_binary_and(img_binary_1, img_binary_2):
    img_binary_and_ret = img_binary_1 * img_binary_2
    return img_binary_and_ret


def img_binary_or(img_binary_1, img_binary_2):
    img_binary_or_ret = img_binary_1 + img_binary_2
    img_binary_or_ret = np.where(img_binary_or_ret > 0, 1, 0)
    return img_binary_or_ret


def img_iqa_metric_mul_combination(metric_list, weight_list=None):
    metric_num = len(metric_list)
    if weight_list is None:
        weight_list = np.ones(metric_num)
    metric_mul_ret = 1
    i = 0
    while i < metric_num:
        metric_single = metric_list[i]
        metric_weight = weight_list[i]
        metric_single = np.power(metric_single, metric_weight)
        metric_mul_ret = metric_mul_ret * metric_single
        i += 1
    return metric_mul_ret


def img_plane_pc_correlation(img_plane_1, img_plane_2, C=1, nscale=4, norient=8, minWaveLength=12, mult=2, sigmaOnf=0.55):
    M1, m1, ori1, ft1, PC1, EO1, T_1 = pc.phasecong(img_plane_1, nscale=nscale, norient=norient, minWaveLength=minWaveLength, mult=mult, sigmaOnf=sigmaOnf)
    M2, m2, ori2, ft2, PC2, EO2, T_2 = pc.phasecong(img_plane_2, nscale=nscale, norient=norient, minWaveLength=minWaveLength, mult=mult, sigmaOnf=sigmaOnf)
    PC1 = np.sum(np.array(PC1), 0)
    PC2 = np.sum(np.array(PC2), 0)
    img_plane_correlation_matrix, img_plane_correlation_pc = hvs_sim_img_plane_correlation_cal(PC1, PC2, C=C)
    img_plane_correlation_pc = zero_proc_metric_correlation_cal(img_plane_correlation_matrix, PC1, PC2)
    #PCm = np.maximum(PC1, PC2)
    #img_plane_correlation_pc = np.sum(img_plane_correlation_matrix*PCm)/np.sum(PCm)
    return img_plane_correlation_matrix, img_plane_correlation_pc


def img_plane_hist_diff2_correlation(img_plane_1, img_plane_2, C=0.001):
    hist_1, bin_edges_1 = np.histogram(np.round(img_plane_1 * INT_GRAY_LEVEL_BAR, 0), bins=range(256), density=True)
    hist_2, bin_edges_2 = np.histogram(np.round(img_plane_2 * INT_GRAY_LEVEL_BAR, 0), bins=range(256), density=True)
    hist_1_diff2 = np.diff(np.diff(hist_1))
    hist_2_diff2 = np.diff(np.diff(hist_2))
    img_plane_correlation_matrix_1, img_plane_correlation_hist = hvs_sim_img_plane_correlation_cal(hist_1_diff2, hist_2_diff2, C=C)
    return img_plane_correlation_matrix_1, img_plane_correlation_hist


def img_gray_plane_gradient_correlation(img_plane_1, img_plane_2, C=0.001, is_debug=False):
    img_gray_gradient_1 = imq_spt.gradient_map(np.round(img_plane_1 * 255, 0))
    img_gray_gradient_2 = imq_spt.gradient_map(np.round(img_plane_2 * 255, 0))
    if is_debug:
        plt.subplot(1, 2, 1)
        plt.imshow(img_gray_gradient_1, cmap='gray')
        plt.subplot(1, 2, 2)
        plt.imshow(img_gray_gradient_2, cmap='gray')
        plt.show()
    img_plane_correlation_gradient_matrix, img_plane_correlation_gradient = hvs_sim_img_plane_correlation_cal(img_gray_gradient_1, img_gray_gradient_2, C=C)
    img_plane_correlation_gradient = zero_proc_metric_correlation_cal(img_plane_correlation_gradient_matrix, img_gray_gradient_1, img_gray_gradient_2)
    #img_gray_gradient_zero = np.power(img_gray_gradient_1, 2) + np.power(img_gray_gradient_2, 2)
    #img_gray_gradient_zero = np.where(img_gray_gradient_zero > 0, 1, 0)
    #img_plane_correlation_gradient_matrix_zero = img_plane_correlation_gradient_matrix * img_gray_gradient_zero
    #img_gray_gradient_zero = np.where(img_gray_gradient_zero == 1, 0, 1)
    #img_gray_gradient_zero_sum = np.sum(img_gray_gradient_zero)
    #img_plane_correlation_gradient = np.sum(img_plane_correlation_gradient_matrix_zero)/(img_plane_correlation_gradient_matrix_zero.size - img_gray_gradient_zero_sum)
    return img_plane_correlation_gradient_matrix, img_plane_correlation_gradient


def zero_proc_metric_correlation_cal(correlation_matrix, metric_matrix_1, metric_matrix_2):
    metric_correlation_zero = np.power(metric_matrix_1, 2) + np.power(metric_matrix_2, 2)
    metric_correlation_zero = np.where(metric_correlation_zero > 0, 1, 0)
    correlation_matrix_zero = correlation_matrix * metric_correlation_zero
    metric_correlation_zero = np.where(metric_correlation_zero == 1, 0, 1)
    metric_correlation_zero_sum = np.sum(metric_correlation_zero)
    metric_correlation = np.sum(correlation_matrix_zero) / (correlation_matrix_zero.size - metric_correlation_zero_sum)
    return metric_correlation


def rectified_sigmoid(x, scalar=1.0):
    sig_ret = np.exp(-scalar * x)
    sig_ret = 1 + sig_ret
    sig_ret = 1 / sig_ret
    sig_ret = 2 * sig_ret
    sig_ret = sig_ret - 1
    return sig_ret


def zero_magnified_sigmoid(x, s_scalar=0.5, m_scalar=0.02):
    sig_ret = np.exp(-s_scalar * x)
    sig_ret = 1 + m_scalar * sig_ret
    sig_ret = 1 / sig_ret
    return sig_ret


def biased_sigmoid(x, s_scalar=0.02, bias=1):
    sig_ret = np.exp(-(s_scalar * x + bias))
    sig_ret = 1 + sig_ret
    sig_ret = 1 / sig_ret
    return sig_ret


def rectified_exp(x, s_scalar=1.2, m_scalar=0.99):
    exp_ret = np.exp(s_scalar*x)
    exp_ret = m_scalar * exp_ret
    if exp_ret > 1:
        exp_ret = 1
    return exp_ret


def img_gray_plane_mean_diff(img_plane_1, img_plane_2, mean_scalar=1.5):
    mean_1 = np.std(img_plane_1)
    mean_2 = np.std(img_plane_2)
    mean_sub = np.abs(mean_1 - mean_2)
    img_plane_correlation_mean = 1 - rectified_sigmoid(mean_sub, scalar=mean_scalar)
    return img_plane_correlation_mean


def img_gray_plane_mean_positive_negative_diff_diff(img_plane_1, img_plane_2, mean_scalar=3):
    mean_1 = np.mean(img_plane_1)
    mean_2 = np.mean(img_plane_2)
    #mean_sub = np.abs(mean_1 - mean_2)
    mean_sub = mean_2 - mean_1
    img_plane_mean_correlation = biased_sigmoid(mean_sub, s_scalar=35, bias=7)
    #if mean_2 <= mean_1:
        #img_plane_mean_correlation = 1 - rectified_sigmoid(mean_sub, scalar=mean_scalar)
    #else:
        #img_plane_mean_correlation = rectified_exp(mean_sub, s_scalar=2)#rectified_sigmoid(mean_sub + 1, scalar=10)

    return img_plane_mean_correlation


#img_gray_plane_list_2 is considered as the distorted image
def img_gray_plane_signal_level_correlation(img_gray_plane_list_1, img_gray_plane_list_2, kernel_size=(3, 3), conv_mode='same', denoise_threshold=9, C=0.01):
    plane_denoise_sum_1, img_signal_ratio_1, plane_noise_sum_1, img_noise_ratio_1, img_snr_1 = img_gray_planes_denoise_sum(img_gray_plane_list_1, kernel_size=kernel_size, conv_mode=conv_mode, denoise_threshold=denoise_threshold)
    plane_denoise_sum_2, img_signal_ratio_2, plane_noise_sum_2, img_noise_ratio_2, img_snr_2 = img_gray_planes_denoise_sum(img_gray_plane_list_2, kernel_size=kernel_size, conv_mode=conv_mode, denoise_threshold=denoise_threshold)
    #plane_signal_level_correlation_matrix, plane_signal_level_correlation = hvs_sim_img_plane_correlation_cal(img_signal_ratio_1, img_signal_ratio_2, C=C)
    #plane_signal_level_correlation_matrix, plane_signal_level_correlation = hvs_sim_img_plane_correlation_cal(img_snr_1, img_snr_2, C=C)
    img_snr_sub = np.abs(img_snr_1 - img_snr_2)
    if img_snr_2 <= img_snr_1:
        plane_signal_level_correlation = 1 - rectified_sigmoid(img_snr_sub, scalar=1.5)
    else:
        plane_signal_level_correlation = rectified_sigmoid(img_snr_sub + 1.3, scalar=7.5)
    return plane_signal_level_correlation


def img_plane_cluster_sum_sim(img_gray_1, img_gray_2, seg_num, is_segment_mod, centroid_num, weight_list=(0.8, 0.1, 0, 0.5, 0, 0.5), C_PC=1, C_HIST=0.001, C_SUM=0.001, C_GRAD=0.001, C_DYNA_R=0.001, mean_scalar=10, kernel_size=(3, 3), conv_mode='same', denoise_threshold=9, C_SIG=0.01, is_rec_aligned=True):
    img_plane_list_1, img_gray_plane_sum_1, img_gray_rec_1 = img_gray_adjusted_planes_extraction_cluster(img_gray_1, seg_num=seg_num, is_segment_mod=is_segment_mod, centroid_num=centroid_num, is_rec_aligned=is_rec_aligned)
    img_plane_list_2, img_gray_plane_sum_2, img_gray_rec_2 = img_gray_adjusted_planes_extraction_cluster(img_gray_2, seg_num=seg_num, is_segment_mod=is_segment_mod, centroid_num=centroid_num, is_rec_aligned=is_rec_aligned)

    #test
    #img_plane_list_1_b, img_gray_plane_sum_1_b, img_gray_rec_1_b = img_gray_adjusted_planes_extraction_cluster(img_gray_1, seg_num=128, is_segment_mod=is_segment_mod, centroid_num=centroid_num, is_rec_aligned=is_rec_aligned)
    #img_plane_list_2_b, img_gray_plane_sum_2_b, img_gray_rec_2_b = img_gray_adjusted_planes_extraction_cluster(img_gray_2, seg_num=128, is_segment_mod=is_segment_mod, centroid_num=centroid_num, is_rec_aligned=is_rec_aligned)
    #test

    img_gray_hist_correlation_matrix, img_gray_hist_correlation = img_plane_hist_diff2_correlation(img_gray_1, img_gray_2, C=C_HIST)

    #pc_sum_correlation_matrix, pc_sum_correlation = img_plane_pc_correlation(img_gray_rec_1, img_gray_rec_2, C=C_PC)
    pc_sum_correlation_matrix, pc_sum_correlation = img_plane_pc_correlation(img_gray_1, img_gray_2, C=C_PC)
    #pc_sum_correlation_matrix, pc_sum_correlation = img_plane_pc_correlation(img_gray_plane_sum_1, img_gray_plane_sum_2, C=C_PC)

    img_plane_sum_correlation_matrix, img_plane_sum_correlation = hvs_sim_img_plane_correlation_cal(img_gray_plane_sum_1, img_gray_plane_sum_2, C=C_SUM)

    img_plane_correlation_mean = img_gray_plane_mean_diff(img_gray_1, img_gray_2, mean_scalar=mean_scalar)
    #img_plane_correlation_mean = img_gray_plane_mean_positive_negative_diff_diff(img_gray_1, img_gray_2, mean_scalar=mean_scalar)

    #img_plane_correlation_gradient_matrix, img_plane_correlation_gradient = hvs_sim_img_plane_correlation_cal(np.round(img_gray_rec_1*255, 0), np.round(img_gray_rec_2*255, 0), C=C_GRAD)
    img_plane_correlation_gradient_matrix, img_plane_correlation_gradient = img_gray_plane_gradient_correlation(img_gray_1, img_gray_2, C=C_GRAD)
    #img_plane_correlation_gradient_matrix, img_plane_correlation_gradient = img_gray_plane_gradient_correlation(img_gray_rec_1_b, img_gray_rec_2_b, C=C_GRAD)

    plane_signal_level_correlation = img_gray_plane_signal_level_correlation(img_plane_list_1, img_plane_list_2, kernel_size=kernel_size, conv_mode=conv_mode, denoise_threshold=denoise_threshold, C=C_SIG)
    #img_dynamic_range_correlation = img_gray_dynamic_range_sim(img_gray_1, img_gray_2, C=C_DYNA_R)

    img_dynamic_range_correlation, img_gray_dynamic_range_sub_sign = img_gray_dynamic_range_positive_negative_diff_sim(img_gray_1, img_gray_2, C=C_DYNA_R)
    img_gray_planes_binary_shape_sim_val = img_gray_planes_binary_shape_sim(img_plane_list_1, img_plane_list_2)
    metric_list = [pc_sum_correlation, img_gray_hist_correlation, img_plane_sum_correlation, img_plane_correlation_mean, img_plane_correlation_gradient, plane_signal_level_correlation, img_dynamic_range_correlation, img_gray_planes_binary_shape_sim_val]
    #
    print(metric_list)
    #
    img_plane_cluster_sum_sim_metric = img_iqa_metric_mul_combination(metric_list, weight_list=weight_list)
    return img_plane_cluster_sum_sim_metric


def cal_mos_plane_cluster_sum_sim_scores_gray(ref_image_map, dist_image_list, dist_image_name_list, seg_num, is_segment_mod, centroid_num, weight_list=(1, 0, 0, 0.1, 0.5, 0.2, 0.2, 0.1, 0.1, 0.2, 0.2), C_PC=1, C_HIST=0.001, C_SUM=0.001, mean_scalar=10, C_GRAD=1, kernel_size=(3, 3), conv_mode='same', denoise_threshold=9, C_SIG=0.01, C_DYNA_R=0.001, is_rgb=False, dist_image_name_spliter='_', is_test=False, is_rec_aligned=True):
    dist_image_num = len(dist_image_list)
    plane_sum_sims = []
    i = 0
    while i < dist_image_num:
        dist_image_name = dist_image_name_list[i]
        ref_key = dist_image_name.split(dist_image_name_spliter)[0]
        ref_key = ref_key.upper()
        ref_image = ref_image_map[ref_key]
        dist_image = dist_image_list[i]

        #if is_rgb:
            #ref_image_gray = img_rgb_to_gray_array_cal(np.round(ref_image * 255, 0))
            #dist_image_gray = img_rgb_to_gray_array_cal(np.round(dist_image * 255, 0))
        #else:
            #ref_image_gray = ref_image
            #dist_image_gray = dist_image

        img_plane_cluster_sum_sim_measure = img_plane_sim(ref_image, dist_image, seg_num=seg_num, is_segment_mod=is_segment_mod, centroid_num=centroid_num, weight_list=weight_list, C_PC=C_PC, C_HIST=C_HIST, C_SUM=C_SUM, C_GRAD=C_GRAD, C_DYNA_R=C_DYNA_R, mean_scalar=mean_scalar, kernel_size=kernel_size, conv_mode=conv_mode, denoise_threshold=denoise_threshold, C_SIG=C_SIG, is_rec_aligned=is_rec_aligned)
        #img_plane_cluster_sum_sim_measure = img_plane_cluster_sum_sim(ref_image_gray, dist_image_gray, seg_num=seg_num, is_segment_mod=is_segment_mod, centroid_num=centroid_num, weight_list=weight_list, C_PC=C_PC, C_HIST=C_HIST, C_SUM=C_SUM, C_GRAD=C_GRAD, C_DYNA_R=C_DYNA_R, mean_scalar=mean_scalar, kernel_size=kernel_size, conv_mode=conv_mode, denoise_threshold=denoise_threshold, C_SIG=C_SIG, is_rec_aligned=is_rec_aligned)
        #test
        if is_test:
            FSIMc = imq_spt.fsim(np.round(ref_image * 255, 0), np.round(dist_image * 255, 0), is_output_fsimc=True)
            img_plane_cluster_sum_sim_measure = FSIMc
        #test
        plane_sum_sims.append(img_plane_cluster_sum_sim_measure)
        i += 1
        print("image completed:", i, dist_image_name, ref_key, "plane_sum_sim:", img_plane_cluster_sum_sim_measure)
    plane_sum_sims = np.array(plane_sum_sims)
    return plane_sum_sims


def mos_dist_images_extraction(mos_file_path, dist_file_path, is_rgb=True, img_map_index=IMG_MAP_GREEN_IDX, resize_ratio=0.5):
    mos_scores, dist_image_name_list = retrieve_mos_info(mos_file_path)
    dist_file_paths = generate_mos_file_paths(dist_image_name_list, dist_file_path)
    dist_images = retrieve_mos_raw_images(dist_file_paths, is_rgb=is_rgb, img_map_idx=img_map_index)
    dist_images = resize_mos_images(dist_images, resize_ratio=resize_ratio)
    return dist_images, dist_image_name_list, mos_scores


def mos_ref_images_extraction(ref_file_path, is_rgb=True, img_map_index=IMG_MAP_GREEN_IDX, resize_ratio=0.5):
    ref_file_path_list, ref_file_name_list = retrieve_file_paths(ref_file_path)
    ref_images = retrieve_mos_raw_images(ref_file_path_list, is_rgb=is_rgb, img_map_idx=img_map_index)
    ref_images = resize_mos_images(ref_images, resize_ratio=resize_ratio)
    ref_images = generate_mos_ref_file_map(ref_images, ref_file_name_list)
    return ref_images


def img_gray_planes_denoise_sum(img_gray_plane_list, kernel_size=(3, 3), conv_mode='same', denoise_threshold=9, noise_threshold=1):
    plane_num = len(img_gray_plane_list)
    plane_shape = img_gray_plane_list[0].shape
    cov_kernel = np.ones(kernel_size)
    plane_denoise_sum = np.zeros(plane_shape)
    plane_noise_sum = np.zeros(plane_shape)
    i = 0
    while i < plane_num:
        img_gray_plane = img_gray_plane_list[i]
        #img_gray_plane_conv = convolve2d(img_gray_plane, cov_kernel, mode=conv_mode)
        #denoise_img_gray_plane = np.where(img_gray_plane_conv >= denoise_threshold, 1, 0)
        #noise_img_gray_plane = np.where(img_gray_plane_conv <= noise_threshold, 1, 0)
        #noise_img_gray_plane = noise_img_gray_plane * img_gray_plane_conv
        #noise_img_gray_plane = np.where(noise_img_gray_plane > 0, 1, 0)
        denoise_img_gray_plane, noise_img_gray_plane = img_binary_plane_noise_proc(img_gray_plane, conv_kernel=cov_kernel, signal_threshold=denoise_threshold, noise_threshold=noise_threshold, kernel_size=kernel_size, conv_mode='same')
        plane_denoise_sum = plane_denoise_sum + denoise_img_gray_plane
        plane_noise_sum = plane_noise_sum + noise_img_gray_plane
        i += 1
    plane_denoise_sum = np.where(plane_denoise_sum > 0, 1, 0)
    plane_noise_sum = np.where(plane_noise_sum > 0, 1, 0)
    img_signal_ratio = np.mean(plane_denoise_sum)
    img_noise_ratio = np.mean(plane_noise_sum)
    img_snr = img_signal_ratio / (img_noise_ratio + img_signal_ratio) #img_signal_ratio / img_noise_ratio
    return plane_denoise_sum, img_signal_ratio, plane_noise_sum, img_noise_ratio, img_snr


def img_gray_planes_denoise_adjust(img_gray_plane_list, signal_threshold=9, noise_threshold=2, kernel_size=(3, 3), conv_mode='same'):
    plane_num = len(img_gray_plane_list)
    cov_kernel = np.ones(kernel_size)
    denoise_plane_list = []
    noise_plane_list = []
    i = 0
    while i < plane_num:
        img_gray_plane = img_gray_plane_list[i]
        denoise_img_binary_plane, noise_img_binary_plane = img_binary_plane_noise_proc(img_gray_plane, conv_kernel=cov_kernel, signal_threshold=signal_threshold, noise_threshold=noise_threshold, kernel_size=kernel_size, conv_mode=conv_mode)
        denoise_plane_list.append(denoise_img_binary_plane)
        noise_plane_list.append(noise_img_binary_plane)
        i += 1
    return denoise_plane_list, noise_plane_list


def img_binary_plane_noise_proc(img_binary_plane, conv_kernel=None, signal_threshold=9, noise_threshold=2, kernel_size=(3, 3), conv_mode='same'):
    if conv_kernel is None:
        conv_kernel = np.ones(kernel_size)
    img_binary_plane_conv = convolve2d(img_binary_plane, conv_kernel, mode=conv_mode)
    denoise_img_binary_plane = np.where(img_binary_plane_conv >= signal_threshold, 1, 0)
    noise_img_binary_plane = np.where(img_binary_plane_conv <= noise_threshold, 1, 0)
    noise_img_binary_plane = noise_img_binary_plane * img_binary_plane_conv
    noise_img_binary_plane = np.where(noise_img_binary_plane > 0, 1, 0)
    denoise_img_binary_plane = np.where(denoise_img_binary_plane > 0, 1, 0)
    return denoise_img_binary_plane, noise_img_binary_plane


def img_binary_sub(img_binary_1, img_binary_2, sub_left_to_right=True):
    if sub_left_to_right:
        img_binary_sub_ret = img_binary_1 - img_binary_2
    else:
        img_binary_sub_ret = img_binary_2 - img_binary_1
    img_binary_sub_ret = np.where(img_binary_sub_ret > 0, 1, 0)
    return img_binary_sub_ret


def img_binary_sub_lef_to_right(img_binary_1, img_binary_2):
    img_binary_sub_ret = img_binary_1 - img_binary_2
    img_binary_sub_ret = ndarray_threshold_delta_func(img_binary_sub_ret)
    return img_binary_sub_ret


def img_binary_sub_right_to_left(img_binary_1, img_binary_2):
    img_binary_sub_ret = img_binary_2 - img_binary_1
    img_binary_sub_ret = ndarray_threshold_delta_func(img_binary_sub_ret)
    return img_binary_sub_ret


def ndarray_threshold_delta_func(x, threshold=0):
    delta_ret = np.where(x > threshold, 1, 0)
    return delta_ret


def img_planes_xor(img_plane_list_1, img_plane_list_2):
    plane_num = len(img_plane_list_1)
    plane_shape = img_plane_list_1[0].shape
    img_planes_xor_or = np.zeros(plane_shape)
    img_plane_xor_list = []
    i = 0
    while i < plane_num:
        img_plane_1 = img_plane_list_1[i]
        img_plane_2 = img_plane_list_2[i]
        img_plane_xor = img_binary_xor(img_plane_1, img_plane_2)
        img_plane_xor_list.append(img_plane_xor)
        img_planes_xor_or = img_binary_or(img_planes_xor_or, img_plane_xor)
        i += 1
    return img_plane_xor_list, img_planes_xor_or


def img_planes_sub(img_plane_list_1, img_plane_list_2, sub_left_to_right=True):
    plane_num = len(img_plane_list_1)
    plane_shape = img_plane_list_1[0].shape
    img_planes_sub_or = np.zeros(plane_shape)
    img_plane_sub_list = []
    i = 0
    while i < plane_num:
        img_plane_1 = img_plane_list_1[i]
        img_plane_2 = img_plane_list_2[i]
        img_plane_sub = img_binary_sub(img_plane_1, img_plane_2, sub_left_to_right=sub_left_to_right)
        #
        img_plane_sub = img_binary_plane_noise_proc(img_plane_sub)[0]
        #
        img_plane_sub_list.append(img_plane_sub)
        img_planes_sub_or = img_binary_or(img_planes_sub_or, img_plane_sub)
        i += 1
    return img_plane_sub_list, img_planes_sub_or


def img_planes_and(img_plane_list_1, img_plane_list_2):
    plane_num = len(img_plane_list_1)
    plane_shape = img_plane_list_1[0].shape
    img_planes_sub_or = np.zeros(plane_shape)
    img_plane_sub_list = []
    i = 0
    while i < plane_num:
        img_plane_1 = img_plane_list_1[i]
        img_plane_2 = img_plane_list_2[i]
        img_plane_sub = img_binary_and(img_plane_1, img_plane_2)
        img_plane_sub_list.append(img_plane_sub)
        img_planes_sub_or = img_binary_or(img_planes_sub_or, img_plane_sub)
        i += 1
    return img_plane_sub_list, img_planes_sub_or


def img_planes_binary_operate(img_plane_list_1, img_plane_list_2, binary_operator, is_denoise=True):
    plane_num = len(img_plane_list_1)
    plane_shape = img_plane_list_1[0].shape
    img_plane_binary_ret_or = np.zeros(plane_shape)
    img_plane_sub_list = []
    i = 0
    while i < plane_num:
        img_plane_1 = img_plane_list_1[i]
        img_plane_2 = img_plane_list_2[i]
        img_plane_binary_ret = binary_operator(img_plane_1, img_plane_2)
        if is_denoise:
            img_plane_binary_ret = img_binary_plane_noise_proc(img_plane_binary_ret)[0]
        img_plane_sub_list.append(img_plane_binary_ret)
        img_plane_binary_ret_or = img_binary_or(img_plane_binary_ret_or, img_plane_binary_ret)
        i += 1
    return img_plane_sub_list, img_plane_binary_ret_or


def img_gray_diff_density_filter(img_gray_diff, diff_mean_threshold_num=10):
    img_gray_diff_sum = np.sum(img_gray_diff)
    img_gray_diff_mean = img_gray_diff_sum / img_gray_diff.size
    diff_mean_threshold = diff_mean_threshold_num / img_gray_diff.size
    if img_gray_diff_mean >= diff_mean_threshold:
        img_gray_diff_ret = img_gray_diff_sum
    else:
        img_gray_diff_ret = 0
    return img_gray_diff_ret


def threshold_delta(x, delta_threshold):
    if x >= delta_threshold:
        delta_ret = 1
    else:
        delta_ret = 0
    return delta_ret


def img_planes_sub_density_filter_ratio(img_planes_sub_or_1, img_planes_sub_or_2, filter_threshold_num=50, filter_threshold_ratio=0.1):
    sub_or_mean_1 = np.mean(img_planes_sub_or_1)
    sub_or_mean_2 = np.mean(img_planes_sub_or_2)
    sub_or_mean_min = np.fmin(sub_or_mean_1, sub_or_mean_2)
    sub_or_mean_max = np.fmax(sub_or_mean_1, sub_or_mean_2)
    sub_or_mean_min_threshold = filter_threshold_num/img_planes_sub_or_1.size
    if sub_or_mean_min < sub_or_mean_min_threshold:
        sub_or_mean_min = 0
    if sub_or_mean_max == 0:
        sub_or_delta = 0
    else:
        sub_or_mean_ratio = sub_or_mean_min / sub_or_mean_max
        sub_or_delta = threshold_delta(sub_or_mean_ratio, filter_threshold_ratio)
    img_planes_sub_or_ret_1 = img_planes_sub_or_1 * sub_or_delta
    img_planes_sub_or_ret_2 = img_planes_sub_or_2 * sub_or_delta
    return img_planes_sub_or_ret_1, img_planes_sub_or_ret_2


def img_planes_sub_density_filter(img_planes_sub_or_1, img_planes_sub_or_2, filter_threshold_num=400):
    sub_or_mean_1 = np.mean(img_planes_sub_or_1)
    sub_or_mean_2 = np.mean(img_planes_sub_or_2)
    #
    print("sub_ratio:", np.fmin(sub_or_mean_1, sub_or_mean_2) / np.fmax(sub_or_mean_1, sub_or_mean_2))
    #
    sub_or_mean_cross = sub_or_mean_1 * sub_or_mean_2
    filter_threshold = np.power(filter_threshold_num/img_planes_sub_or_1.size, 2)
    sub_or_delta = threshold_delta(sub_or_mean_cross, filter_threshold)
    img_planes_sub_or_ret_1 = img_planes_sub_or_1 * sub_or_delta
    img_planes_sub_or_ret_2 = img_planes_sub_or_2 * sub_or_delta
    return img_planes_sub_or_ret_1, img_planes_sub_or_ret_2


def img_gray_planes_binary_shape_sim(img_plane_list_1, img_plane_list_2, sub_xor_sig_ratio=0.5, is_debug=False):
    img_plane_and_sub_list_1, img_planes_and_sub_or_1 = img_planes_binary_operate(img_plane_list_1, img_plane_list_2, img_binary_sub_lef_to_right)
    img_plane_and_sub_list_2, img_planes_and_sub_or_2 = img_planes_binary_operate(img_plane_list_2, img_plane_list_1, img_binary_sub_lef_to_right)

    img_planes_and_sub_or_and = img_binary_and(img_planes_and_sub_or_1, img_planes_and_sub_or_2)
    img_planes_and_sub_or_xor = img_binary_xor(img_planes_and_sub_or_1, img_planes_and_sub_or_2)

    #test
    if is_debug:
        plt.subplot(1, 2, 1)
        plt.imshow(img_planes_and_sub_or_and, cmap='gray')
        plt.subplot(1, 2, 2)
        plt.imshow(img_planes_and_sub_or_xor, cmap='gray')
        plt.show()
    #test

    img_planes_and_sub_or_and, img_planes_and_sub_or_xor = img_planes_sub_density_filter_ratio(img_planes_and_sub_or_and, img_planes_and_sub_or_xor)

    #img_planes_and_sub_or_and = img_binary_plane_noise_proc(img_planes_and_sub_or_and)[0]
    #img_planes_and_sub_or_xor = img_binary_plane_noise_proc(img_planes_and_sub_or_xor)[0]

    #img_planes_and_sub_or_and_sum = np.sum(img_planes_and_sub_or_and)
    #img_planes_and_sub_or_xor_sum = np.sum(img_planes_and_sub_or_xor)
    #mg_planes_and_sub_or_and_mean = img_planes_and_sub_or_and_sum / img_planes_and_sub_or_and.size
    #img_planes_and_sub_or_xor_mean = img_planes_and_sub_or_xor_sum / img_planes_and_sub_or_xor.size

    img_planes_and_sub_or_and_sum = img_gray_diff_density_filter(img_planes_and_sub_or_and)
    img_planes_and_sub_or_xor_sum = img_gray_diff_density_filter(img_planes_and_sub_or_xor)

    img_planes_and_sub_or_ratio_u = np.fmin(img_planes_and_sub_or_and_sum, img_planes_and_sub_or_xor_sum)
    img_planes_and_sub_or_ratio_d = np.fmax(img_planes_and_sub_or_and_sum, img_planes_and_sub_or_xor_sum)

    if img_planes_and_sub_or_ratio_d > 0:
        img_gray_planes_binary_shape_sim_val = img_planes_and_sub_or_ratio_u / img_planes_and_sub_or_ratio_d
    else:
        img_gray_planes_binary_shape_sim_val = 1
    if img_planes_and_sub_or_xor_sum * sub_xor_sig_ratio < img_planes_and_sub_or_and_sum:
        img_gray_planes_binary_shape_sim_val = 1 - img_gray_planes_binary_shape_sim_val

    return img_gray_planes_binary_shape_sim_val


def get_img_plane_rec_intensity_plane(img_gray_org, img_binary_plane):
    pixel_num = np.sum(img_binary_plane)
    if pixel_num == 0:
        img_gray_rec_intensity = 0
    else:
        img_gray_rec_intensity = np.sum(img_gray_org * img_binary_plane)/np.sum(img_binary_plane)
    img_rec_plane = img_binary_plane * img_gray_rec_intensity
    return img_rec_plane


def img_gray_planes_reconstruction_avg_mask(img_gray_org, img_gray_plane_list):
    plane_num = len(img_gray_plane_list)
    img_shape = img_gray_org.shape
    img_gray_rec = np.zeros(img_shape)
    i = 0
    while i < plane_num:
        img_rec_plane = get_img_plane_rec_intensity_plane(img_gray_org, img_gray_plane_list[i])
        img_gray_rec = img_gray_rec + img_rec_plane
        i += 1
    return img_gray_rec


def img_planes_sum(img_plane_list, is_normalize=True):
    img_plane_array = np.array(img_plane_list)
    img_plane_sum = np.sum(np.sum(img_plane_array, axis=2), axis=1)
    if is_normalize:
        img_plane_sum = img_plane_sum / np.sum(img_plane_sum)
    return img_plane_sum


def img_gray_planes_hist_sim(img_plane_list_1, img_plane_list_2, C=0.01):
    img_plane_sum_1 = img_planes_sum(img_plane_list_1)
    img_plane_sum_2 = img_planes_sum(img_plane_list_2)
    img_plane_correlation_matrix, img_plane_correlation = hvs_sim_img_plane_correlation_cal(img_plane_sum_1, img_plane_sum_2, C=C)
    return img_plane_correlation_matrix, img_plane_correlation


def img_gray_planes_hist_diff(img_plane_list_1, img_plane_list_2):
    plane_num = len(img_plane_list_1)
    img_plane_sum_1 = img_planes_sum(img_plane_list_1)
    img_plane_sum_2 = img_planes_sum(img_plane_list_2)
    img_plane_sum_sub = img_plane_sum_1 - img_plane_sum_2
    hist_sub_weight = np.array(range(plane_num)) + 1
    hist_sub_weight = hist_sub_weight[::-1]
    img_plane_sum_sub_diff = np.sum(img_plane_sum_sub*hist_sub_weight)
    return img_plane_sum_sub_diff


def img_i_convert(img):
    img_shape = img.shape
    I = np.zeros(img_shape)
    if img.ndim == 3:
        img_plane_0 = img[:, :, 0]
        img_plane_1 = img[:, :, 1]
        img_plane_2 = img[:, :, 2]
        I = 0.596 * img_plane_0 - 0.274 * img_plane_1 - 0.322 * img_plane_2
    return I


def img_q_convert(img):
    img_shape = img.shape
    Q = np.zeros(img_shape)
    if img.ndim == 3:
        img_plane_0 = img[:, :, 0]
        img_plane_1 = img[:, :, 1]
        img_plane_2 = img[:, :, 2]
        Q = 0.211 * img_plane_0 - 0.523 * img_plane_1 + 0.312 * img_plane_2
    return Q


def img_rgb_planes_hist_diff(img_red_plane_list, img_green_plane_list, img_blue_plane_list, img_gray_plane_list):
    img_plane_sum_sub_diff_red = img_gray_planes_hist_diff(img_red_plane_list, img_gray_plane_list)
    img_plane_sum_sub_diff_green = img_gray_planes_hist_diff(img_green_plane_list, img_gray_plane_list)
    img_plane_sum_sub_diff_blue = img_gray_planes_hist_diff(img_blue_plane_list, img_gray_plane_list)
    return img_plane_sum_sub_diff_red, img_plane_sum_sub_diff_green, img_plane_sum_sub_diff_blue


def img_rgb_overall_chromatic_metric(img_plane_sum_sub_diff_red, img_plane_sum_sub_diff_green, img_plane_sum_sub_diff_blue):
    img_plane_sum_sub_diff_range_array = np.array([img_plane_sum_sub_diff_red, img_plane_sum_sub_diff_green, img_plane_sum_sub_diff_blue])
    overall_chromatic_metric = np.sum(img_plane_sum_sub_diff_range_array)
    return overall_chromatic_metric


def img_rgb_chromatic_contrast_quality_metric(img_plane_sum_sub_diff_red, img_plane_sum_sub_diff_green, img_plane_sum_sub_diff_blue):
    img_plane_sum_sub_diff_range_array = np.array([img_plane_sum_sub_diff_red, img_plane_sum_sub_diff_green, img_plane_sum_sub_diff_blue])
    chromatic_contrast_quality_metric = np.max(img_plane_sum_sub_diff_range_array) - np.min(img_plane_sum_sub_diff_range_array)
    return chromatic_contrast_quality_metric


def img_rgb_overall_chromatic_diff_metric(overall_chromatic_metric_1, overall_chromatic_metric_2, metric_scalar=1.5):
    overall_chromatic_diff_metric = np.abs(overall_chromatic_metric_1 - overall_chromatic_metric_2)
    overall_chromatic_diff_metric = np.exp(-metric_scalar * overall_chromatic_diff_metric)
    return overall_chromatic_diff_metric


# The order of dist and ref should be kept because of subtraction
def img_rgb_color_contrast_quality_diff_metric(color_contrast_quality_metric_dist, color_contrast_quality_metric_ref, s_scalar=20, bias=5, metric_low_threshold=0.7):
    color_contrast_quality_diff_metric = color_contrast_quality_metric_dist - color_contrast_quality_metric_ref
    if color_contrast_quality_diff_metric == 0:
        color_contrast_quality_diff_metric = 1
    else:
        color_contrast_quality_diff_metric = biased_sigmoid(color_contrast_quality_diff_metric, s_scalar=s_scalar, bias=bias)
    if color_contrast_quality_diff_metric <= metric_low_threshold:
        color_contrast_quality_diff_metric = metric_low_threshold
    return color_contrast_quality_diff_metric


def img_rgb_basic_channels(img_rgb, resize_ratio=0.5, interpolation=cv2.INTER_NEAREST):
    img_rgb_red = img_rgb[:, :, IMG_MAP_RED_IDX]
    img_rgb_green = img_rgb[:, :, IMG_MAP_GREEN_IDX]
    img_rgb_blue = img_rgb[:, :, IMG_MAP_BLUE_IDX]
    img_rgb_gray = img_rgb_to_gray_array_cal(np.round(img_rgb * INT_GRAY_LEVEL_BAR, 0))
    #img_rgb_gray_resized = img_gray_resize(np.round(img_rgb_gray*INT_GRAY_LEVEL_BAR, 0), resize_ratio, interpolation=interpolation)/INT_GRAY_LEVEL_BAR
    return img_rgb_gray, img_rgb_red, img_rgb_green, img_rgb_blue


def img_rgb_channels(img_rgb, seg_num, is_segment_mod, is_rec_aligned=True):
    img_rgb_gray, img_rgb_red, img_rgb_green, img_rgb_blue = img_rgb_basic_channels(img_rgb)
    img_plane_list_gray, img_gray_rec_gray, img_gray_gradient_map_gray = img_gray_plane_list_extraction_rec(img_rgb_gray, seg_num=seg_num, is_segment_mod=is_segment_mod, is_rec_aligned=is_rec_aligned)
    img_plane_list_red, img_gray_rec_red, img_gray_gradient_map_red = img_gray_plane_list_extraction_rec(img_rgb_red, seg_num=seg_num, is_segment_mod=is_segment_mod, is_rec_aligned=is_rec_aligned)
    img_plane_list_green, img_gray_rec_green, img_gray_gradient_map_green = img_gray_plane_list_extraction_rec(img_rgb_green, seg_num=seg_num, is_segment_mod=is_segment_mod, is_rec_aligned=is_rec_aligned)
    img_plane_list_blue, img_gray_rec_blue, img_gray_gradient_map_blue = img_gray_plane_list_extraction_rec(img_rgb_blue, seg_num=seg_num, is_segment_mod=is_segment_mod, is_rec_aligned=is_rec_aligned)
    return img_rgb_gray, img_plane_list_gray, img_plane_list_red, img_plane_list_green, img_plane_list_blue


def img_rgb_chromatic_metric(img_rgb_ref, img_rgb_dist, seg_num, is_segment_mod, is_rec_aligned=True):
    img_rgb_gray_dist, img_plane_list_gray_dist, img_plane_list_red_dist, img_plane_list_green_dist, img_plane_list_blue_dist = img_rgb_channels(img_rgb_dist, seg_num, is_segment_mod, is_rec_aligned=is_rec_aligned)
    img_rgb_gray_ref, img_plane_list_gray_ref, img_plane_list_red_ref, img_plane_list_green_ref, img_plane_list_blue_ref = img_rgb_channels(img_rgb_ref, seg_num, is_segment_mod, is_rec_aligned=is_rec_aligned)

    img_plane_sum_sub_diff_red_dist, img_plane_sum_sub_diff_green_dist, img_plane_sum_sub_diff_blue_dist = img_rgb_planes_hist_diff(img_plane_list_red_dist, img_plane_list_green_dist, img_plane_list_blue_dist, img_plane_list_gray_dist)
    img_plane_sum_sub_diff_red_ref, img_plane_sum_sub_diff_green_ref, img_plane_sum_sub_diff_blue_ref = img_rgb_planes_hist_diff(img_plane_list_red_ref, img_plane_list_green_ref, img_plane_list_blue_ref, img_plane_list_gray_ref)

    overall_chromatic_metric_dist = img_rgb_overall_chromatic_metric(img_plane_sum_sub_diff_red_dist, img_plane_sum_sub_diff_green_dist, img_plane_sum_sub_diff_blue_dist)
    overall_chromatic_metric_ref = img_rgb_overall_chromatic_metric(img_plane_sum_sub_diff_red_ref, img_plane_sum_sub_diff_green_ref, img_plane_sum_sub_diff_blue_ref)

    chromatic_contrast_quality_metric_dist = img_rgb_chromatic_contrast_quality_metric(img_plane_sum_sub_diff_red_dist, img_plane_sum_sub_diff_green_dist, img_plane_sum_sub_diff_blue_dist)
    chromatic_contrast_quality_metric_ref = img_rgb_chromatic_contrast_quality_metric(img_plane_sum_sub_diff_red_ref, img_plane_sum_sub_diff_green_ref, img_plane_sum_sub_diff_blue_ref)

    overall_chromatic_diff_metric = img_rgb_overall_chromatic_diff_metric(overall_chromatic_metric_dist, overall_chromatic_metric_ref)
    color_contrast_quality_diff_metric = img_rgb_color_contrast_quality_diff_metric(chromatic_contrast_quality_metric_dist, chromatic_contrast_quality_metric_ref)
    return img_rgb_gray_dist, img_rgb_gray_ref, img_plane_list_gray_dist, img_plane_list_gray_ref, overall_chromatic_diff_metric, color_contrast_quality_diff_metric


def vector_cos_distance(v1, v2):
    v_cos_distance = np.sum(v1 * v2)
    v_cos_distance = v_cos_distance/(np.linalg.norm(v1)*np.linalg.norm(v2))
    return v_cos_distance


def img_plane_sim(img_ref, img_dist, seg_num, is_segment_mod, centroid_num, weight_list=(1, 0, 0, 0.1, 0.5, 0.2, 0.2, 0.1, 0.1, 0.2, 0.2), C_PC=1, C_HIST=0.001, C_SUM=0.001, C_GRAD=0.001, C_DYNA_R=0.001, mean_scalar=10, kernel_size=(3, 3), conv_mode='same', denoise_threshold=9, C_SIG=0.01, is_rec_aligned=True):
    if img_ref.ndim == 3:
        #img_rgb_gray_dist, img_rgb_gray_ref, img_plane_list_gray_dist, img_plane_list_gray_ref, overall_chromatic_diff_metric, color_contrast_quality_diff_metric = img_rgb_chromatic_metric(img_dist, img_ref, seg_num, is_segment_mod=is_segment_mod, is_rec_aligned=is_rec_aligned)
        img_rgb_gray_ref, img_rgb_red_ref, img_rgb_green_ref, img_rgb_blue_ref = img_rgb_basic_channels(img_ref)
        img_rgb_gray_dist, img_rgb_red_dist, img_rgb_green_dist, img_rgb_blue_dist = img_rgb_basic_channels(img_dist)
        v_skewness_diff_distance_score, skewness_diff_max_sub_score = img_rgb_chromatic_mean_skewness_metric(img_rgb_gray_ref, img_rgb_red_ref, img_rgb_green_ref, img_rgb_blue_ref, img_rgb_gray_dist, img_rgb_red_dist, img_rgb_green_dist, img_rgb_blue_dist)
    else:
        #overall_chromatic_diff_metric = 1
        #color_contrast_quality_diff_metric = 1
        v_skewness_diff_distance_score = 1
        skewness_diff_max_sub_score = 1
        img_rgb_gray_dist = img_dist
        img_rgb_gray_ref = img_ref
        #img_plane_list_gray_dist, img_gray_plane_sum_1, img_gray_rec_1 = img_gray_adjusted_planes_extraction_cluster(img_rgb_gray_dist, seg_num=seg_num, is_segment_mod=is_segment_mod, centroid_num=centroid_num, is_rec_aligned=is_rec_aligned)
        #img_plane_list_gray_ref, img_gray_plane_sum_2, img_gray_rec_2 = img_gray_adjusted_planes_extraction_cluster(img_rgb_gray_ref, seg_num=seg_num, is_segment_mod=is_segment_mod, centroid_num=centroid_num, is_rec_aligned=is_rec_aligned)
    pc_sum_correlation, img_gray_hist_correlation, img_plane_sum_correlation, img_plane_correlation_mean, img_plane_correlation_gradient, plane_signal_level_correlation, img_dynamic_range_correlation, img_gray_planes_binary_shape_sim_val, skewness_diff_score = img_gray_sim_metrics(img_rgb_gray_ref, img_rgb_gray_dist, seg_num=seg_num, is_segment_mod=is_segment_mod, centroid_num=centroid_num, C_PC=C_PC, C_HIST=C_HIST, C_SUM=C_SUM, C_GRAD=C_GRAD, C_DYNA_R=C_DYNA_R, mean_scalar=mean_scalar, kernel_size=kernel_size, conv_mode=conv_mode, denoise_threshold=denoise_threshold, C_SIG=C_SIG, is_rec_aligned=is_rec_aligned)
    metric_list = [pc_sum_correlation, img_gray_hist_correlation, img_plane_sum_correlation, img_plane_correlation_mean, img_plane_correlation_gradient, plane_signal_level_correlation, img_dynamic_range_correlation, img_gray_planes_binary_shape_sim_val, v_skewness_diff_distance_score, skewness_diff_max_sub_score, skewness_diff_score]
    print(metric_list)
    img_plane_sim_metric = img_iqa_metric_mul_combination(metric_list, weight_list=weight_list)
    return img_plane_sim_metric


def img_gray_plane_mean_skewness(img_gray_plane):
    plane_min = np.min(img_gray_plane)
    plane_max = np.max(img_gray_plane)
    plane_mean = np.mean(img_gray_plane)
    plane_mean_skewness = (plane_mean - plane_min)/(plane_max - plane_min)
    return plane_mean_skewness


def img_rgb_plane_mean_skewness(img_gray_plane, img_red_plane, img_green_plane, img_blue_plane):
    gray_mean_skewness = img_gray_plane_mean_skewness(img_gray_plane)
    red_mean_skewness = img_gray_plane_mean_skewness(img_red_plane)
    green_mean_skewness = img_gray_plane_mean_skewness(img_green_plane)
    blue_mean_skewness = img_gray_plane_mean_skewness(img_blue_plane)
    v_mean_skewness = np.array([red_mean_skewness, green_mean_skewness, blue_mean_skewness])
    return gray_mean_skewness, v_mean_skewness


def img_rgb_plane_mean_skewness_gray_diff_v(v_mean_skewness, gray_mean_skewness):
    v_skewness_gray_diff = v_mean_skewness - gray_mean_skewness
    return v_skewness_gray_diff


def img_rgb_plane_mean_skewness_gray_diff_max(v_mean_skewness, gray_mean_skewness):
    v_skewness_gray_diff_sub = v_mean_skewness - gray_mean_skewness
    #skewness_gray_diff_max = np.max(np.abs(v_mean_skewness - gray_mean_skewness))
    skewness_gray_diff_max = np.max(v_skewness_gray_diff_sub) - np.min(v_skewness_gray_diff_sub)
    return skewness_gray_diff_max, v_skewness_gray_diff_sub


def img_rgb_plane_mean_skewness_diff_v_dist_ref_distance(v_skewness_gray_diff_ref, v_skewness_gray_diff_dist, s_scalar=0.5):
    v_skewness_diff_sub = v_skewness_gray_diff_ref - v_skewness_gray_diff_dist
    v_skewness_diff_sub = np.linalg.norm(v_skewness_diff_sub)
    v_skewness_diff_distance_score = np.exp(-s_scalar*v_skewness_diff_sub)
    return v_skewness_diff_sub, v_skewness_diff_distance_score


# dist and ref are ordered
def img_rgb_plane_mean_skewness_diff_v_dist_ref_max_diff(skewness_gray_diff_max_ref, skewness_gray_diff_max_dist, s_scalar=40, bias=5):
    skewness_diff_max_sub = skewness_gray_diff_max_dist - skewness_gray_diff_max_ref
    skewness_diff_max_sub_score = biased_sigmoid(skewness_diff_max_sub, s_scalar=s_scalar, bias=bias)
    return skewness_diff_max_sub, skewness_diff_max_sub_score


def img_gray_mean_skewness_diff(img_gray_ref, img_gray_dist, s_scalar=21, bias=5):
    img_gray_mean_skewness_ref = img_gray_plane_mean_skewness(img_gray_ref)
    img_gray_mean_skewness_dist = img_gray_plane_mean_skewness(img_gray_dist)
    skewness_diff, skewness_diff_score = img_rgb_plane_mean_skewness_diff_v_dist_ref_max_diff(img_gray_mean_skewness_ref, img_gray_mean_skewness_dist, s_scalar=s_scalar, bias=bias)
    return skewness_diff, skewness_diff_score


def img_rgb_plane_mean_skewness_quality_diff(v_skewness_gray_diff_sub_ref, v_skewness_gray_diff_sub_dist):
    v_skewness_gray_diff_sub_ref_sign = np.sign(v_skewness_gray_diff_sub_ref)
    v_skewness_gray_diff_sub_dist_sign = np.sign(v_skewness_gray_diff_sub_dist)
    dist_ref_sign_cross = v_skewness_gray_diff_sub_ref_sign * v_skewness_gray_diff_sub_dist_sign
    dist_ref_sign_cross_neg_mask = np.where(dist_ref_sign_cross == -1, 1, 0)
    dist_ref_sign_cross_pos_mask = np.where(dist_ref_sign_cross == 1, 1, 0)
    mean_skewness_quality_diff_pos = np.abs(v_skewness_gray_diff_sub_dist) - np.abs(v_skewness_gray_diff_sub_ref)
    mean_skewness_quality_diff_neg = -np.abs(v_skewness_gray_diff_sub_dist - v_skewness_gray_diff_sub_ref)
    mean_skewness_quality_diff = (dist_ref_sign_cross_pos_mask * mean_skewness_quality_diff_pos) + (dist_ref_sign_cross_neg_mask * mean_skewness_quality_diff_neg)
    mean_skewness_quality_diff = np.sum(mean_skewness_quality_diff)
    return mean_skewness_quality_diff


def img_rgb_plane_mean_skewness_quality_diff_channel_wise_old(v_skewness_gray_diff_sub_ref, v_skewness_gray_diff_sub_dist):
    v_skewness_sub = np.zeros(3)
    v_skewness_sub[0] = v_skewness_gray_diff_sub_dist[0] - v_skewness_gray_diff_sub_ref[0]
    v_skewness_sub[1] = -1 * (np.abs(v_skewness_gray_diff_sub_dist[1]) - np.abs(v_skewness_gray_diff_sub_ref[1]))
    v_skewness_sub[2] = -1 * (v_skewness_gray_diff_sub_dist[2] - v_skewness_gray_diff_sub_ref[2])
    mean_skewness_quality_diff = np.sum(v_skewness_sub)
    return mean_skewness_quality_diff


def long_tail_func(x, p_scalar=2.5, s_scalar=100):
    y = np.exp(-np.power(x, p_scalar) * s_scalar)
    return y


def negative_range_constrained_power_func(x, c, p_scalar=2):
    y = np.abs((1/np.power(c, p_scalar))*np.power((x + c), p_scalar)) - 1
    return y


def img_rgb_plane_mean_skewness_quality_diff_channel_wise(v_skewness_gray_diff_sub_ref, v_skewness_gray_diff_sub_dist, green_sig_scalar=1):
    v_skewness_sub = np.abs(v_skewness_gray_diff_sub_dist) - np.abs(v_skewness_gray_diff_sub_ref)
    v_skewness_sub[1] = -1 * green_sig_scalar * v_skewness_sub[1]
    red_sign = np.abs(np.sign(np.abs(np.sign(v_skewness_gray_diff_sub_ref[0]) - np.sign(v_skewness_gray_diff_sub_dist[0]))))
    blue_sign = np.abs(np.sign(np.abs(np.sign(v_skewness_gray_diff_sub_ref[2]) - np.sign(v_skewness_gray_diff_sub_dist[2]))))
    red_blue_sign = np.abs(red_sign - blue_sign)
    green_sign = np.abs(np.sign(v_skewness_gray_diff_sub_ref[1]) - np.sign(v_skewness_gray_diff_sub_dist[1]))
    if green_sign > 0:
        v_skewness_sub = -1 * v_skewness_sub
    elif red_blue_sign > 0:
        v_red_blue_sign_neg = np.array([red_sign, 1, blue_sign]) * -2
        v_skewness_sub = v_skewness_sub + v_skewness_sub * v_red_blue_sign_neg
    mean_skewness_quality_diff = np.sum(v_skewness_sub)
    if mean_skewness_quality_diff == 0:
        mean_skewness_quality_diff = 1
    return mean_skewness_quality_diff


def rgb_plane_mean_skewness_color_strength(v_skewness_gray_diff_sub, green_scalar=0.3):
    v_color_strength = np.abs(v_skewness_gray_diff_sub)
    color_strength = v_color_strength[0] + green_scalar * v_color_strength[1] + v_color_strength[2]
    return color_strength


def rgb_plane_mean_skewness_color_strength_diff(v_skewness_gray_diff_sub_ref, v_skewness_gray_diff_sub_dist):
    color_strength_ref = rgb_plane_mean_skewness_color_strength(v_skewness_gray_diff_sub_ref)
    color_strength_dist = rgb_plane_mean_skewness_color_strength(v_skewness_gray_diff_sub_dist)
    color_strength_diff = color_strength_dist - color_strength_ref
    return color_strength_ref, color_strength_dist, color_strength_diff


def general_gaussian_func(x, bias, p_scalar, s_scalar, x_center):
    y = np.exp(-(np.power(x - x_center, p_scalar) + bias) * s_scalar)
    return y


def chromatic_gaussian_score_func(x, p_scalar=2.5, s_scalar=100, slow_threshold=3, slow_s_scalar_ratio=2.1, slow_p_scalar_ratio=1.9):
    bias = 0
    x_center = 0
    if x >= slow_threshold:
        s_scalar_slow = s_scalar * slow_s_scalar_ratio
        bias = (np.power(slow_threshold, p_scalar) * s_scalar) / s_scalar_slow
        p_scalar = p_scalar * slow_p_scalar_ratio
        s_scalar = s_scalar_slow
        x_center = slow_threshold
    y = general_gaussian_func(x, bias, p_scalar, s_scalar, x_center)
    return y


def var_biased_power_func(x, c_var_bias, p_scalar=2):
    y = (1 / np.power(c_var_bias, p_scalar)) * np.power((x + c_var_bias), p_scalar) - 1
    return y


def p_scalar_pos(ref_color_strength, s_scalar, bias=0.0, a_scalar=1.0):
    y = a_scalar * np.exp(-s_scalar*ref_color_strength) + bias
    return y


def p_scalar_neg(ref_color_strength, s_scalar, y_base=1.15):
    y = s_scalar * np.tan((np.pi * ref_color_strength)/6) + y_base
    return y


def cal_chromatic_color_strength_score_ratio(color_strength_ref, strength_delta, p_s_scalar_pos=100.0, p_s_scalar_neg=100.0, gaussion_s_scalar=10.0, gaussion_a_scalar=1.0):
    if strength_delta == 0:
        score_ratio = 1
    elif strength_delta > 0:
        p_scalar = p_scalar_pos(color_strength_ref, p_s_scalar_pos, a_scalar=5.6392518847594735, bias=0)
        #p_scalar = 0.35323510819674914#2.8930375122909573
        s_scalar = p_scalar_pos(color_strength_ref, s_scalar=gaussion_s_scalar, a_scalar=gaussion_a_scalar, bias=0)
        #s_scalar = 2.751847064314213#10506.096566786684
        score_ratio = chromatic_gaussian_score_func(strength_delta, p_scalar=p_scalar, s_scalar=s_scalar)
    else:
        p_scalar = p_scalar_neg(color_strength_ref, p_s_scalar_neg)
        score_ratio = var_biased_power_func(strength_delta, color_strength_ref, p_scalar=p_scalar)
    return score_ratio


def get_chromatic_color_strength_score_ratio(v_skewness_gray_diff_sub_ref, v_skewness_gray_diff_sub_dist, p_s_scalar_pos=55.071743597317834, p_s_scalar_neg=100, gaussion_s_scalar=215.9849212750911, gaussion_a_scalar=143968.492104378):
    color_strength_ref, color_strength_dist, color_strength_diff = rgb_plane_mean_skewness_color_strength_diff(v_skewness_gray_diff_sub_ref, v_skewness_gray_diff_sub_dist)
    score_ratio = cal_chromatic_color_strength_score_ratio(color_strength_ref, color_strength_diff, p_s_scalar_pos=p_s_scalar_pos, p_s_scalar_neg=p_s_scalar_neg, gaussion_s_scalar=gaussion_s_scalar, gaussion_a_scalar=gaussion_a_scalar)
    return score_ratio


def img_rgb_plane_mean_skewness_quality_diff_green_sign_dom(v_skewness_gray_diff_sub_ref, v_skewness_gray_diff_sub_dist, green_sig_scalar=1.1):
    v_skewness_sub = np.abs(v_skewness_gray_diff_sub_dist) - np.abs(v_skewness_gray_diff_sub_ref)
    green_sign = np.abs(np.sign(v_skewness_gray_diff_sub_ref[1]) - np.sign(v_skewness_gray_diff_sub_dist[1]))
    if green_sign == 0:
        v_skewness_sub = np.abs(v_skewness_gray_diff_sub_dist) - np.abs(v_skewness_gray_diff_sub_ref)
    else:
        v_skewness_sub = -1 * np.abs(v_skewness_gray_diff_sub_dist - v_skewness_gray_diff_sub_ref)
    mean_skewness_quality_diff = np.sum(v_skewness_sub)
    return mean_skewness_quality_diff


def img_rgb_plane_mean_skewness_quality_diff_score(v_skewness_gray_diff_sub_ref, v_skewness_gray_diff_sub_dist, img_gray_dynamic_range_sub_sign=-1, s_scalar=15, bias=2.5):
    mean_skewness_quality_diff = img_rgb_plane_mean_skewness_quality_diff(v_skewness_gray_diff_sub_ref, v_skewness_gray_diff_sub_dist)
    #if img_gray_dynamic_range_sub_sign > 0:
        #mean_skewness_quality_diff = np.abs(mean_skewness_quality_diff)
    continuity_point = -0.08
    if mean_skewness_quality_diff < continuity_point:
        s_scalar_neg = 1
        bias = img_gray_mean_skewness_diff_sigmoid_neg_flatten_bias(s_scalar, s_scalar_neg, bias, continuity_point)
        s_scalar = s_scalar_neg
    print("mean_skewness_quality_diff", mean_skewness_quality_diff)
    mean_skewness_quality_diff_score = biased_sigmoid(mean_skewness_quality_diff, s_scalar=s_scalar, bias=bias)
    return mean_skewness_quality_diff_score


def img_gray_mean_skewness_diff_sigmoid_neg_flatten_bias(s_scalar_pos, s_scalar_neg, bias_pos, continuity_point):
    bias_neg = (s_scalar_pos - s_scalar_neg)*continuity_point + bias_pos
    return bias_neg


def v_skewness_gray_diff_norm_sub_score(v_skewness_gray_diff_ref, v_skewness_gray_diff_dist):
    v_skewness_gray_diff_ref_norm = np.linalg.norm(v_skewness_gray_diff_ref)
    v_skewness_gray_diff_dist_norm = np.linalg.norm(v_skewness_gray_diff_dist)
    #
    print("v_skewness_gray_diff_ref_norm", v_skewness_gray_diff_ref_norm)
    print("v_skewness_gray_diff_dist_norm", v_skewness_gray_diff_dist_norm)
    #
    norm_sub = v_skewness_gray_diff_dist_norm - v_skewness_gray_diff_ref_norm
    norm_sub = norm_sub# / (v_skewness_gray_diff_dist_norm + v_skewness_gray_diff_ref_norm)
    return norm_sub


def img_rgb_chromatic_mean_skewness_metric(img_rgb_gray_ref, img_rgb_red_ref, img_rgb_green_ref, img_rgb_blue_ref, img_rgb_gray_dist, img_rgb_red_dist, img_rgb_green_dist, img_rgb_blue_dist):
    gray_mean_skewness_ref, v_mean_skewness_ref = img_rgb_plane_mean_skewness(img_rgb_gray_ref, img_rgb_red_ref, img_rgb_green_ref, img_rgb_blue_ref)
    gray_mean_skewness_dist, v_mean_skewness_dist = img_rgb_plane_mean_skewness(img_rgb_gray_dist, img_rgb_red_dist, img_rgb_green_dist, img_rgb_blue_dist)
    v_skewness_gray_diff_ref = img_rgb_plane_mean_skewness_gray_diff_v(v_mean_skewness_ref, gray_mean_skewness_ref)
    v_skewness_gray_diff_dist = img_rgb_plane_mean_skewness_gray_diff_v(v_mean_skewness_dist, gray_mean_skewness_dist)

    #
    norm_sub = v_skewness_gray_diff_norm_sub_score(v_skewness_gray_diff_ref, v_skewness_gray_diff_dist)
    print("norm_sub:", norm_sub)
    chromatic_metric_channel_wise = img_rgb_plane_mean_skewness_quality_diff_channel_wise(v_skewness_gray_diff_ref, v_skewness_gray_diff_dist)
    chromatic_color_strength_score_ratio = get_chromatic_color_strength_score_ratio(v_skewness_gray_diff_ref, v_skewness_gray_diff_dist)
    #chromatic_metric_channel_wise = img_rgb_plane_mean_skewness_quality_diff_green_sign_dom(v_skewness_gray_diff_ref, v_skewness_gray_diff_dist)
    print("chromatic_metric_channel_wise", chromatic_metric_channel_wise)
    print("chromatic_color_strength_score_ratio", chromatic_color_strength_score_ratio)
    #

    skewness_gray_diff_max_ref, v_skewness_gray_diff_sub_ref = img_rgb_plane_mean_skewness_gray_diff_max(v_mean_skewness_ref, gray_mean_skewness_ref)
    skewness_gray_diff_max_dist, v_skewness_gray_diff_sub_dist = img_rgb_plane_mean_skewness_gray_diff_max(v_mean_skewness_dist, gray_mean_skewness_dist)

    # test
    print("v_skewness_gray_diff_sub_dist:", v_skewness_gray_diff_sub_ref)
    print("v_skewness_gray_diff_sub_dist:", v_skewness_gray_diff_sub_dist)
    # test

    #test
    #print(v_skewness_gray_diff_sub_dist - v_skewness_gray_diff_sub_ref, np.sum(v_skewness_gray_diff_sub_dist - v_skewness_gray_diff_sub_ref))
    #mean_skewness_quality_diff = img_rgb_plane_mean_skewness_quality_diff(v_skewness_gray_diff_sub_ref, v_skewness_gray_diff_sub_dist)

    img_gray_dynamic_range_sub, img_gray_dynamic_range_sub_sign = img_gray_dynamic_range_sub_cal(img_rgb_gray_ref, img_rgb_gray_dist)
    skewness_diff_max_sub_score = img_rgb_plane_mean_skewness_quality_diff_score(v_skewness_gray_diff_sub_ref, v_skewness_gray_diff_sub_dist, img_gray_dynamic_range_sub_sign=img_gray_dynamic_range_sub_sign)

    #print(biased_sigmoid(mean_skewness_quality_diff, s_scalar=25, bias=4))
    #test
    v_skewness_diff_sub, v_skewness_diff_distance_score = img_rgb_plane_mean_skewness_diff_v_dist_ref_distance(v_skewness_gray_diff_ref, v_skewness_gray_diff_dist)
    #
    print("v_skewness_distance:", v_skewness_diff_sub)
    #
    #skewness_diff_max_sub, skewness_diff_max_sub_score = img_rgb_plane_mean_skewness_diff_v_dist_ref_max_diff(skewness_gray_diff_max_ref, skewness_gray_diff_max_dist)
    return v_skewness_diff_distance_score, skewness_diff_max_sub_score


def img_gray_sim_metrics(img_gray_ref, img_gray_dist, seg_num, is_segment_mod, centroid_num, C_PC=1, C_HIST=0.001, C_SUM=0.001, C_GRAD=0.001, C_DYNA_R=0.001, mean_scalar=10, kernel_size=(3, 3), conv_mode='same', denoise_threshold=9, C_SIG=0.01, is_rec_aligned=True):
    img_gray_1 = img_gray_ref
    img_gray_2 = img_gray_dist

    img_plane_list_1, img_gray_plane_sum_1, img_gray_rec_1 = img_gray_adjusted_planes_extraction_cluster(img_gray_1, seg_num=seg_num, is_segment_mod=is_segment_mod, centroid_num=centroid_num, is_rec_aligned=is_rec_aligned)
    img_plane_list_2, img_gray_plane_sum_2, img_gray_rec_2 = img_gray_adjusted_planes_extraction_cluster(img_gray_2, seg_num=seg_num, is_segment_mod=is_segment_mod, centroid_num=centroid_num, is_rec_aligned=is_rec_aligned)

    #img_plane_list_1_s, img_gray_plane_sum_1_s, img_gray_rec_1_s = img_gray_adjusted_planes_extraction_cluster(img_gray_1, seg_num=8, is_segment_mod=is_segment_mod, centroid_num=centroid_num, is_rec_aligned=is_rec_aligned)
    #img_plane_list_2_s, img_gray_plane_sum_2_s, img_gray_rec_2_s = img_gray_adjusted_planes_extraction_cluster(img_gray_2, seg_num=8, is_segment_mod=is_segment_mod, centroid_num=centroid_num, is_rec_aligned=is_rec_aligned)

    img_gray_hist_correlation_matrix, img_gray_hist_correlation = img_plane_hist_diff2_correlation(img_gray_1, img_gray_2, C=C_HIST)

    #pc_sum_correlation_matrix, pc_sum_correlation = img_plane_pc_correlation(img_gray_rec_1, img_gray_rec_2, C=C_PC)
    pc_sum_correlation_matrix, pc_sum_correlation = img_plane_pc_correlation(img_gray_1, img_gray_2, C=C_PC)
    #pc_sum_correlation_matrix, pc_sum_correlation = img_plane_pc_correlation(img_gray_plane_sum_1, img_gray_plane_sum_2, C=C_PC)

    img_plane_sum_correlation_matrix, img_plane_sum_correlation = hvs_sim_img_plane_correlation_cal(img_gray_plane_sum_1, img_gray_plane_sum_2, C=C_SUM)

    #img_plane_correlation_mean = img_gray_plane_mean_diff(img_gray_1, img_gray_2, mean_scalar=mean_scalar)
    img_plane_correlation_mean = img_gray_plane_mean_positive_negative_diff_diff(img_gray_1, img_gray_2, mean_scalar=mean_scalar)

    #img_plane_correlation_gradient_matrix, img_plane_correlation_gradient = hvs_sim_img_plane_correlation_cal(np.round(img_gray_rec_1*255, 0), np.round(img_gray_rec_2*255, 0), C=C_GRAD)
    img_plane_correlation_gradient_matrix, img_plane_correlation_gradient = img_gray_plane_gradient_correlation(img_gray_1, img_gray_2, C=C_GRAD)
    #img_plane_correlation_gradient_matrix, img_plane_correlation_gradient = img_gray_plane_gradient_correlation(img_gray_rec_1_b, img_gray_rec_2_b, C=C_GRAD)

    plane_signal_level_correlation = img_gray_plane_signal_level_correlation(img_plane_list_1, img_plane_list_2, kernel_size=kernel_size, conv_mode=conv_mode, denoise_threshold=denoise_threshold, C=C_SIG)

    #img_dynamic_range_correlation = img_gray_dynamic_range_sim(img_gray_1, img_gray_2, C=C_DYNA_R)
    img_dynamic_range_correlation, img_gray_dynamic_range_sub_sign = img_gray_dynamic_range_positive_negative_diff_sim(img_gray_1, img_gray_2, C=C_DYNA_R)

    img_gray_planes_binary_shape_sim_val = img_gray_planes_binary_shape_sim(img_plane_list_1, img_plane_list_2)

    skewness_diff, skewness_diff_score = img_gray_mean_skewness_diff(img_gray_1, img_gray_2)

    return pc_sum_correlation, img_gray_hist_correlation, img_plane_sum_correlation, img_plane_correlation_mean, img_plane_correlation_gradient, plane_signal_level_correlation, img_dynamic_range_correlation, img_gray_planes_binary_shape_sim_val, skewness_diff_score


def cal_p_s_scalar_gaussian(x1, x2, v1, v2):
    v3 = np.log(v1) / np.log(v2)
    x3 = x1 / x2
    p_scalar = np.log(v3) / np.log(x3)
    s_scalar = -np.log(v2) / np.power(x2, p_scalar)
    return p_scalar, s_scalar


def cal_a_s_scalar_gaussian_p(c1, c2, v1, v2):
    c3 = 1 / (c1 - c2)
    v3 = v1 / v2
    s_scalar = -c3 * np.log(v3)
    a_scalar = v1/(np.exp(-s_scalar * c1))
    return s_scalar, a_scalar
