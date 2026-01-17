import numpy as np
import cv2
import torch
import time as tm

IMAGE_ROUND_PRECISION = 8

TORCH_ONE = torch.as_tensor(1)
TORCH_ZERO = torch.as_tensor(0)
DEVICE = torch.device('cpu')


def set_param_torch(distance_seg_num, device=torch.device('cpu')):
    global DISTANCE_SEG_NUM
    global DEVICE
    DISTANCE_SEG_NUM = distance_seg_num
    DISTANCE_SEG_NUM = torch.as_tensor(DISTANCE_SEG_NUM).to(device)
    TORCH_ONE.to(device)
    TORCH_ZERO.to(device)
    DEVICE = device


def obj_to_device(obj, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    obj_device = torch.as_tensor(obj, device=device)
    return obj_device, device


def img_integer_segmentation_equal_range_thresholds_gpu(img, seg_num, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_tensor = torch.as_tensor(img, device=device)
    seg_num_tensor = torch.as_tensor(seg_num, device=device)
    thresholds = img_integer_segmentation_equal_range_thresholds_gpu_do(img_tensor, seg_num_tensor, device)
    return thresholds


def img_integer_segmentation_equal_range_thresholds_gpu_do(img_tensor, seg_num_tensor, device):
    img_max = torch.max(img_tensor)
    img_min = torch.min(img_tensor)

    img_single_step_value = ((img_max - img_min) / seg_num_tensor).to(torch.uint8)

    indices = torch.arange(1, seg_num_tensor + 0, 1, device=device)

    thresholds = img_max - img_single_step_value * indices

    img_min_append = img_min.unsqueeze(0)
    thresholds = torch.cat([thresholds, img_min_append])
    return thresholds


def img_integer_segmentation_equal_range_thresholds_cpu(img, seg_num):
    img_max = np.max(img)
    img_min = np.min(img)

    img_single_step_value = np.int_((img_max - img_min) / seg_num)

    indices = np.arange(1, seg_num + 0, 1)

    thresholds = img_max - img_single_step_value * indices

    thresholds = np.append(thresholds, img_min)
    return thresholds


def img_integer_segmentation_equal_range_thresholds_float_gpu_combined(img, seg_num, device=None):
    device = 'cuda'  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_tensor = torch.as_tensor(img, device=device)
    seg_num_tensor = torch.as_tensor(seg_num, device=device)
    #img_tensor, seg_num_tensor, device
    img_max = torch.max(img_tensor)
    img_min = torch.min(img_tensor)

    img_single_step_value = ((img_max - img_min) / seg_num_tensor)

    indices = torch.arange(1, seg_num_tensor + 0, 1, device=device)

    thresholds = img_max - img_single_step_value * indices

    img_min_append = img_min.unsqueeze(0)
    thresholds = torch.cat([thresholds, img_min_append])
    return thresholds


def img_integer_segmentation_equal_range_thresholds_float_gpu(img, seg_num, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_tensor = torch.as_tensor(img, device=device)
    seg_num_tensor = torch.as_tensor(seg_num, device=device)
    thresholds = img_integer_segmentation_equal_range_thresholds_float_gpu_do(img_tensor, seg_num_tensor, device)
    return thresholds


def img_integer_segmentation_equal_range_thresholds_float_gpu_do(img_tensor, seg_num_tensor, device):
    img_max = torch.max(img_tensor)
    img_min = torch.min(img_tensor)

    img_single_step_value = ((img_max - img_min) / seg_num_tensor)

    indices = torch.arange(1, seg_num_tensor + 0, 1, device=device)

    thresholds = img_max - img_single_step_value * indices

    img_min_append = img_min.unsqueeze(0)
    thresholds = torch.cat([thresholds, img_min_append])
    return thresholds


def img_segmentation_threshold_list_gpu(img, thresholds, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_tensor = torch.as_tensor(img, device=device)
    thresholds_tensor = torch.as_tensor(thresholds, device=device)
    segmentation_results = img_segmentation_threshold_list_gpu_do(img_tensor, thresholds_tensor, device)
    return segmentation_results


def img_segmentation_threshold_list_gpu_do(img_tensor, thresholds, device=None):

    threshold_num = thresholds.shape[0]
    img_shape = img_tensor.shape

    segmentation_results = torch.empty((threshold_num, *img_shape), device=device, dtype=img_tensor.dtype)

    current_seg_residual = img_tensor.clone()
    target_dtype = img_tensor.dtype

    for i in range(threshold_num):
        precision_proc_threshold = thresholds[i]
        segmentation_results[i] = torch.where(
            current_seg_residual >= precision_proc_threshold,
            torch.tensor(1, device=device),
            torch.tensor(0, device=device)
        )
        current_seg_residual = torch.where(
            current_seg_residual >= precision_proc_threshold,
            torch.tensor(-1, dtype=target_dtype, device=device),
            current_seg_residual
        )
    return segmentation_results


def img_integer_segmentation_equal_range_thresholds_float_cpu(img, seg_num):
    img_max = np.max(img)
    img_min = np.min(img)
    img_single_step_value = ((img_max - img_min) / seg_num)
    indices = torch.arange(1, seg_num + 0, 1)
    thresholds = img_max - img_single_step_value * indices
    thresholds = np.append(thresholds, img_min)
    return thresholds


def img_segmentation_threshold_list_cpu(img_array, thresholds):
    threshold_num = thresholds.shape[0]
    img_shape = img_array.shape
    segmentation_results = []
    #segmentation_results = np.empty((threshold_num, *img_shape), dtype=img_array.dtype)
    #current_seg_residual = img_array.copy()
    current_seg_residual = np.round(img_array, IMAGE_ROUND_PRECISION)
    i = 0
    while i < threshold_num:
        precision_proc_threshold = thresholds[i]

        #segmentation_results[i] =
        segmentation_result = np.where(current_seg_residual >= precision_proc_threshold,1,0)
        current_seg_residual = np.where(current_seg_residual >= precision_proc_threshold, -1, current_seg_residual)
        segmentation_results.append(segmentation_result)
        i += 1
    return segmentation_results
