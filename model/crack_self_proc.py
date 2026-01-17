import matplotlib.pyplot as plt
import crack500_support as cr5_spt
import gauss_fun_generator as gs_gen
import numpy as np


self_file_name_path_pref = "D:/source_code/exp_code/11215-"
#self_file_name_ext = ".jpg"
SELF_FILE_NAME_EXT = ".jpg"
COL_LIST_EXTRACTION_TYPE_COL = 0
COL_LIST_EXTRACTION_TYPE_ROW = 1

GRAD_SEQ_INVERSE = 0
GRAD_SEQ_REGULAR = 1

IMG_POISSON_SEG = True
IMG_NO_POISSON_SEG = False
IMG_GAUSS_BLUR = 0
IMG_SIMPLE_BLUR_POISSON = 1
IMG_SIMPLE_BLUR_NO_POISSON = 2


def extract_curve_display_segment(self_img_num, col_list_extraction_type, seg_start, point_start, point_end, is_poisson_seg=IMG_NO_POISSON_SEG, self_file_name_ext=SELF_FILE_NAME_EXT):
    img_path_name1 = cr5_spt.generate_self_image_file_path_name(self_file_name_path_pref, self_file_name_ext, self_img_num)
    img_rgb1 = cr5_spt.get_rgb_img_from_file(img_path_name1)
    img_gray1 = get_self_img_gray(self_img_num)#cr5_spt.get_gray_img_from_file(img_path_name1)
    if is_poisson_seg:
        img_gray1 = cr5_spt.get_img_gray_poisson_seg(img_gray1)
    elif is_poisson_seg == IMG_GAUSS_BLUR:
        img_gray1 = cr5_spt.get_img_gray_gauss_blur(img_gray1)
    elif is_poisson_seg == IMG_SIMPLE_BLUR_POISSON:
        img_gray1 = cr5_spt.get_img_gray_simple_blur(img_gray1)
        img_gray1 = cr5_spt.get_img_gray_poisson_seg(img_gray1)
    elif is_poisson_seg == IMG_SIMPLE_BLUR_NO_POISSON:
        img_gray1 = cr5_spt.get_img_gray_simple_blur(img_gray1)
    if col_list_extraction_type == COL_LIST_EXTRACTION_TYPE_COL:
        img_crack_nonzero_col_list = cr5_spt.img_gray_nonzero_val_col_extract(img_gray1, True)
    else:
        img_crack_nonzero_col_list = cr5_spt.img_gray_nonzero_val_row_extract(img_gray1, True)
    img_crack_curve_segment = cr5_spt.val_curve_extract(img_crack_nonzero_col_list[seg_start], [point_start, point_end])
    img_gray1_cracked_display_col = cr5_spt.val_curve_display_extract(img_crack_curve_segment)
    return [img_rgb1, img_gray1, img_gray1_cracked_display_col]


def self_img_analysis(img_num, proc_direction, seg_start, start_point, end_point, grad_seq_type=GRAD_SEQ_INVERSE, is_poisson_seg=IMG_NO_POISSON_SEG, seg_mark_intensity=cr5_spt.SEG_MARK_DARK):
    self_img_seg = extract_curve_display_segment(img_num, proc_direction, seg_start, start_point, end_point, is_poisson_seg)
    seg_direction = get_seg_direction(proc_direction)
    img_gray = cr5_spt.mark_img_gray_seg(self_img_seg[1], seg_start, seg_direction, start_point, end_point, mark_intensity=seg_mark_intensity)
    grad_seq_fun = get_grad_seq_fun(grad_seq_type)
    self_img_grad = grad_seq_fun(self_img_seg[2][1])
    display_self_img_grad = cr5_spt.val_curve_display_extract(self_img_grad)
    seg_triple_monotonic_info = gs_gen.get_seg_triple_monotonic_delta(self_img_seg[2][1], start_point)
    display_seg_triple_monotonic_delta = cr5_spt.val_curve_display_extract(seg_triple_monotonic_info[0])
    display_seg_triple_monotonic_data = cr5_spt.val_curve_display_extract(seg_triple_monotonic_info[1])
    #plt.imshow(img_gray, cmap='gray')
    #plt.show()
    plt.subplot(2, 2, 1)
    plt.imshow(self_img_seg[0])
    plt.subplot(2, 2, 2)
    plt.imshow(img_gray, cmap='gray', vmin=0, vmax=1)
    #plt.imshow(self_img_seg[1], cmap='gray')
    plt.subplot(2, 2, 3)
    plt.plot(self_img_seg[2][0], self_img_seg[2][1], marker='+', markersize=10)
    plt.subplot(2, 2, 4)
    plt.plot(display_self_img_grad[0], display_self_img_grad[1], marker='+', markersize=10)
    plt.show()
    # plt.subplot(2, 1, 1)

    # plt.subplot(2, 1, 2)
    plt.imshow(self_img_seg[1], cmap='gray')
    plt.show()
    plt.plot(display_seg_triple_monotonic_delta[0], display_seg_triple_monotonic_delta[1], color='blue', label='delta')
    plt.plot(display_seg_triple_monotonic_data[0], display_seg_triple_monotonic_data[1], color='red', label='data')
    plt.legend()
    plt.show()
    return [self_img_seg[1], seg_triple_monotonic_info, self_img_seg[2][1]]


def get_self_img_gray(self_img_num, img_map_idx=cr5_spt.IMG_MAP_GRAY_IDX, self_file_name_ext=SELF_FILE_NAME_EXT):
    img_path_name1 = cr5_spt.generate_self_image_file_path_name(self_file_name_path_pref, self_file_name_ext, self_img_num)
    img_gray1 = cr5_spt.get_gray_img_from_file(img_path_name1, img_map_idx)
    return img_gray1


def get_self_img_rgb(self_img_num, self_file_name_ext=SELF_FILE_NAME_EXT):
    img_path_name1 = cr5_spt.generate_self_image_file_path_name(self_file_name_path_pref, self_file_name_ext, self_img_num)
    img_rgb = cr5_spt.get_rgb_img_from_file(img_path_name1)/255
    return img_rgb


def get_self_img_original(self_img_num, self_file_name_ext=SELF_FILE_NAME_EXT):
    img_path_name1 = cr5_spt.generate_self_image_file_path_name(self_file_name_path_pref, self_file_name_ext, self_img_num)
    img_original = cr5_spt.get_rgb_img_from_file(img_path_name1)
    return img_original


def get_self_img_original_bgr(self_img_num, self_file_name_ext=SELF_FILE_NAME_EXT):
    img_path_name1 = cr5_spt.generate_self_image_file_path_name(self_file_name_path_pref, self_file_name_ext, self_img_num)
    img_original = cr5_spt.get_bgr_img_from_file(img_path_name1)
    return img_original


def get_seg_direction(proc_direction):
    if proc_direction == COL_LIST_EXTRACTION_TYPE_COL:
        seg_direction = cr5_spt.SEG_DIRECTION_COL
    else:
        seg_direction = cr5_spt.SEG_DIRECTION_ROW
    return seg_direction


def get_grad_seq_fun(grad_seq_type):
    if grad_seq_type == GRAD_SEQ_INVERSE:
        grad_seq_fun = gs_gen.get_seg_grad_inv_seq
    else:
        grad_seq_fun = gs_gen.get_seg_grad
    return grad_seq_fun


def get_self_img_gray_batch(self_img_num_list):
    img_num_len = len(self_img_num_list)
    i = 0
    img_batch_list = []
    while i < img_num_len:
        self_img_num = self_img_num_list[i]
        img_gray = get_self_img_gray(self_img_num)
        img_batch_list.append(img_gray)
        i += 1
    return img_batch_list


def array_idx_flip(org_idx, array_dim_len):
    flip_idx = array_dim_len - org_idx
    return flip_idx


def seg_idx_flip(seg_start_idx, point_start_idx, point_end_idx, img_shape, extraction_direction):
    seg_shape = img_shape
    if extraction_direction == COL_LIST_EXTRACTION_TYPE_COL:
        seg_shape = img_shape[::-1]
    seg_dim_len = seg_shape[0]
    point_dim_len = seg_shape[1]
    seg_start_flip = array_idx_flip(seg_start_idx, seg_dim_len)
    point_end_flip = array_idx_flip(point_start_idx, point_dim_len)
    point_start_flip = array_idx_flip(point_end_idx, point_dim_len)
    return seg_start_flip, point_start_flip, point_end_flip


def analyse_img_seg(img_seg):
    img_seg_delta = np.diff(img_seg)
    code_shape_metric, code_shape, pixel_num_list = cr5_spt.monocular_3d_code_parse_analysis(img_seg_delta)
    return code_shape_metric, code_shape, pixel_num_list, img_seg_delta


def get_normalized_shape_metric(code_shape_metric, pixel_num_list):
    fill_num_array = pixel_num_list
    normalized_shape_metric = code_shape_metric / pixel_num_list
    normalized_shape_metric_sum = cr5_spt.one_dim_array_gard_sum(normalized_shape_metric, fill_num_array)
    return normalized_shape_metric_sum


def get_mark_direction_from_extraction_direction(extraction_direction):
    if extraction_direction == COL_LIST_EXTRACTION_TYPE_COL:
        mark_direction = cr5_spt.SEG_DIRECTION_COL
    else:
        mark_direction = cr5_spt.SEG_DIRECTION_ROW
    return mark_direction


def plot_seg_array(seg_array, plot_marker="", plot_marker_size=10):
    display_seg_info = cr5_spt.val_curve_display_extract(seg_array)
    display_seg_aix = display_seg_info[0]
    display_seg_val = display_seg_info[1]
    plt.plot(display_seg_aix, display_seg_val, marker=plot_marker, markersize=plot_marker_size)
    return display_seg_info


def double_direction_monocular_3d_code_analysis(img_num, seg_start, point_start, point_end, extraction_direction, seg_method, mark_intensity, is_avg_pool=False, avg_pool_param=None):
    self_img_seg_info = extract_curve_display_segment(img_num, extraction_direction, seg_start, point_start, point_end, seg_method)
    self_img_seg = self_img_seg_info[2][1]
    self_img_seg_rotate = np.flip(self_img_seg)
    code_shape_metric, code_shape, pixel_num_list, img_seg_delta = analyse_img_seg(self_img_seg)
    code_shape_metric_rotate, code_shape_rotate, pixel_num_list_rotate, img_seg_delta_rotate = analyse_img_seg(self_img_seg_rotate)
    normalized_shape_metric_sum = get_normalized_shape_metric(code_shape_metric, pixel_num_list)
    normalized_shape_metric_sum_rotate = get_normalized_shape_metric(code_shape_metric_rotate, pixel_num_list_rotate)
    img_gray = self_img_seg_info[1]
    if is_avg_pool:
        if avg_pool_param is None:
            img_gray = cr5_spt.img_gray_avg_pool(img_gray)
        else:
            img_gray = cr5_spt.img_gray_avg_pool(img_gray, avg_pool_param[0], avg_pool_param[1])
    img_gray_rotate = np.flip(img_gray)
    seg_start_rotate, point_start_rotate, point_end_rotate = seg_idx_flip(seg_start, point_start, point_end, img_gray_rotate.shape, extraction_direction)
    mark_direction = get_mark_direction_from_extraction_direction(extraction_direction)
    img_gray_marked = cr5_spt.mark_img_gray_seg(img_gray, seg_start, mark_direction, point_start, point_end, mark_intensity=mark_intensity)
    img_gray_rotate_marked = cr5_spt.mark_img_gray_seg(img_gray_rotate, seg_start_rotate, mark_direction, point_start_rotate, point_end_rotate, mark_intensity=mark_intensity)
    plt.subplot(4, 2, 1)
    plt.imshow(img_gray_marked, cmap='gray', vmin=0, vmax=1)
    plt.subplot(4, 2, 2)
    plt.imshow(img_gray_rotate_marked, cmap='gray', vmin=0, vmax=1)
    plt.subplot(4, 2, 3)
    plot_seg_array(normalized_shape_metric_sum)
    plot_seg_array(np.flip(normalized_shape_metric_sum_rotate))
    plt.subplot(4, 2, 4)
    plot_seg_array(normalized_shape_metric_sum_rotate)
    plt.subplot(4, 2, 5)
    plot_seg_array(self_img_seg, plot_marker="")
    plt.subplot(4, 2, 6)
    plot_seg_array(self_img_seg_rotate, plot_marker="")
    plt.subplot(4, 2, 7)
    plot_seg_array(img_seg_delta, plot_marker="")
    plt.subplot(4, 2, 8)
    plot_seg_array(img_seg_delta_rotate, plot_marker="")
    plt.show()
    return normalized_shape_metric_sum, normalized_shape_metric_sum_rotate
