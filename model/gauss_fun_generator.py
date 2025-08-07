import numpy as np
from sklearn.linear_model import LinearRegression
import math

TRIPLE_PIXEL_GRAD_MONOTONIC_NO = 0
TRIPLE_PIXEL_GRAD_MONOTONIC_CONCAVE = 1
TRIPLE_PIXEL_GRAD_MONOTONIC_BULGE = 2
TRIPLE_PIXEL_GRAD_MONOTONIC_SIMPLE = 3
TRIPLE_PIXEL_GRAD_MONOTONIC_SIMPLE_VIBRATE = 5
TRIPLE_PIXEL_GRAD_MONOTONIC_SIMPLE_FLAT = 6
TRIPLE_PIXEL_GRAD_MONOTONIC_LINEAR = 4
TRIPLE_PIXEL_GRAD_MONOTONIC_LINEAR_BULGE = 7
TRIPLE_PIXEL_GRAD_MONOTONIC_LINEAR_CONCAVE = 8
TRIPLE_PIXEL_GRAD_MONOTONIC_SIMPLE_VIBRATE_FLAT = 9
TRIPLE_PIXEL_IS_MONOTONIC_IDX = 0
TRIPLE_PIXEL_GRAD_MONOTONIC_TYPE_IDX = 1
TRIPLE_PIXEL_TOTAL_DELTA_VAL_IDX = 2
TRIPLE_PIXEL_MIN_DATA_IDX = 3
TRIPLE_PIXEL_MAX_DATA_IDX = 4
TRIPLE_PIXEL_ABS_TOTAL_DELTA_VAL_IDX = 5
TRIPLE_PIXEL_TOTAL_DELTA_VAL_SIGN_IDX = 6
TRIPLE_PIXEL_UPDATE_IDX = 7
TRIPLE_PIXEL_MAX_DATA_TOP = 0
TRIPLE_PIXEL_MIN_DATA_TOP = 1
TRIPLE_PIXEL_DATA_TOP_TYPE_IDX = 8
TRIPLE_PIXEL_ABS_GRAD_DELTA_RATIO_IDX = 9
TRIPLE_PIXEL_IS_STRICT_MONOTONIC_IDX = 10
TRIPLE_PIXEL_DELTA_1_IDX = 11
TRIPLE_PIXEL_DELTA_2_IDX = 12
TRIPLE_PIXEL_CURVE_SIDE_IDX = 13
TRIPLE_PIXEL_ABS_DELTA_1_IDX = 14
TRIPLE_PIXEL_ABS_DELTA_2_IDX = 15
TRIPLE_PIXEL_ABS_DELTA_SUM_IDX = 16
TRIPLE_PIXEL_CURVE_SIDE_LEFT = True
TRIPLE_PIXEL_CURVE_SIDE_RIGHT = False
TRIPLE_PIXEL_MONOTONIC_DIRECTION_UPWARD = 0
TRIPLE_PIXEL_MONOTONIC_DIRECTION_DOWNWARD = 1
TRIPLE_PIXEL_MONOTONIC_DIRECTION_VIBRATE = 2
TRIPLE_PIXEL_MONOTONIC_DIRECTION_IDX = 17
TRIPLE_PIXEL_VIBRATE_DIRECTION_DOWNWARD = 0
TRIPLE_PIXEL_VIBRATE_DIRECTION_UPWARD = 1
TRIPLE_PIXEL_VIBRATE_DIRECTION_NO = 2
TRIPLE_PIXEL_VIBRATE_DIRECTION_IDX = 18
RANGE_POWER_RATIO = 0.3


def gauss_fun(x, a=1, b=0, c=1):
    coefficient1 = np.power((x-b), 2)
    coefficient2 = 2*np.power(c, 2)
    y = a*np.exp((-1)*(coefficient1/coefficient2))
    return y


def gauss_fun_transpose(x, a=1, b=0, c=1):
    y = gauss_fun(x, a, b, c)
    y = a - y
    return y


def gauss_fun_trans_grad(x, c=1, a=1, b=0):
    y = gauss_fun(x, a, b, c)
    y = y*(x-b)
    y = y/np.power(c, 2)
    return y


def neg_exp_trans(x):
    x_abs_neg = (-1)*(np.abs(x))
    y = 1 - np.exp(x_abs_neg)
    return y


def neg_exp_trans_grad(x):
    x_abs_neg = (-1)*(np.abs(x))
    y = np.exp(x_abs_neg)
    return y


def sigmoid_fun(x, t, alpha, miu, a=1):
    y = alpha*(a*x-miu)
    y = np.exp((-1)*(y/t))
    y += 1
    y = 1/y
    return y


def sigmoid_fun_grad(x, t, alpha, miu, a=1):
    y = alpha*(a*x-miu)
    y = alpha*np.exp((-1)*(y/t))
    y = y/t
    y = y*np.power(sigmoid_fun(x, t, alpha, miu, a), 2)
    return y


def square_fun(x, a, b, c):
    y = a*np.power(x, 2)+b*x+c
    return y


def square_fun_grad(x, a, b, c=0):
    y = 2*a*x+b
    return y


def linear_fun(x, a, b):
    y = a*x+b
    return y


def linear_fun_grad(x, a, b):
    y = a
    return y


def cubic_fun(x, a, b, c, d):
    x2 = np.power(x, 2)
    x3 = x2*x
    y = a*x3+b*x2+c*x+d
    return y


def cubic_fun_grad(x, a, b, c, d):
    y = 3*a*np.power(x, 2)+2*b*x+c
    return y


def square_root_fun(x, a):
    y = a*np.power(x, 0.5)
    return y


def square_root_fun_grad(x, a):
    y = (1/2)*a*(1/np.power(x, 0.5))
    return y


def linear_parameter_resolver(val1, val2, x1, x2):
    alpha = (val2-val1)/(x2-x1)
    b = val1 - alpha*x1
    return [alpha, b]


def get_seg_grad(seg):
    seg_len = len(seg)
    seg_grad_list = [None]*(seg_len-1)
    i = 1
    while i < seg_len:
        grad_idx = i - 1
        seg_grad = seg[i] - seg[grad_idx]
        seg_grad_list[grad_idx] = seg_grad
        i += 1
    return seg_grad_list


def get_seg_grad_inv_seq(seg):
    seg_len = len(seg)
    seg_grad_list = [None]*(seg_len-1)
    i = seg_len-1
    j = 0
    while i > 0:
        seg_grad = seg[i-1] - seg[i]
        seg_grad_list[j] = seg_grad
        i -= 1
        j += 1
    return seg_grad_list


def get_seg_triple_monotonic_delta(seg, seg_start_point):
    seg_len = len(seg)
    delta_list = []
    mini_data_list = []
    center_point_list = []
    filtered_delta_sign_list = []
    i = 2
    while i < seg_len:
        #delta_1 = seg[i-1]-seg[i-2]
        #delta_2 = seg[i] - seg[i-1]
        triple_pixel_info = seg_triple_pixel_monotonic_analysis(seg, i)
        if triple_pixel_info[TRIPLE_PIXEL_IS_MONOTONIC_IDX] and triple_pixel_info[TRIPLE_PIXEL_GRAD_MONOTONIC_TYPE_IDX] == TRIPLE_PIXEL_GRAD_MONOTONIC_CONCAVE:
            delta = triple_pixel_info[TRIPLE_PIXEL_TOTAL_DELTA_VAL_IDX]
            abs_delta = triple_pixel_info[TRIPLE_PIXEL_ABS_TOTAL_DELTA_VAL_IDX]
            delta_list.append(delta)
            mini_data = triple_pixel_info[TRIPLE_PIXEL_MIN_DATA_IDX]
            mini_data_list.append(mini_data)
            if mini_data <= 0.32 and abs_delta >= 0.15: #temp code 0.3
                center_point_list.append(i-1)
                filtered_delta_sign_list.append(triple_pixel_info[TRIPLE_PIXEL_TOTAL_DELTA_VAL_SIGN_IDX])
        i += 1
    triple_monotonic_delta = np.array(delta_list)
    triple_monotonic_data = np.array(mini_data_list)
    tripe_monotonic_center_point = np.array(center_point_list)
    triple_monotonic_delta_sign = np.array(filtered_delta_sign_list)
    return [triple_monotonic_delta, triple_monotonic_data, tripe_monotonic_center_point+seg_start_point, triple_monotonic_delta_sign]


def seg_triple_pixel_monotonic_analysis(seg, top_index):
    is_monotonic = False
    is_strict_monotonic = False
    grad_monotonic_type = TRIPLE_PIXEL_GRAD_MONOTONIC_NO
    total_delta_val = 0
    abs_total_delta_val = 0
    total_delta_sign = 0
    min_data = 0
    max_data = 0
    delta_1 = seg[top_index - 1] - seg[top_index - 2]
    delta_2 = seg[top_index] - seg[top_index - 1]
    curve_side = delta_2 < 0
    delta_sign_checker = np.sign(delta_1)*np.sign(delta_2)
    edge_idx = top_index
    curve_top_type = TRIPLE_PIXEL_MIN_DATA_TOP
    total_delta_val = seg[top_index] - seg[top_index - 2]
    abs_total_delta_val = np.abs(total_delta_val)
    abs_delta_1 = np.abs(delta_1)
    abs_delta_2 = np.abs(delta_2)
    total_abs_delta = abs_delta_1 + abs_delta_2
    if total_abs_delta > 0:
        abs_grad_delta_ratio = np.fmax(abs_delta_1, abs_delta_2)/total_abs_delta
    else:
        abs_grad_delta_ratio = 0
    pixel_monotonic_direction = TRIPLE_PIXEL_MONOTONIC_DIRECTION_VIBRATE
    vibrate_direction = TRIPLE_PIXEL_VIBRATE_DIRECTION_NO
    if delta_1 == 0 and delta_2 == 0:
        grad_monotonic_type = TRIPLE_PIXEL_GRAD_MONOTONIC_SIMPLE_FLAT
        is_strict_monotonic = False
    elif delta_sign_checker <= 0: #< 0:
        if delta_sign_checker < 0:
            grad_monotonic_type = TRIPLE_PIXEL_GRAD_MONOTONIC_SIMPLE_VIBRATE
        else:
            grad_monotonic_type =   TRIPLE_PIXEL_GRAD_MONOTONIC_SIMPLE_VIBRATE_FLAT
        if delta_2 <= 0: #< 0:
            vibrate_direction = TRIPLE_PIXEL_VIBRATE_DIRECTION_DOWNWARD
        else:
            vibrate_direction = TRIPLE_PIXEL_VIBRATE_DIRECTION_UPWARD
    elif delta_sign_checker > 0 or (delta_sign_checker == 0 and (delta_1 != 0 or delta_2 != 0)):
        #abs_grad_delta_ratio = np.fmax(abs_delta_1, abs_delta_2) / total_abs_delta
        is_monotonic = True
        is_strict_monotonic = True
        if delta_sign_checker > 0:
            if delta_1 < 0 and delta_2 < 0:
                pixel_monotonic_direction = TRIPLE_PIXEL_MONOTONIC_DIRECTION_DOWNWARD
            elif delta_1 > 0 and delta_2 > 0:
                pixel_monotonic_direction = TRIPLE_PIXEL_MONOTONIC_DIRECTION_UPWARD
        if delta_1 == 0 or delta_2 == 0:
            grad_monotonic_type = TRIPLE_PIXEL_GRAD_MONOTONIC_SIMPLE
            is_strict_monotonic = False
        elif delta_2 < delta_1:
            grad_monotonic_type = TRIPLE_PIXEL_GRAD_MONOTONIC_CONCAVE
        elif delta_2 == delta_1:
            #grad_monotonic_type = TRIPLE_PIXEL_GRAD_MONOTONIC_LINEAR
            if delta_1 < 0:
                grad_monotonic_type = TRIPLE_PIXEL_GRAD_MONOTONIC_LINEAR_CONCAVE
            else:
                grad_monotonic_type = TRIPLE_PIXEL_GRAD_MONOTONIC_LINEAR_BULGE
        else:
            grad_monotonic_type = TRIPLE_PIXEL_GRAD_MONOTONIC_BULGE
        #total_delta_val = seg[top_index] - seg[top_index - 2]
        #abs_total_delta_val = np.abs(total_delta_val)
        total_delta_sign = np.sign(total_delta_val)
        '''
        if delta_sign_checker > 0:
            total_delta_sign = np.sign(total_delta_val)
            edge_idx = top_index
        else:
            total_delta_sign = 0
            edge_idx = top_index - 2
        '''
        edge_idx = top_index - 1
        min_data = np.fmin(seg[top_index], seg[top_index - 2])
        max_data = np.fmax(seg[top_index], seg[top_index - 2])
        if min_data == seg[top_index - 2]:
            curve_top_type = TRIPLE_PIXEL_MAX_DATA_TOP
        #else:
            #curve_top_type = TRIPLE_PIXEL_MIN_DATA_TOP
    else:
        grad_monotonic_type = TRIPLE_PIXEL_GRAD_MONOTONIC_SIMPLE_FLAT
    return [is_monotonic, grad_monotonic_type, total_delta_val, min_data, max_data, abs_total_delta_val, total_delta_sign, edge_idx, curve_top_type, abs_grad_delta_ratio, is_strict_monotonic, delta_1, delta_2, curve_side, abs_delta_1, abs_delta_2, total_abs_delta, pixel_monotonic_direction, vibrate_direction]


def get_seg_intensity_upper_bounded_points(seg, seg_start_point, intensity_upper_bound):
    seg_len = len(seg)
    point_list = []
    i = 0
    while i < seg_len:
        if seg[i] <= intensity_upper_bound:
            point_list.append(i)
        i += 1
    intensity_upper_bounded_points = np.array(point_list)+seg_start_point
    return intensity_upper_bounded_points


def filter_grad_signed_seg_points_simple(grad_sign_list, point_list):
    list_len = len(point_list)
    local_sign_list = []
    local_point_list = []
    filtered_point_list = []
    i = 0
    while i < list_len:
        current_sign = grad_sign_list[i]
        if current_sign == -1:
            local_sign_list.append(-1)
            local_point_list.append(point_list[i])
        else:
            if len(local_sign_list) > 0:
                local_sign_list.pop()
                prev_neg_point = local_point_list.pop()
                filtered_point_list.append(prev_neg_point)
                filtered_point_list.append(point_list[i])
        i += 1
    return filtered_point_list


def exponential_distribution_fun(x_range, distb_param, step_div=10, is_param_avg=True, is_log=True):
    param_lambda = distb_param
    if is_param_avg:
        param_lambda = 1/param_lambda
    x_range_start = x_range[0]
    x_range_end = x_range[1]
    step = (x_range_end - x_range_start)/step_div
    x_point = x_range_start
    x = []
    while x_point <= x_range_end:
        x.append(x_point)
        x_point += step
    x = np.array(x)
    y = param_lambda*np.exp(-param_lambda*x)
    y_min = np.min(y)
    if np.max(y) != y_min:
        insertion_value = y_min
    else:
        insertion_value = 0
    y = np.where(y == 0, insertion_value, y)
    if is_log:
        y = np.log(y)
    return x, y


def linear_regression_1d_data(x, y):
    model = LinearRegression()
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    model.fit(x, y)
    a = model.intercept_
    b = model.coef_
    r = model.score(x, y)
    return a, b, r


def nd_array_transpose_concatenate_1d(array_x, array_y):
    array_shape = array_x.shape
    x_2d = np.reshape(array_x, (1, array_shape[0]))
    y_2d = np.reshape(array_y, (1, array_shape[0]))
    xy = np.concatenate((x_2d, y_2d))
    xy = xy.T
    return xy


def nd_array_transpose_concatenate_1d_list(array_list):
    expanded_array_list = nd_array_dim_expand_list(array_list)
    concatenated_array = np.concatenate(expanded_array_list)
    concatenated_array = concatenated_array.T
    return concatenated_array


def nd_array_dim_expand_list(array_list):
    list_len = len(array_list)
    i = 0
    while i < list_len:
        nd_array = array_list[i]
        array_list[i] = np.reshape(nd_array, (1, nd_array.shape[0]))
        i += 1
    return array_list


def nd_array_segmentation_list_1d(array_list, col_start, col_len):
    list_len = len(array_list)
    i = 0
    segmented_array_list = []*list_len
    while i < list_len:
        nd_array = array_list[i]
        nd_array = nd_array[col_start:col_len]
        segmented_array_list.append(nd_array)
        i += 1
    return segmented_array_list


def range_factorial(integer_range):
    range_low = integer_range[0]
    range_high = integer_range[1]
    range_list = range(range_low, range_high+1)
    range_list_len = len(range_list)
    factorial_result_list = []
    range_factorial_result = math.factorial(range_low)
    factorial_result_list.append(range_factorial_result)
    i = 1
    while i < range_list_len:
        range_factorial_result = range_factorial_result * range_list[i]
        factorial_result_list.append(range_factorial_result)
        i += 1
    factorial_result_list = np.array(factorial_result_list)
    range_list = np.array(range_list)
    return factorial_result_list, range_list


def poisson_dist_func(integer_range, lambda_param, range_power_ratio=RANGE_POWER_RATIO):
    factorial_result_list, range_list = range_factorial(integer_range)
    power_list_with_ratio = range_list * range_power_ratio
    poisson_dist_result = np.power(lambda_param, power_list_with_ratio)
    poisson_dist_result = poisson_dist_result * np.exp(-lambda_param) / factorial_result_list
    poisson_dist_result = poisson_dist_result * np.exp(-lambda_param)
    return range_list, poisson_dist_result


def exp_exp_sub_1():
    res = np.exp(1)
    res = res/(res - 1)
    return res
