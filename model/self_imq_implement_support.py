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
import phasepack as pc
from scipy.signal import convolve2d


def img_yiq_convert(img):
    img_shape = img.shape
    img_row_num = img_shape[0]
    img_col_num = img_shape[1]
    I = np.ones((img_row_num, img_col_num))
    Q = np.ones((img_row_num, img_col_num))
    if img.ndim == 2:
        Y = img
    else:
        img_plane_0 = img[:, :, 0]
        img_plane_1 = img[:, :, 1]
        img_plane_2 = img[:, :, 2]
        Y = 0.299 * img_plane_0 + 0.587 * img_plane_1 + 0.114 * img_plane_2
        I = 0.596 * img_plane_0 - 0.274 * img_plane_1 - 0.322 * img_plane_2
        Q = 0.211 * img_plane_0 - 0.523 * img_plane_1 + 0.312 * img_plane_2
    return Y, I, Q


def img_fsim_cal_convert(img_ref, img_dist):
    Y1, I1, Q1 = img_yiq_convert(img_ref)
    Y2, I2, Q2 = img_yiq_convert(img_dist)
    return I1, Y1, Q1, I2, Y2, Q2


def gradient_map(Y):
    dx = np.array([[3, 0, - 3], [10, 0, - 10], [3, 0, - 3]])
    dy = np.array([[3, 10, 3], [0, 0, 0], [-3, -10, -3]])
    dx = dx / 16
    dy = dy / 16
    IxY = convolve2d(Y, dx, mode='same')
    IyY = convolve2d(Y, dy, 'same');
    gradientMap = np.sqrt(np.power(IxY, 2) + np.power(IyY, 2))
    return gradientMap


def fsim(img_ref, img_dist, T1=0.85, T2=160, T3 = 200, T4 = 200,is_output_fsimc=True):
    I1, Y1, Q1, I2, Y2, Q2 = img_fsim_cal_convert(img_ref, img_dist)
    M1, m1, ori1, ft1, PC1, EO1, T_1 = pc.phasecong(Y1, nscale=4, norient=8, minWaveLength=12, mult=2, sigmaOnf=0.55)
    M2, m2, ori2, ft2, PC2, EO2, T_2 = pc.phasecong(Y2, nscale=4, norient=8, minWaveLength=12, mult=2, sigmaOnf=0.55)
    PC1 = np.sum(np.array(PC1), 0)
    PC2 = np.sum(np.array(PC2), 0)
    gradientMap1 = gradient_map(Y1)
    gradientMap2 = gradient_map(Y2)
    #plt.subplot(1, 2, 1)
    #plt.imshow(gradientMap1, cmap='gray')
    #plt.subplot(1, 2, 2)
    #plt.imshow(gradientMap2, cmap='gray')
    #plt.show()
    PCSimMatrix = (2 * PC1 * PC2 + T1) / (np.power(PC1, 2) + np.power(PC2, 2) + T1)
    gradientSimMatrix = (2 * gradientMap1 * gradientMap2 + T2) / (np.power(gradientMap1, 2) + np.power(gradientMap2, 2) + T2)
    #
    #print("FSIM_gradient_mean:", np.mean(gradientSimMatrix))
    #
    PCm = np.maximum(PC1, PC2)
    SimMatrix = gradientSimMatrix * PCSimMatrix * PCm
    FSIM = np.sum(SimMatrix) / np.sum(PCm)
    ISimMatrix = (2 * I1 * I2 + T3) / (np.power(I1, 2) + np.power(I2, 2) + T3)
    QSimMatrix = (2 * Q1 * Q2 + T4) / (np.power(Q1, 2) + np.power(Q2, 2) + T4)
    p_lambda = 0.003
    SimMatrixC = gradientSimMatrix * PCSimMatrix * np.real(np.power((np.complex_(ISimMatrix) * np.complex_(QSimMatrix)), p_lambda)) * PCm
    FSIMc = np.sum(SimMatrixC) / np.sum(PCm)
    if is_output_fsimc:
        output = FSIMc
    else:
        output = FSIM
    return output
