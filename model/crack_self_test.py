import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import torch
from PIL import Image
import crack500_support as cr5_spt
import torchvision.transforms as transforms

cr5_spt.DISTANCE_SEG_NUM = 8  # 25
plane_kept_ratio = 0  # 0.005#0.005
input_type = 0
fragment_shape = (10, 10)

#dataset = "/home/stu1/crack_test/crack_fused/crack_python2/datasets/img_all"
#label = "/home/stu1/crack_test/crack_fused/crack_python2/datasets/lab_all"
#save_folder_fused = "/home/stu1/crack_test/crack_fused/crack_python2/datasets/select_fused"
#save_folder_plane = "/home/stu1/crack_test/crack_fused/crack_python2/datasets/select_plane"
#img_plane_all_folder = "/home/stu1/crack_test/crack_fused/crack_python2/datasets/img_gray_all_plane"
#fused_all_folder = "/home/stu1/crack_test/crack_fused/crack_python2/datasets/fused_all"
#img_gray_all_folder = "/home/stu1/crack_test/crack_fused/crack_python2/datasets/img_gray_all"
train_dataset = "/home/stu1/crack_test/crack_fused/crack_python21/datasets/train_img"
train_label = "/home/stu1/crack_test/crack_fused/crack_python21/datasets/train_lab"
test_dataset = "/home/stu1/crack_test/crack_fused/crack_python21/datasets/test_img"
test_label = "/home/stu1/crack_test/crack_fused/crack_python21/datasets/test_lab"
train_img_plane_folder = "/home/stu1/crack_test/crack_fused/crack_python21/datasets/train_plane8"
train_img_gray_folder ="/home/stu1/crack_test/crack_fused/crack_python21/datasets/train_gray"
test_img_plane_folder = "/home/stu1/crack_test/crack_fused/crack_python21/datasets/test_plane8"
test_img_gray_folder = "/home/stu1/crack_test/crack_fused/crack_python21/datasets/test_gray"
# save_folder_gray = "D:/PycharmProjects/test/crack_python01/datasets/img_gray"
# save_folder_original = "D:/PycharmProjects/test/crack_python01/datasets/img_gray_original"
# save_train_plane = "D:/PycharmProjects/test/crack_python01/datasets/train_img_gray_plane"
# save_test_plane = "D:/PycharmProjects/test/crack_python01/datasets/test_img_gray_plane"


# image_paths = cr5_spt.load_images(dataset)
# label_paths = cr5_spt.load_images(label)
train_image_paths = cr5_spt.load_images(train_dataset)
train_label_paths = cr5_spt.load_images(train_label)
test_image_paths = cr5_spt.load_images(test_dataset)
test_label_paths = cr5_spt.load_images(test_label)

train_img_gray_plane_list_all, train_img_gray_list = cr5_spt.process_images_and_save_planes(train_image_paths, train_img_plane_folder, train_img_gray_folder,  plane_kept_ratio=0)
test_img_gray_plane_list_all, test_img_gray_list = cr5_spt.process_images_and_save_planes(test_image_paths, test_img_plane_folder, test_img_gray_folder,  plane_kept_ratio=0)
# img_gray_plane_list_all, img_gray_list = cr5_spt.process_images_and_save_planes(image_paths, img_plane_all_folder, img_gray_all_folder,  plane_kept_ratio=0)
print('image loaded')
#train_image_gray_paths = cr5_spt.load_images(train_img_gray_folder)
#train_image_plane_paths = cr5_spt.load_images(train_img_plane_folder)
#test_image_gray_paths = cr5_spt.load_images(test_img_gray_folder)
#test_image_plane_paths = cr5_spt.load_images(test_img_plane_folder)


#train_source_tensor = cr5_spt.load_images_as_tensors(train_image_plane_paths)
#train_target_tensor = cr5_spt.load_images_as_tensors(train_image_plane_paths)
#test_source_tensor = cr5_spt.load_images_as_tensors(test_image_plane_paths)
#test_target_tensor = cr5_spt.load_images_as_tensors(test_image_plane_paths)

#train_image_names = cr5_spt.get_img_name(train_image_plane_paths)
#train_target_names = cr5_spt.get_img_name(train_image_plane_paths)
#test_image_names = cr5_spt.get_img_name(test_image_plane_paths)
#test_target_names = cr5_spt.get_img_name(test_image_plane_paths)
#print(len(train_source_tensor))
#print(len(test_source_tensor))
