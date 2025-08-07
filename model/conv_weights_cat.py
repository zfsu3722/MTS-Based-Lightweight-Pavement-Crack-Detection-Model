import os
import torch
from pre_networks import AutoEncoder
import torch.nn as nn
import cv2
import numpy as np

import crack500_support as cr5_spt
import torch
from convert_tensor import train_source_tensor
from convert_tensor import test_source_tensor
from convert_tensor import train_image_names
from convert_tensor import test_image_names

train_target_file = "/home/stu1/crack_test/crack_fused/crack_python21/datasets/train_lab"
test_target_file = "/home/stu1/crack_test/crack_fused/crack_python21/datasets/test_lab"
train_target_paths = cr5_spt.load_images(train_target_file)
train_target = cr5_spt.load_images_as_tensors(train_target_paths)
test_target_paths = cr5_spt.load_images(test_target_file)
test_target = cr5_spt.load_images_as_tensors(test_target_paths)
print("train_target:", len(train_target))
print("test_target:", len(test_target))
test_label_names = cr5_spt.get_img_name(test_target_paths)

model = AutoEncoder()
encoder = model.encoder

checkpoint_path = 'pretrained_model35_hardtanh01_layer4_k3_seg8_concat_checkpoint_cuda1.pth'
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)

    encoder.load_state_dict(checkpoint['model_state_dict'])

encoder.eval()

activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output
    return hook

for layer_name, layer in encoder.named_children():
    layer.register_forward_hook(get_activation(layer_name))

def activate_img(sample_input, target_names, group_size=8):
    group_activations = []
    #group_name_list = []
    concat_img = []
    for i in range(0, len(sample_input), group_size):
        group_input = sample_input[i:i + group_size]
        group_name = target_names[i:i + group_size]

        group_activation = 0

        for input_image, name in zip(group_input, group_name):
            input_image = input_image.unsqueeze(0).unsqueeze(0)

            with torch.no_grad():

            layer_activation = activation['tanh4']

            layer_activation = layer_activation

            group_activations.append(layer_activation)

        activation_tensor_long = len(group_activations)

        tensor_out = group_activations[0]
        i = 1
        while i < activation_tensor_long:
            tensor_activation = group_activations[i]
            tensor_out = torch.cat((tensor_out, tensor_activation), dim=1)
            i += 1
            # if i == activation_tensor_long:
        group_activations = []
        concat_img.append(tensor_out)
        activation.clear()

    print(len(concat_img))
    return concat_img

test_activations_list = activate_img(test_source_tensor, test_image_names, group_size=8)

print("test_activations_list_length:", len(test_activations_list))

length = len(test_activations_list)

split_index = np.ceil(length // 10)
print("split_index_len", split_index)

first_half = test_activations_list[:int(split_index)]
second_half = test_activations_list[int(split_index):int(2*split_index)]


third_half = test_activations_list[int(2*split_index):int(3*split_index)]
fourth_half = test_activations_list[int(3*split_index):int(4*split_index)]
fifth_half = test_activations_list[int(4*split_index):int(5*split_index)] 
sixth_half = test_activations_list[int(5*split_index):int(6*split_index)]
seventh_half = test_activations_list[int(6*split_index):int(7*split_index)]
eighth_half = test_activations_list[int(7*split_index):int(8*split_index)]
ninth_half = test_activations_list[int(8*split_index):int(9*split_index)]
tenth_half = test_activations_list[int(9*split_index):]


print("First Half:", len(first_half))
print("Second Half:", len(second_half))
print("Third Half:", len(third_half))
print("Fourth Half:", len(fourth_half))
print("Fifth Half:", len(fifth_half))
print("Sixth Half:", len(sixth_half))
print("Seventh Half:", len(seventh_half))
print("Eightth Half:", len(eighth_half))
print("Ninth Half:", len(ninth_half))
print("Tenth Half:", len(tenth_half))


test_file_path1 = "test_activations_data35all_seg8_sort1.pkl"
test_file_path2 = "test_activations_data35all_seg8_sort2.pkl"
test_file_path3 = "test_activations_data35all_seg8_sort3.pkl"
test_file_path4 = "test_activations_data35all_seg8_sort4.pkl"
test_file_path5 = "test_activations_data35all_seg8_sort5.pkl"
test_file_path6 = "test_activations_data35all_seg8_sort6.pkl"
test_file_path7 = "test_activations_data35all_seg8_sort7.pkl"
test_file_path8 = "test_activations_data35all_seg8_sort8.pkl"
test_file_path9 = "test_activations_data35all_seg8_sort9.pkl"
test_file_path10 = "test_activations_data35all_seg8_sort10.pkl"
cr5_spt.save_data_to_pickle1(first_half, test_file_path1)
cr5_spt.save_data_to_pickle1(second_half, test_file_path2)
cr5_spt.save_data_to_pickle1(third_half, test_file_path3)
cr5_spt.save_data_to_pickle1(fourth_half, test_file_path4)
cr5_spt.save_data_to_pickle1(fifth_half, test_file_path5)
cr5_spt.save_data_to_pickle1(sixth_half, test_file_path6)
cr5_spt.save_data_to_pickle1(seventh_half, test_file_path7)
cr5_spt.save_data_to_pickle1(eighth_half, test_file_path8)
cr5_spt.save_data_to_pickle1(ninth_half, test_file_path9)
cr5_spt.save_data_to_pickle1(tenth_half, test_file_path10)



