import os
import torch
import torch.nn as nn
import cv2
import numpy as np
import crack500_support as cr5_spt
from decoder_networks import Decoder_train
from pre_networks import AutoEncoder
from prf_metrics import cal_prf_metrics
from segment_metrics import cal_semantic_metrics


DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
ENCODER_PATH = '48.pth'
DECODER_PATH = 'decoder_epoch_187.pth'
TEST_PLANE_FOLDER = "/home/stu1/crack_test/crack_fused/crack_python21/datasets/test_plane8"
TEST_LABEL_FOLDER = "/home/stu1/crack_test/crack_fused/crack_python21/datasets/test_lab"
OUTPUT_DIR = "generalization_results_dc"
os.makedirs(OUTPUT_DIR, exist_ok=True)
THRESHOLD = 0.0
ENCODER_ACTIVATION_LAYER = 'tanh4'  #
print("--- Initializing and loading weights ---")
model = AutoEncoder()
encoder = model.encoder.to(DEVICE)
decoder = Decoder_train().to(DEVICE)
encoder_ckpt = torch.load(ENCODER_PATH, map_location=DEVICE)
encoder.load_state_dict(encoder_ckpt['model_state_dict'])
decoder_ckpt = torch.load(DECODER_PATH, map_location=DEVICE)
decoder.load_state_dict(decoder_ckpt['model_state_dict'])
decoder.eval()
activation = {}


def get_activation(name):
    def hook(module, input, output):
        activation[name] = output.clone().detach()
    return hook
for name, module in encoder.named_modules():
    if name == ENCODER_ACTIVATION_LAYER:
        module.register_forward_hook(get_activation(ENCODER_ACTIVATION_LAYER))
test_plane_paths = cr5_spt.load_images(TEST_PLANE_FOLDER)
test_source_tensor = cr5_spt.load_images_as_tensors(test_plane_paths)
test_label_paths = cr5_spt.load_images(TEST_LABEL_FOLDER)
test_target = cr5_spt.load_images_as_tensors(test_label_paths)
test_label_names = cr5_spt.get_img_name(test_label_paths)
pred_list = []
test_label_list = []
print("--- Starting the inference process ---")
with torch.no_grad():
    for i, test_label_tensor in enumerate(test_target):
        label = test_label_tensor.float().unsqueeze(0).unsqueeze(0).to(DEVICE)
        test_group_activations = []
        group_start_idx = i * 8
        for j in range(group_start_idx, min(group_start_idx + 8, len(test_source_tensor))):
            input_img = test_source_tensor[j].float().unsqueeze(0).unsqueeze(0).to(DEVICE)
            _ = encoder(input_img)
            test_group_activations.append(activation[ENCODER_ACTIVATION_LAYER])
            activation.clear()

        if len(test_group_activations) < 8: continue

        test_tensor_out = torch.cat(test_group_activations, dim=1)
        output = decoder(test_tensor_out)
        if label.size(3) != output.size(3):
            label = label.transpose(3, 2)
        pred_binary = (output > THRESHOLD).float()
        pred_list.append(pred_binary)
        test_label_list.append(label)
        name = test_label_names[i]
        pred_img = (pred_binary.squeeze().cpu().numpy() * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}.png"), pred_img)
if pred_list:
    cal_prf_metrics(pred_list, test_label_list, 1)
    cal_semantic_metrics(pred_list, test_label_list, 1, num_cls=2)
