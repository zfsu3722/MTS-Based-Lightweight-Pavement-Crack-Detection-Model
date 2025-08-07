import pickle as pik
import time
import torch.nn as nn
import torch.optim as optim
from decoder_networks import Decoder_train #
from pre_networks import AutoEncoder #
import torch
import os
import cv2
import numpy as np
import crack500_support as cr5_spt #
from prf_metrics import cal_prf_metrics #
from segment_metrics import cal_semantic_metrics #
from convert_tensor import train_source_tensor #
from convert_tensor import test_source_tensor #
from convert_tensor import train_image_names #
from convert_tensor import test_image_names #
import torch
from PIL import Image
import matplotlib.pyplot as plt


DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu' #
checkpoint_path = '48.pth'
train_target_file = "/home/stu1/crack_test/crack_fused/crack_python21/datasets/train_lab"
test_target_file = "/home/stu1/crack_test/crack_fused/crack_python21/datasets/test_lab"
num_epochs = 500
metric_mode = 'prf'
threshold = 0.0

EPOCH_CHECKPOINT_DIR = "decoder_checkpoints_each_epoch"
os.makedirs(EPOCH_CHECKPOINT_DIR, exist_ok=True) #

model = AutoEncoder() #
encoder = model.encoder.to(DEVICE) #

if os.path.exists(checkpoint_path): #
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    encoder.load_state_dict(checkpoint['model_state_dict'])
    print(f"Checkpoint file found at: {checkpoint_path}")

else:
    print(f"Checkpoint file not found at: {checkpoint_path}") #
    exit()

activation = {}

def get_activation(name): #
    def hook(module, input, output):
        activation[name] = output.clone().detach() #
    return hook

ENCODER_ACTIVATION_LAYER = 'tanh4'
target_layer_found = False
for layer_name, layer_module in encoder.named_modules(): #
    if layer_name == ENCODER_ACTIVATION_LAYER:
        layer_module.register_forward_hook(get_activation(ENCODER_ACTIVATION_LAYER)) #
        print(f"Forward hook registered at encoder level： {ENCODER_ACTIVATION_LAYER}")
        target_layer_found = True
        break
if not target_layer_found:
     print(f"Warning: No encoder layer found for activation hook '{ENCODER_ACTIVATION_LAYER}'。")

train_target_paths = cr5_spt.load_images(train_target_file) #
test_target_paths = cr5_spt.load_images(test_target_file) #
train_target = cr5_spt.load_images_as_tensors(train_target_paths) #
test_target = cr5_spt.load_images_as_tensors(test_target_paths) #
print("train_target:", len(train_target)) #
print("test_target:", len(test_target)) #
train_label_names = cr5_spt.get_img_name(train_target_paths) #
test_label_names = cr5_spt.get_img_name(test_target_paths) #

decoder = Decoder_train() #
decoder = decoder.to(DEVICE) #
optimizer = torch.optim.Adam(decoder.parameters(), lr=0.001) #

bce_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.0 / 3e-2).to(DEVICE)) #

class SoftDiceLoss(nn.Module):
    def __init__(self, variable=2, smooth=1., dims=(-2, -1)):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth
        self.dims = dims
        self.variable = variable
    def forward(self, x, y):
        y = y.float()
        tp = (x * y).sum(self.dims)
        fp = (x * (1 - y)).sum(self.dims)
        fn = ((1 - x) * y).sum(self.dims)
        dc = (self.variable * tp + self.smooth) / (self.variable * tp + fp + fn + self.smooth)
        dc = dc.mean()
        return 1 - dc

dice_fn = SoftDiceLoss()

def loss_fn(y_pred, y_true):
    bce = bce_fn(y_pred, y_true)
    dice = dice_fn(y_pred.sigmoid(), y_true)
    return bce + 0.2 * dice


best_f1 = 0
best_epoch = 0
best_pred_list = []
best_test_label_list = []
t_list = []
f_list = []

for epoch in range(1, num_epochs + 1):
    print("epoch", epoch, "started")
    start_time = time.time()
    decoder.train()
    tlosses_epoch = []

    for i, label_tensor in enumerate(train_target):
        label = label_tensor.float().unsqueeze(0).unsqueeze(0).to(DEVICE)
        group_size = 8

        group_activations = []
        group_start_index = i * group_size

        with torch.no_grad(): #
            for j in range(group_start_index, min(group_start_index + group_size, len(train_source_tensor))): #
                 if j >= len(train_source_tensor): break
                 input_image = train_source_tensor[j].float().unsqueeze(0).unsqueeze(0).to(DEVICE) #
                 output1 = encoder(input_image) #
                 if ENCODER_ACTIVATION_LAYER not in activation:
                      print(f"Error(training): Activation layer '{ENCODER_ACTIVATION_LAYER}' not found at example {i}, index {j}.")
                      group_activations = []
                      break
                 layer_activation = activation[ENCODER_ACTIVATION_LAYER]
                 group_activations.append(layer_activation)
                 activation.clear()

        if not group_activations: continue

        try:
            activation_tensor_long = len(group_activations)
            if activation_tensor_long == 0 : continue
            tensor_out = torch.cat(group_activations, dim=1)
        except RuntimeError as e:
            print(f"ERROR(training): Error concatenating activations for example {i}: {e}")
            continue

        optimizer.zero_grad()
        train_pred_image = decoder(tensor_out)

        if label.size(3) != train_pred_image.size(3):
            label = label.transpose(3, 2)
        try:
             tloss = loss_fn(train_pred_image, label) #
        except Exception as e:
             print(f"ERROR(training): Error computing loss for example {i}: {e}")
             print(f"  Prediction shape: {train_pred_image.shape}, label shape: {label.shape}")
             continue

        tloss.backward() #
        optimizer.step() #
        tlosses_epoch.append(tloss.item())

    tloss_avg = np.mean(tlosses_epoch) if tlosses_epoch else 0
    print(f'Epoch [{epoch}/{num_epochs}], tLoss: {tloss_avg:.6f}')

    end_time = time.time()
    epoch_duration = end_time - start_time
    t_list.append(epoch_duration)
    print(f'Epoch duration: {epoch_duration:.4f} seconds')

    with torch.no_grad():
        pred_list = []
        test_label_list = []
        vlosses_epoch = []
        val_start_time = time.time()
        decoder.eval()
        total_time_val = 0.0
        num_frames = 0

        for i, test_label_tensor in enumerate(test_target):
            test_label = test_label_tensor.float().unsqueeze(0).unsqueeze(0).to(DEVICE)
            test_label_name = test_label_names[i]
            group_size = 8
            test_group_activations = []
            group_start_index = i * group_size

            for j in range(group_start_index, min(group_start_index + group_size, len(test_source_tensor))): #
                if j >= len(test_source_tensor): break
                test_input_image = test_source_tensor[j].float().unsqueeze(0).unsqueeze(0).to(DEVICE)
                output2 = encoder(test_input_image) #
                if ENCODER_ACTIVATION_LAYER not in activation:
                    print(f"ERROR(validation): Activation layer '{ENCODER_ACTIVATION_LAYER}' not found at sample {i}, index {j}。")
                    test_group_activations = []
                    break
                test_layer_activation = activation[ENCODER_ACTIVATION_LAYER]
                test_group_activations.append(test_layer_activation)
                activation.clear()

            if not test_group_activations: continue


            try:
                activation_tensor_long = len(test_group_activations) #
                if activation_tensor_long == 0: continue
                test_tensor_out = torch.cat(test_group_activations, dim=1) #
            except RuntimeError as e:
                 print(f"Error (validation): Error in activating splice of sample {i}: {e}")
                 continue

            output = decoder(test_tensor_out)


            if test_label.size(3) != output.size(3):
                test_label = test_label.transpose(3, 2)

            try:
                 vloss = loss_fn(output, test_label) #
            except Exception as e:
                 print(f"ERROR(validation): Error computing loss for sample {i}: {e}")
                 print(f"  Prediction shape: {output.shape}, label shape: {test_label.shape}")
                 continue
            num_frames += 1
            vlosses_epoch.append(vloss.item())

            pred_binary = (output > threshold).float()
            pred_list.append(pred_binary)
            test_label_list.append(test_label)


        vloss_avg = np.mean(vlosses_epoch) if vlosses_epoch else 0
        print(f'Epoch [{epoch}/{num_epochs}], vLoss: {vloss_avg:.6f}')
        val_end_time = time.time()
        total_time_val = val_end_time - val_start_time
        fps = num_frames / total_time_val if total_time_val > 0 else 0

        f_list.append(fps) #
        print(f'Total frames processed: {num_frames}') #
        print(f'Total time taken (validation): {total_time_val:.5f} seconds')
        print(f'FPS (validation): {fps:.5f}') #


        f1 = 0.0
        if pred_list and test_label_list:
            if metric_mode == 'prf':
                 try:
                     final_results = cal_prf_metrics(pred_list, test_label_list, epoch)
                     final_results1 = cal_semantic_metrics(pred_list, test_label_list, epoch, num_cls=2)
                     print(final_results)
                     if final_results:
                         f1 = final_results[0][3]
                     else:
                          print("Warning: PRF indicator calculation returns empty。")
                          f1 = 0.0
                 except Exception as e:
                      print(f"ERROR: Error calculating indicator: {e}")
                      f1 = 0.0

                 if f1 > best_f1:
                     best_f1 = f1
                     best_pred_list = pred_list
                     best_test_label_list = test_label_list
                     best_epoch = epoch
                     with open('best_epoch1.txt', 'w') as f:
                         f.write(str(best_epoch))
                     print(f"*** New best F1 score： {best_f1:.6f} 在 epoch {best_epoch}。 ***")

                     output_folder21 = 'output_folder21'
                     if not os.path.exists(output_folder21):
                         os.makedirs(output_folder21)
                     output_folder22 = 'output_folder22'
                     if not os.path.exists(output_folder22):
                         os.makedirs(output_folder22)


                     for idx, (pred_b, test_l, test_ln) in enumerate(zip(best_pred_list, best_test_label_list, test_label_names)): #
                         if idx < len(test_label_names):
                             filename = test_ln
                             output_filename_pred = os.path.join(output_folder21, f'best_F1_{filename}.png')
                             output_filename_gt = os.path.join(output_folder22, f'best_F1_{filename}.png')
                             pred_binary_array = (pred_b.squeeze().squeeze().cpu().numpy() * 255).astype('uint8')
                             test_label_array = (test_l.squeeze().squeeze().cpu().numpy() * 255).astype('uint8')

                             cv2.imwrite(output_filename_pred, pred_binary_array) #
                             cv2.imwrite(output_filename_gt, test_label_array) #
                         else:
                              print(f"Warning (saving best image): Index {idx} is out of range for test_label_names。")

            else:
                print("Unknown mode of metrics.") #
        else:
             print("Warning: Validation prediction/label list is empty, unable to calculate metrics。")
             f1 = 0.0


    epoch_checkpoint = {
        'model_state_dict': decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'last_train_loss': tloss_avg if 'tloss_avg' in locals() else None,
        'last_val_loss': vloss_avg if 'vloss_avg' in locals() else None,
        'last_val_f1': f1
    }

    epoch_chkpt_filename = os.path.join(EPOCH_CHECKPOINT_DIR, f'decoder_epoch_{epoch}.pth')

    torch.save(epoch_checkpoint, epoch_chkpt_filename)
    print(f"Epoch {epoch} checkpoint saved to: {epoch_chkpt_filename}")

t_avg = np.mean(t_list) if t_list else 0
f_avg = np.mean(f_list) if f_list else 0
print("Average epoch training duration:", t_avg)
print("Average validation FPS:", f_avg)

with open('best_epoch.txt', 'w') as f:
    f.write(str(best_epoch))
print(f"Final best epoch number ({best_epoch}) saved to best_epoch.txt")
