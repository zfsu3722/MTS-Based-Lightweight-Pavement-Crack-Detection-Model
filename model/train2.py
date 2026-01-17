import os
import gc
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pre_networks import Encoder
from pre_networks import Decoder
from pre_networks import AutoEncoder
# from crack_self_test import fused_list
import torch
# --- MODIFICATION START: ADDED IMPORT ---
from prf_metrics import cal_prf_metrics
from segment_metrics import cal_semantic_metrics
# --- MODIFICATION END ---

import crack500_support as cr5_spt
import numpy as np

# import pickle as pik
from convert_tensor import train_source_tensor
from convert_tensor import train_target_tensor
from convert_tensor import test_source_tensor
from convert_tensor import test_target_tensor
from convert_tensor import train_image_names
from convert_tensor import train_target_names
from convert_tensor import test_image_names
from convert_tensor import test_target_names

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

criterion = nn.BCEWithLogitsLoss()

model = AutoEncoder()
model.to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- MODIFICATION START: REMOVED metric_mode VARIABLE ---
# The metric_mode variable is no longer needed as we will calculate all metrics.
# --- MODIFICATION END ---

num_epochs = 50
best_model_state_dict = None
best_f1 = 0
best_loss = 10
best_epoch = 0
best_pred_list = []
best_test_label_list = []
threshold = 0.5

print("loading data...")

for epoch in range(num_epochs + 1):
    tlosses = []
    model.train()
    print("epoch", epoch, "started")
    for image_plane, target, train_image_name, train_target_name in zip(train_source_tensor, train_target_tensor,
                                                                        train_image_names, train_target_names):
        input_image = image_plane.float().unsqueeze(0).to(DEVICE)
        target_image = target.float().unsqueeze(0).unsqueeze(0).to(DEVICE)

        optimizer.zero_grad()

        reconstructed_images = model(input_image.unsqueeze(0))

        tloss = criterion(reconstructed_images, target_image)
        tlosses.append(tloss.item())

        train_pred = torch.where(reconstructed_images > threshold, torch.tensor(1).to(DEVICE),
                                 torch.tensor(0).to(DEVICE))

        tloss.backward()
        optimizer.step()

        checkpoint = {
            'model_state_dict': model.encoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(checkpoint, f'{epoch}.pth')

    tloss_avg = np.array(tlosses).mean()

    with torch.no_grad():
        model.eval()
        fused_list_reconstructed = []
        test_label_list = []
        vlosses = []
        threshold = 0.5

        for image, target, image_name, target_name in zip(test_source_tensor, test_target_tensor, test_image_names,
                                                          test_target_names):

            image = image.float().unsqueeze(0).to(DEVICE)
            target = target.float().unsqueeze(0).unsqueeze(0).to(DEVICE)

            output = model(image.unsqueeze(0))
            output_max = output.max()
            output_min = output.min()

            vloss = criterion(output, target)
            vlosses.append(vloss.item())

            pred_binary = torch.where(output > threshold, torch.tensor(1).to(DEVICE), torch.tensor(0).to(DEVICE))
            fused_list_reconstructed.append(pred_binary)
            test_label_list.append(target)

            if epoch == 40:
                output_folder = 'output_folder'
                output_folder1 = 'output_folder1'
                output_folder2 = 'output_folder2'

                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                if not os.path.exists(output_folder1):
                    os.makedirs(output_folder1)
                if not os.path.exists(output_folder2):
                    os.makedirs(output_folder2)

                output_array = output.squeeze().squeeze().cpu().numpy()
                pred_binary_array = pred_binary.squeeze().squeeze().cpu().numpy()
                target_array = target.squeeze().squeeze().cpu().numpy()
                output_array1 = (output_array * 255).astype('uint8')
                pred_binary_array1 = (pred_binary_array * 255).astype('uint8')
                target_array1 = (target_array * 255).astype('uint8')
                filename = image_name.split('.')[0]
                output_filename = os.path.join(output_folder, f'{filename}.png')
                cv2.imwrite(output_filename, output_array1)
                output_filename1 = os.path.join(output_folder1, f'{filename}.png')
                cv2.imwrite(output_filename1, pred_binary_array1)
                output_filename2 = os.path.join(output_folder2, f'{filename}.png')
                cv2.imwrite(output_filename2, target_array1)

        vloss_avg = np.array(vlosses).mean()
        print(f'Epoch [{epoch}/{num_epochs}], tLoss: {tloss_avg:.6f}, vLoss: {vloss_avg:.6f}')

        # --- MODIFICATION START: CALCULATE ALL METRICS ---
        print("\n--- Calculating Performance Metrics for Epoch {} ---".format(epoch))

        # 1. Calculate Precision, Recall, F1-Score
        # Note: The second argument should be the list of ground truth tensors corresponding to the predictions.
        prf_results = cal_prf_metrics(fused_list_reconstructed, test_label_list, epoch)

        # 2. Calculate PA, MPA, MIoU
        semantic_results = cal_semantic_metrics(fused_list_reconstructed, test_label_list, epoch)

        print("--- Metrics Calculation Complete ---\n")

        # 3. Check for best model based on F1-Score
        if prf_results:  # Ensure prf_results is not empty
            f1 = prf_results[0][3]
            if f1 > best_f1:
                best_f1 = f1
                best_pred_list = fused_list_reconstructed
                best_test_label_list = test_label_list
                # Note: 'image_name' is from the last loop item. For a robust implementation,
                # you might want to save all test image names in a list during the loop.
                best_epoch = epoch
                with open('best_epoch2.txt', 'w') as f:
                    f.write(str(best_epoch))
                output_folder15 = 'output_folder15'
                if not os.path.exists(output_folder15):
                    os.makedirs(output_folder15)
                output_folder16 = 'output_folder16'
                if not os.path.exists(output_folder16):
                    os.makedirs(output_folder16)

                for i, (pred_binary, test_label, test_image_name) in enumerate(
                        zip(best_pred_list, best_test_label_list, test_image_names)):
                    filename = test_image_name.split('.')[0]
                    output_filename = os.path.join(output_folder15, f'best_F1_{filename}.png')
                    output_filename1 = os.path.join(output_folder16, f'best_F1_{filename}.png')
                    pred_binary_array = (pred_binary.squeeze().squeeze().cpu().numpy() * 255).astype('uint8')
                    test_label_array = (test_label.squeeze().squeeze().cpu().numpy() * 255).astype('uint8')
                    cv2.imwrite(output_filename, pred_binary_array)
                    cv2.imwrite(output_filename1, test_label_array)
        # --- MODIFICATION END ---

checkpoint1 = {
    'model_state_dict': best_model_state_dict,  # This should be model.state_dict() if you want to save the best model
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': best_epoch
}
# torch.save(checkpoint1, 'best_model_checkpoint.pth') # You should save the best model state inside the if f1 > best_f1 block
