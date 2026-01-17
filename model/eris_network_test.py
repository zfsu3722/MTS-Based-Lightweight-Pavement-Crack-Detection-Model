import eris_network as eris
import numpy as np
import crack500_support as cr5_spt
import crack_self_proc as cr_sf_proc
import matplotlib
import matplotlib.pyplot as plt
import time as tm
import torch

matplotlib.use('Qt5Agg')
#print(plt.get_backend())

self_img_num = 663#1264#2313#2295#1214
frame_show_single = False
#SEG_NUM = 3
#IMG_MAX = 1

THRESHOLD_NUM = 7
SEG_NUM = THRESHOLD_NUM + 1

IS_SEGMENT_MOD = False

SELF_FILE_NAME_EXT = ".jpg"#".bmp"#".jpg"  # ".png"
file_path = "D:\\zhangsource\\Crack Detection\\data_set\\ref_distored\\self_test_score_8\\self_"#"D:\\zhangsource\\Crack Detection\\data_set\\ref_distored\\CSIQ\\complete_test\\self_test_score_1\\self_"#"D:\\zhangsource\\Crack Detection\\data_set\\ref_distored\\tid2013\\complete_test\\self_test_score_8\\self_"#"D:\\zhangsource\\Crack Detection\\data_set\\ref_distored\\CSIQ\\complete_test\\self_test_score_1\\self_"#"D:\\zhangsource\\Crack Detection\\data_set\\ref_distored\\tid2013\\complete_test\\self_test_score_8\\self_"#"D:\\zhangsource\\Crack Detection\\data_set\\ref_distored\\tid2013\\complete_test\\self_test_score_8\\self_"#"D:\\zhangsource\\Crack Detection\\data_set\\self_taken\\self_"
IS_SIZE_GRADIENT_PROC = False
IS_BINARY_TEST = False
IS_SUMMED_TEST = False
ALIGNED_SHOW = True
img_map_index = cr5_spt.IMG_MAP_GRAY_IDX

model = eris.ErisBlock()


model.init_linear_weights()
model.init_cov_weights()


model.eval()



img_gray = cr_sf_proc.get_self_img_gray(self_img_num, img_map_idx=img_map_index, self_file_name_ext=SELF_FILE_NAME_EXT, self_path=file_path)
img_gray = cr5_spt.img_gray_resize_dim_wise_spec(np.round(img_gray*255, 0), 256,  256)/255
img_gray_bak = img_gray
img_gray = torch.from_numpy(img_gray).unsqueeze(0).to(torch.float32)



with torch.no_grad():
    img_gray_list, thresholds, img_gray_list_trans = model(img_gray)


img_gray_list = img_gray_list.detach().cpu().squeeze(0).numpy()
img_gray_list_1 = img_gray_list_trans.permute(1, 0, 2, 3).contiguous()
img_gray_list_1 = img_gray_list_1.detach().cpu().squeeze(0).numpy()
thresholds = thresholds.detach().cpu().squeeze(0).numpy()

cr5_spt.display_img_gray_plane_list(img_gray_list_1, is_single_frame=frame_show_single)

img_gray_rec_avg_mask = cr5_spt.img_gray_planes_reconstruction_avg_mask(img_gray_bak, img_gray_list)

plt.imshow(img_gray_rec_avg_mask, cmap='gray')
plt.show()

#print(thresholds)
