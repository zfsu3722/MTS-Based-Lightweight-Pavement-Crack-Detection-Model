import infer_trt_mtscrack as infer_machine
import numpy as np
import time as tm
import pycuda.driver as cuda
import crack_500_seg_torch_support as cr5_seg
import torch
import cv2

#input_np = np.random.randn(8, 1 , 256, 256).astype(np.float16)
print("testing....")
device = "cuda"
size = (256, 256)
original_image_path = "20.jpg"
img_rgb = cv2.imread(original_image_path)
img_rgb = cv2.resize(img_rgb, size, interpolation=cv2.INTER_LINEAR)
img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
img_rgb = img_rgb.astype("float32")
img_rgb = img_rgb / 255.0
img_rgb = img_rgb.transpose(2, 0, 1)
img_rgb = np.ascontiguousarray(img_rgb)
img_rgb = img_rgb[None, ...]
print(img_rgb.shape)
#img_gray = np.random.randn(1, 3, 256, 256).astype(np.float32)
seg_num = 8

i = 0
start_evt = cuda.Event()
end_evt   = cuda.Event()
start_evt.record()
for i in range(1000):
    #thresholds = cr5_seg.img_integer_segmentation_equal_range_thresholds_float_cpu(img_gray, seg_num)

    #img_planes_tensor = cr5_seg.img_segmentation_threshold_list_cpu(img_gray, thresholds)
    #img_planes_tensor = np.asarray(img_planes_tensor).reshape(8, 1, 256, 256).astype(np.float16)
    #torch.cuda.synchronize()
    output = infer_machine.infer(img_rgb)
end_evt.record()
end_evt.synchronize()
print(1000/(start_evt.time_till(end_evt) / 1000))
save_path = "20.png"
output = output.squeeze(1)
pred_np = (output > 0.5).astype(np.uint8) * 255
pred_np = pred_np.transpose(1, 2, 0)
print(pred_np.shape)
cv2.imwrite(save_path, pred_np)
