import torch
import torch.nn as nn
import time
import numpy as np
#from torchvision import transforms
from thop import profile
from decoder_networks import Decoder_train
from pre_networks import Encoder
import cv2
import crack_500_seg_torch_support as cr5_seg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def convert_bn_to_instance_norm(module):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_features = child.num_features
            affine = child.affine
            new_layer = nn.InstanceNorm2d(num_features, affine=affine, track_running_stats=False)
            if child.weight is not None:
                new_layer.to(child.weight.device)
            if affine:
                new_layer.weight.data.copy_(child.weight.data)
                new_layer.bias.data.copy_(child.bias.data)
            setattr(module, name, new_layer)
        else:
            convert_bn_to_instance_norm(child)

def save_output(pred, save_path):
    pred = torch.sigmoid(pred)
    if pred.ndim == 4:
        pred_np = pred.detach().cpu().numpy()[0, 0, :, :]
    else:
        pred_np = pred.detach().cpu().numpy().squeeze()

    pred_np = (pred_np > 0.5).astype(np.uint8) * 255
    cv2.imwrite(save_path, pred_np)

def load_and_preprocess_full_pipeline(original_image_path_i, encoder_i, decoder_i, size=(256, 256)):
    img_gray_uint8 = cv2.imread(original_image_path_i, cv2.IMREAD_GRAYSCALE)
    img_gray_uint8 = cv2.resize(img_gray_uint8, size, interpolation=cv2.INTER_LINEAR)
    img_gray = img_gray_uint8.astype(np.float32) / 255.0
    seg_num = 8
    thresholds = cr5_seg.img_integer_segmentation_equal_range_thresholds_float_gpu(
        img_gray, seg_num, device=device
    )
    img_planes_tensor = cr5_seg.img_segmentation_threshold_list_gpu(
        img_gray, thresholds, device=device
    )
    img_planes_tensor = img_planes_tensor.unsqueeze(1).to(torch.float32)

    with torch.no_grad():
        for _ in range(10):
            _ = encoder_i(img_planes_tensor)
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            thresholds = cr5_seg.img_integer_segmentation_equal_range_thresholds_float_gpu_combined(
                img_gray, seg_num, device=device
            )
            img_planes_tensor = cr5_seg.img_segmentation_threshold_list_gpu(
                img_gray, thresholds, device=device
            )
            img_planes_tensor = img_planes_tensor.unsqueeze(1).to(torch.float32)
            enc_feats = encoder_i(img_planes_tensor)
            feat_cat_l = enc_feats.view(1, -1, enc_feats.size(2), enc_feats.size(3))
            out = decoder_i(feat_cat_l)
    torch.cuda.synchronize()
    end = time.time()
    fps = 100 / (end - start)
    print(f"FPS: {fps:.2f}")
    save_output(out, "prediction_111212-1.png")
    print("The results have been saved to prediction_111212-1.png")
    return out



if __name__ == '__main__':
    encoder_ckpt = torch.load("50.pth", map_location=device)
    encoder = Encoder().to(device)
    encoder.load_state_dict(encoder_ckpt['model_state_dict'])
    encoder.eval()
    convert_bn_to_instance_norm(encoder)
    decoder_ckpt = torch.load("decoder_epoch_241.pth", map_location=device)
    decoder = Decoder_train().to(device)
    decoder.load_state_dict(decoder_ckpt['model_state_dict'])
    decoder.eval()
    print("Computing FLOPs & Params...")
    dummy_input = torch.randn(8, 1, 256, 256).to(device)
    try:
        enc_flops, _ = profile(encoder, inputs=(dummy_input,), verbose=False)
        dummy_out = encoder(dummy_input)
        feat_cat = dummy_out.view(1, -1, dummy_out.size(2), dummy_out.size(3))
        dec_flops, _ = profile(decoder, inputs=(feat_cat,), verbose=False)
        total_flops = enc_flops + dec_flops
        total_params = sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in decoder.parameters())
        print(f"FLOPs: {total_flops / 1e9:.4f} GFLOPs")
        print(f"Params: {total_params / 1e6:.4f} M")
    except Exception as e:
        print(f"Skip FLOPs: {e}")
    print("GPU Warmup...")
    original_image_path = "111212-1.jpg"
    print(f"--- Processing {original_image_path} ---")
    imgs = load_and_preprocess_full_pipeline(original_image_path, encoder, decoder)
