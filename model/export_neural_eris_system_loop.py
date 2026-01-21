import torch
import torch.nn as nn
from pre_networks import Encoder
from decoder_networks import Decoder_train


# ==========================================
# 1. 预处理层 (BGR -> Gray -> Norm)
# ==========================================
class Preprocess_Layer(nn.Module):
    def __init__(self):
        super(Preprocess_Layer, self).__init__()
        # BGR coefficients
        self.register_buffer('weight_b', torch.tensor(0.114))
        self.register_buffer('weight_g', torch.tensor(0.587))
        self.register_buffer('weight_r', torch.tensor(0.299))
        self.register_buffer('div_255', torch.tensor(255.0))

    def forward(self, x):
        # Input: [1, 3, H, W] BGR, 0-255
        gray = x[:, 0:1, :, :] * self.weight_b + \
               x[:, 1:2, :, :] * self.weight_g + \
               x[:, 2:3, :, :] * self.weight_r
        gray = gray / self.div_255
        return gray  # [1, 1, H, W] 0-1


# ==========================================
# 2. 神经网络版 ERIS (基于你提供的 eris_network_deploy.py)
# ==========================================
class RangeActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, lower, upper, x):
        return ((x - lower) > 0).to(x.dtype) * ((upper - x) >= 0).to(x.dtype)


class ErisBlock(nn.Module):
    def __init__(self):
        super(ErisBlock, self).__init__()
        self.linear_lower = nn.Linear(1, 8)
        self.linear_upper = nn.Linear(1, 8)
        self.relu = nn.ReLU()
        # 注意：你提供的代码中 stride=1, kernel_size=1
        self.conv8 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=1, stride=1, bias=False)
        self.range_activation = RangeActivation()
        self.register_buffer("seg_num", torch.tensor(8.0))

        # 【关键】必须在这里调用初始化，否则权重是随机的，无法模拟数学逻辑
        self.init_linear_weights()
        self.init_cov_weights()

    def init_linear_weights(self):
        # 初始化 Linear 权重来模拟等差数列生成
        fixed_weight_upper = torch.tensor([[0], [1], [2], [3], [4], [5], [6], [7]],
                                          dtype=self.linear_lower.weight.dtype)
        fixed_weight_lower = torch.tensor([[1], [2], [3], [4], [5], [6], [7], [100]],
                                          dtype=self.linear_lower.weight.dtype)
        with torch.no_grad():
            self.linear_upper.weight.copy_(fixed_weight_upper)
            self.linear_upper.bias.zero_()
            self.linear_lower.weight.copy_(fixed_weight_lower)
            self.linear_lower.bias.zero_()

    def init_cov_weights(self):
        # 初始化 Conv 为复制操作 (1.0)
        with torch.no_grad():
            self.conv8.weight.fill_(1.0)
            #self.conv8.bias.zero_()

    def forward(self, x):
        x_max = torch.max(x)
        x_min = torch.min(x)
        x_step = (x_max - x_min) / self.seg_num
        x_step = x_step.unsqueeze(0).unsqueeze(0)

        thresh_upper = self.linear_upper(x_step)
        thresh_upper = x_max - thresh_upper

        thresh_lower = self.linear_lower(x_step)
        thresh_lower = x_max - thresh_lower
        thresh_lower = self.relu(thresh_lower)

        # 使用固定尺寸 expand (解决 TensorRT 动态尺寸问题)
        thresh_upper = thresh_upper.view(1, 8, 1, 1).expand(1, 8, 256, 256)
        thresh_lower = thresh_lower.view(1, 8, 1, 1).expand(1, 8, 256, 256)

        x = self.conv8(x)
        x = self.range_activation(thresh_lower, thresh_upper, x)

        # [1, 8, 256, 256] -> [8, 1, 256, 256] (Batch=8)
        x = x.permute(1, 0, 2, 3).contiguous()
        #x_list = []
        #for i in range(8):
        #    plane_i = x[:, i:i + 1]
        #    x_list.append(plane_i)
        #x = torch.cat(x_list, dim=0)
        return x


# ==========================================
# 3. 全系统整合 (带 Encoder 循环的版本)
# ==========================================
class MTSCrack_Neural_System_Loop(nn.Module):
    def __init__(self, encoder, decoder):
        super(MTSCrack_Neural_System_Loop, self).__init__()
        self.preprocess = Preprocess_Layer()
        self.eris = ErisBlock()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        # 1. 预处理 & ERIS
        x_gray = self.preprocess(x)
        planes = self.eris(x_gray)  # Shape: [8, 1, 256, 256]

        enc_feat = self.encoder(planes)

        # 4. Fusion
        fused_feat = enc_feat.view(1, -1, enc_feat.size(2), enc_feat.size(3))

        # 5. Decoder
        final_mask = self.decoder(fused_feat)

        return final_mask


# ==========================================
# 辅助函数
# ==========================================
def convert_bn_to_instance_norm(module):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            new_layer = nn.InstanceNorm2d(child.num_features, affine=child.affine, track_running_stats=False)
            if child.weight is not None:
                new_layer.to(child.weight.device)
            if child.affine:
                new_layer.weight.data.copy_(child.weight.data)
                new_layer.bias.data.copy_(child.bias.data)
            setattr(module, name, new_layer)
        else:
            convert_bn_to_instance_norm(child)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 加载权重
    print("1. Loading weights...")
    enc_ckpt = torch.load("48.pth", map_location="cpu")  # 请确认路径
    encoder = Encoder()
    encoder.load_state_dict(enc_ckpt["model_state_dict"])

    dec_ckpt = torch.load("decoder_epoch_187.pth", map_location="cpu")  # 请确认路径
    decoder = Decoder_train()
    decoder.load_state_dict(dec_ckpt["model_state_dict"])

    # 2. 转换 BN -> IN
    print("2. Converting Encoder BN -> IN...")
    convert_bn_to_instance_norm(encoder)

    # 3. 组装模型 (Loop 版)
    print("3. Assembling Neural Eris System (Explicit Loop Version)...")
    neural_system = MTSCrack_Neural_System_Loop(encoder, decoder)
    neural_system.to(device)
    neural_system.eval()

    # 4. 导出 ONNX
    onnx_path = "mtscrack_neural_eris_loop.onnx"
    print(f"4. Exporting to {onnx_path}...")

    dummy_input = torch.randn(1, 3, 256, 256).to(device) * 255.0

    torch.onnx.export(
        neural_system,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,  # 这会帮助展开循环
        input_names=["input_bgr"],
        output_names=["final_mask"]
    )

    print("\n 导出成功！")
    print(f"   文件: {onnx_path}")
    print("   特性: 采用显式循环展开，彻底解决 Encoder 维度丢失问题。")
