import torch
import torch.nn as nn
from pre_networks import Encoder
from decoder_networks import Decoder_train



class Preprocess_Layer(nn.Module):
    def __init__(self):
        super(Preprocess_Layer, self).__init__()
        self.register_buffer('weight_b', torch.tensor(0.114))
        self.register_buffer('weight_g', torch.tensor(0.587))
        self.register_buffer('weight_r', torch.tensor(0.299))
        self.register_buffer('div_255', torch.tensor(255.0))

    def forward(self, x):
        gray = x[:, 0:1, :, :] * self.weight_b + \
               x[:, 1:2, :, :] * self.weight_g + \
               x[:, 2:3, :, :] * self.weight_r
        gray = gray / self.div_255
        return gray

class RangeActivation(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, lower, upper, x):
        return ((x - lower) > 0).to(torch.float32) * ((upper - x) >= 0).to(torch.float32)


class ErisBlock(nn.Module):
    def __init__(self):
        super(ErisBlock, self).__init__()
        self.linear_lower = nn.Linear(1, 8)
        self.linear_upper = nn.Linear(1, 8)
        self.relu = nn.ReLU()
        self.conv8 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=1, stride=1, bias=False)
        self.range_activation = RangeActivation()
        self.register_buffer("seg_num", torch.tensor(8.0))
        self.init_weights()

    def init_weights(self):
        fixed_weight_upper = torch.tensor([[0], [1], [2], [3], [4], [5], [6], [7]], dtype=torch.float32)
        fixed_weight_lower = torch.tensor([[1], [2], [3], [4], [5], [6], [7], [100]], dtype=torch.float32)
        with torch.no_grad():
            self.linear_upper.weight.copy_(fixed_weight_upper)
            self.linear_upper.bias.zero_()
            self.linear_lower.weight.copy_(fixed_weight_lower)
            self.linear_lower.bias.zero_()
            self.conv8.weight.fill_(1.0)

    def forward(self, x):
        x_max = torch.max(x.view(x.size(0), -1), dim=1, keepdim=True)[0].view(x.size(0), 1, 1, 1)
        x_min = torch.min(x.view(x.size(0), -1), dim=1, keepdim=True)[0].view(x.size(0), 1, 1, 1)
        x_step = (x_max - x_min) / self.seg_num
        x_step_flat = x_step.view(-1, 1)
        thresh_upper = self.linear_upper(x_step_flat)
        thresh_upper = x_max.view(-1, 1) - thresh_upper
        thresh_lower = self.linear_lower(x_step_flat)
        thresh_lower = x_max.view(-1, 1) - thresh_lower
        thresh_lower = self.relu(thresh_lower)
        thresh_upper = thresh_upper.view(1, 8, 1, 1).expand(1, 8, 256, 256)
        thresh_lower = thresh_lower.view(1, 8, 1, 1).expand(1, 8, 256, 256)
        x = self.conv8(x)
        x = self.range_activation(thresh_lower, thresh_upper, x)
        x = x.view(-1, 1, 256, 256)
        return x

class MTSCrack_Neural_System_Batch_Static(nn.Module):
    def __init__(self, encoder, decoder):
        super(MTSCrack_Neural_System_Batch_Static, self).__init__()
        self.preprocess = Preprocess_Layer()
        self.eris = ErisBlock()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x_gray = self.preprocess(x)
        planes = self.eris(x_gray)
        enc_feat = self.encoder(planes)
        fused_feat = enc_feat.view(1, -1, 64, 64)
        final_mask = self.decoder(fused_feat)
        return final_mask

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
    print("1. Loading weights...")
    enc_ckpt = torch.load("48.pth", map_location="cpu")
    encoder = Encoder()
    encoder.load_state_dict(enc_ckpt["model_state_dict"])

    dec_ckpt = torch.load("decoder_epoch_187.pth", map_location="cpu")
    decoder = Decoder_train()
    decoder.load_state_dict(dec_ckpt["model_state_dict"])
    print("2. Converting Encoder BN -> IN...")
    convert_bn_to_instance_norm(encoder)
    print("3. Assembling Neural Eris System (Static Batch Version)...")
    neural_system = MTSCrack_Neural_System_Batch_Static(encoder, decoder)
    neural_system.to(device)
    neural_system.eval()
    onnx_path = "mtscrack_neural_eris_static.onnx"
    print(f"4. Exporting to {onnx_path}...")

    dummy_input = torch.randn(1, 3, 256, 256).to(device) * 255.0

    torch.onnx.export(
        neural_system,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input_bgr"],
        output_names=["final_mask"],
    )

    print("\n Export successful！")
    print(f"   file: {onnx_path}")
