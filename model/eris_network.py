import torch
from torch import nn


class RangeActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, lower, upper, x):
        return ((x - lower) > 0).to(x.dtype) * ((upper - x) >= 0).to(x.dtype)


class ErisBlock(nn.Module):
    def init_linear_weights(self):
        fixed_weight_upper = torch.tensor([[0], [1], [2], [3], [4], [5], [6], [7]], dtype=self.linear_lower.weight.dtype)
        fixed_weight_lower = torch.tensor([[1], [2], [3], [4], [5], [6], [7], [100]], dtype=self.linear_lower.weight.dtype)
        with torch.no_grad():
            self.linear_upper.weight.copy_(fixed_weight_upper)
            self.linear_upper.bias.zero_()
            self.linear_lower.weight.copy_(fixed_weight_lower)
            self.linear_lower.bias.zero_()

    def init_cov_weights(self):
        with torch.no_grad():
            self.conv8.weight.fill_(1.0)
            self.conv8.bias.zero_()

    def __init__(self):
        super().__init__()
        self.linear_lower = nn.Linear(1, 8)
        self.linear_upper = nn.Linear(1, 8)
        self.relu = nn.ReLU()
        self.conv8 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=1, stride=1)
        self.range_activation = RangeActivation()
        self.register_buffer("seg_num", torch.tensor(8.0))

    def forward(self, x):
        x_max = torch.max(x)
        x_min = torch.min(x)
        x_step = (x_max - x_min)/self.seg_num
        x_step = x_step.unsqueeze(0).unsqueeze(0)
        thresh_upper = self.linear_upper(x_step)
        thresh_upper = x_max - thresh_upper
        thresh_lower = self.linear_lower(x_step)
        thresh_lower = x_max - thresh_lower
        thresh_lower = self.relu(thresh_lower)
        thresh_upper = thresh_upper.view(1, 8, 1, 1).expand(1, 8, 256, 256)
        thresh_lower_bak = thresh_lower
        thresh_lower = thresh_lower_bak.view(1, 8, 1, 1).expand(1, 8, 256, 256)
        x = self.conv8(x)
        x = self.range_activation(thresh_lower, thresh_upper, x)
        y = x.permute(1, 0, 2, 3).contiguous()
        return x, thresh_lower_bak, y
