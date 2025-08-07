import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
class Encoder(nn.Module):
    def initialize_weights(self):
        for layer in self.conv_layers:
            if isinstance(layer, SeparableConv2d):
               torch.nn.init.uniform_(layer.depthwise.weight, -1, 1)
               torch.nn.init.uniform_(layer.pointwise.weight, -1, 1)


    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = SeparableConv2d(in_channels=1, out_channels=32,padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.tanh1 = nn.Hardtanh(min_val=0, max_val=1)
        self.conv2 = SeparableConv2d(in_channels=32, out_channels=32,padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.tanh2 = nn.Hardtanh(min_val=0, max_val=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = SeparableConv2d(in_channels=32, out_channels=32,padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.tanh3 = nn.Hardtanh(min_val=0, max_val=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = SeparableConv2d(in_channels=32, out_channels=32,padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.tanh4 = nn.Hardtanh(min_val=0, max_val=1)

        self.conv_layers = [self.conv1, self.conv2, self.conv3, self.conv4]
        self.initialize_weights()


    def forward(self, x):
        x = self.tanh1(self.bn1(self.conv1(x)))
        x = self.tanh2(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.tanh3(self.bn3(self.conv3(x)))
        x = self.pool2(x)
        x = self.tanh4(self.bn4(self.conv4(x)))
        return x



class Decoder(nn.Module):
    def initialize_weights(self):
        for layer in self.conv_layers:
            torch.nn.init.uniform_(layer.weight, -1, 1)
    def __init__(self):
        super(Decoder, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.tanh1 = nn.Hardtanh(min_val=0, max_val=1)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.deconv2 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.tanh2 = nn.Hardtanh(min_val=0, max_val=1)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.deconv3 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=1,padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.tanh3 = nn.Hardtanh(min_val=0, max_val=1)
        self.deconv4 = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=3, stride=1,padding=1)
        self.bn4 = nn.BatchNorm2d(1)
        self.tanh4 = nn.Hardtanh(min_val=0, max_val=1)
        self.conv_layers = [self.deconv1, self.deconv2, self.deconv3, self.deconv4]
        self.initialize_weights()

    def forward(self, x):
        x = self.tanh1(self.bn1(self.deconv1(x)))
        x = self.upsample1(x)
        x = self.tanh2(self.bn2(self.deconv2(x)))
        x = self.upsample2(x)
        x = self.tanh3(self.bn3(self.deconv3(x)))
        x = self.tanh4(self.bn4(self.deconv4(x)))

        return x


encoder = Encoder()
decoder = Decoder()
autoencoder = AutoEncoder()
total = sum([param.nelement() for param in encoder.parameters()])
print('  + Number of params: %.4fM' % (total / 1e6))
total = sum([param.nelement() for param in decoder.parameters()])
print('  + Number of params: %.4fM' % (total / 1e6))
total = sum([param.nelement() for param in autoencoder.parameters()])
print('  + Number of params: %.4fM' % (total / 1e6))


