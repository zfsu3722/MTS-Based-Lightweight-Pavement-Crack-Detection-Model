import torch.nn as nn
import torch

class Decoder_train(nn.Module):
    def initialize_weights(self):
        for layer in self.conv_layers:
            torch.nn.init.uniform_(layer.weight, -1, 1)

    def __init__(self):
        super(Decoder_train, self).__init__()
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, stride=1)
        self.bn6 = nn.BatchNorm2d(64)
        self.tanh6 = nn.Hardtanh(min_val=0, max_val=1)

        self.conv7 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.bn7 = nn.BatchNorm2d(64)

        self.tanh7 = nn.Hardtanh(min_val=0, max_val=1)
        self.conv8 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.bn8 = nn.BatchNorm2d(64)

        self.tanh8 = nn.Hardtanh(min_val=0, max_val=1)
        self.conv9 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.bn9 = nn.BatchNorm2d(64)
        self.tanh9 = nn.Hardtanh(min_val=0, max_val=1)
        #self.conv10 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1)
        #self.bn10 = nn.BatchNorm2d(32)

        #self.tanh10 = nn.Hardtanh(min_val=0, max_val=1)

        self.deconv1 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        #self.relu = nn.ReLU()
        self.tanh1 = nn.Hardtanh(min_val=0, max_val=1)

        self.deconv2 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.tanh2 = nn.Hardtanh(min_val=0, max_val=1)
    

        self.deconv3 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(32)

        self.tanh3 = nn.Hardtanh(min_val=0, max_val=1)


        self.deconv4 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(32)

        self.tanh4 = nn.Hardtanh(min_val=0, max_val=1)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear')

        self.deconv5 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=1)
        self.bn5_d = nn.BatchNorm2d(32)

        self.tanh5_d = nn.Hardtanh(min_val=0, max_val=1)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.deconv6 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=1)
        self.bn6_d = nn.BatchNorm2d(32)

        self.tanh6_d = nn.Hardtanh(min_val=0, max_val=1)

        #self.deconv7 = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=3, stride=1)
        #self.bn7_d = nn.BatchNorm2d(1)

        #self.tanh7_d = nn.Hardtanh(min_val=0, max_val=1)
        #self.deconv8 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=1)
        #self.bn8_d = nn.BatchNorm2d(32)
        #self.tanh8_d = nn.Hardtanh(min_val=0, max_val=1)
 

        self.deconv9 = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=3, stride=1)
        self.bn9_d = nn.BatchNorm2d(1)
        self.tanh9_d = nn.Hardtanh(min_val=0, max_val=1)
        self.conv_layers = [self.conv6, self.conv7, self.conv8, self.conv9, self.deconv1, self.deconv2, self.deconv3, self.deconv4, self.deconv5, self.deconv6, self.deconv9]
        #self.conv_layers = [self.deconv1, self.deconv5, self.deconv6, self.deconv9]

        # 调用初始化函数对权重进行初始化
        #self.initialize_weights()



    def forward(self, x):
        #x = self.tanh5(self.bn5(self.conv5(x)))
        x = self.tanh6(self.bn6(self.conv6(x)))
        x = self.tanh7(self.bn7(self.conv7(x)))
        x = self.tanh8(self.bn8(self.conv8(x)))
        x = self.tanh9(self.bn9(self.conv9(x)))
        #x = self.tanh10(self.bn10(self.conv10(x)))
        x = self.tanh1(self.bn1(self.deconv1(x)))
        x = self.tanh2(self.bn2(self.deconv2(x)))
        x = self.tanh3(self.bn3(self.deconv3(x)))
        x = self.tanh4(self.bn4(self.deconv4(x)))
        x = self.upsample1(x)
        x = self.tanh5_d(self.bn5_d(self.deconv5(x)))
        x = self.upsample2(x)
        x = self.tanh6_d(self.bn6_d(self.deconv6(x)))
        #x = self.tanh7_d(self.bn7_d(self.deconv7(x)))
        #x = self.tanh8_d(self.bn8_d(self.deconv8(x)))
        x = self.tanh9_d(self.bn9_d(self.deconv9(x)))
        target_h,target_w=256,256
        _, _, h, w = x.size()
        dh, dw = (h - target_h) // 2, (w - target_w) // 2
        if h > target_h and w > target_w:
            x = x[:, :, dh:dh + target_h, dw:dw + target_w]



        #x = self.tanh5(self.conv5(x))
        #x = self.tanh6(self.conv6(x))
        #x = self.tanh7(self.conv7(x))
        
        #x = self.tanh1(self.deconv1(x))
        #x = self.tanh2(self.deconv2(x))
        #x = self.tanh3(self.deconv3(x))
        #x = self.tanh4(self.deconv4(x))
        #x = self.tanh5_d(self.deconv5(x))
        #x = self.tanh6_d(self.deconv6(x))
        return x

decoder = Decoder_train()
# decoder1 = Decoder1
total = sum([param.nelement() for param in decoder.parameters()])
print(' decoder  + Number of params: %.4fM' % (total / 1e6))
with open("decoder_params.txt", "w") as file:
    file.write("Decoder Networks Size:\n")
    file.write('+ Number of params: %.4fM' % (total / 1e6))
    #file.write("\nDecoder Parameters:\n")
with open ('3conv_model_sgd.txt','w') as f:
    print(decoder, file = f)
    for params in decoder.state_dict():   
        f.write("{}\t{}\n".format(params, decoder.state_dict()[params]))
        
    #for name, param in decoder.named_parameters():
        #file.write(f"{name}: {param.numel()}\n")
class Decoder(nn.Module):
    def __init__(self, in_channels=256):
        super(Decoder, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 1, kernel_size=1)
        )

    def forward(self, x):
        return torch.sigmoid(self.model(x))
