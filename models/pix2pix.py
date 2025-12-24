import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x

class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()
        
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)
        
        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)
        
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        output = self.final(u7)
        # 确保输出形状与输入形状一致
        _, _, h, w = x.shape
        output = F.interpolate(output, size=(h, w), mode='bilinear', align_corners=False)
        return output

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()
        
        def discriminator_block(in_filters, out_filters, normalization=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        img_input = torch.cat((img_A, img_B), 1)
        output = self.model(img_input)
        # 调整输出形状与目标一致
        output = F.interpolate(output, size=(30, 30), mode='bilinear', align_corners=False)
        return output

class Pix2Pix:
    def __init__(self, in_channels=3, out_channels=3, lr=0.0002, b1=0.5, b2=0.999, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        self.generator = GeneratorUNet(in_channels, out_channels).to(self.device)
        self.discriminator = Discriminator(in_channels).to(self.device)
        
        self.criterion_GAN = nn.MSELoss().to(self.device)
        self.criterion_pixelwise = nn.L1Loss().to(self.device)
        
        self.criterion_feature = nn.L1Loss().to(self.device)
        
        self.optimizer_G = torch.optim.Adam(
            self.generator.parameters(), lr=lr, betas=(b1, b2)
        )
        self.optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr, betas=(b1, b2)
        )
        
        self.lambda_pixel = 100
        
    def train_step(self, real_A, real_B):
        real_A = real_A.to(self.device)
        real_B = real_B.to(self.device)
        
        valid = torch.ones((real_A.size(0), 1, 30, 30), requires_grad=False).to(self.device)
        fake = torch.zeros((real_A.size(0), 1, 30, 30), requires_grad=False).to(self.device)
        
        # 训练生成器
        self.optimizer_G.zero_grad()
        
        fake_B = self.generator(real_A)
        
        pred_fake = self.discriminator(real_A, fake_B)
        loss_GAN = self.criterion_GAN(pred_fake, valid)
        
        loss_pixel = self.criterion_pixelwise(fake_B, real_B)
        
        loss_G = loss_GAN + self.lambda_pixel * loss_pixel
        
        loss_G.backward()
        self.optimizer_G.step()
        
        # 训练判别器
        self.optimizer_D.zero_grad()
        
        pred_real = self.discriminator(real_A, real_B)
        loss_real = self.criterion_GAN(pred_real, valid)
        
        pred_fake = self.discriminator(real_A, fake_B.detach())
        loss_fake = self.criterion_GAN(pred_fake, fake)
        
        loss_D = 0.5 * (loss_real + loss_fake)
        
        loss_D.backward()
        self.optimizer_D.step()
        
        return {
            'loss_G': loss_G.item(),
            'loss_D': loss_D.item(),
            'loss_GAN': loss_GAN.item(),
            'loss_pixel': loss_pixel.item(),
            'fake_B': fake_B
        }