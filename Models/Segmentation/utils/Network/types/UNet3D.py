import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, batchnorm=True, activation='relu'):
        super(DoubleConv, self).__init__()
        padding = kernel_size // 2  # mantendo as dimensões espaciais iguais
        layers  = []
        
        # Primeira convolução 3D
        layers.append(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding))
        if batchnorm:
            layers.append(nn.BatchNorm3d(out_channels))
        
        if activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif activation == 'elu':
            layers.append(nn.ELU(inplace=True))
        else:
            layers.append(nn.ReLU(inplace=True))
            
        # Segunda convolução 3D
        layers.append(nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, padding=padding))
        if batchnorm:
            layers.append(nn.BatchNorm3d(out_channels))
            
        if activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif activation == 'elu':
            layers.append(nn.ELU(inplace=True))
        else:
            layers.append(nn.ReLU(inplace=True))
            
        self.double_conv = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.double_conv(x)

class UNet3D(nn.Module):
    def __init__(self, img_channels=1, num_filters=16, kernel_size=3, batchnorm=True, activation='relu', dropout=0.1, classes=1):
        super(UNet3D, self).__init__()
        self.classes = classes

        # --- Encoder (Contração) ---
        self.enc_conv1 = DoubleConv(img_channels, num_filters, kernel_size, batchnorm, activation)
        self.pool1 = nn.MaxPool3d(2) # Reduz D, H e W pela metade
        self.drop1 = nn.Dropout(dropout * 0.5)
        
        self.enc_conv2 = DoubleConv(num_filters, num_filters * 2, kernel_size, batchnorm, activation)
        self.pool2 = nn.MaxPool3d(2)
        self.drop2 = nn.Dropout(dropout)
        
        self.enc_conv3 = DoubleConv(num_filters * 2, num_filters * 4, kernel_size, batchnorm, activation)
        self.pool3 = nn.MaxPool3d(2)
        self.drop3 = nn.Dropout(dropout)
        
        self.enc_conv4 = DoubleConv(num_filters * 4, num_filters * 8, kernel_size, batchnorm, activation)
        self.pool4 = nn.MaxPool3d(2)
        self.drop4 = nn.Dropout(dropout)
        
        # --- Bottleneck ---
        self.bottleneck = DoubleConv(num_filters * 8, num_filters * 16, kernel_size, batchnorm, activation)
        
        # --- Decoder (Expansão) ---
        # ConvTranspose3d para aumentar a resolução espacial (D, H, W)
        self.upconv1 = nn.ConvTranspose3d(num_filters * 16, num_filters * 8, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.drop_d1 = nn.Dropout(dropout)
        self.dec_conv1 = DoubleConv(num_filters * 16, num_filters * 8, kernel_size, batchnorm, activation)
        
        self.upconv2 = nn.ConvTranspose3d(num_filters * 8, num_filters * 4, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.drop_d2 = nn.Dropout(dropout)
        self.dec_conv2 = DoubleConv(num_filters * 8, num_filters * 4, kernel_size, batchnorm, activation)
        
        self.upconv3 = nn.ConvTranspose3d(num_filters * 4, num_filters * 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.drop_d3 = nn.Dropout(dropout)
        self.dec_conv3 = DoubleConv(num_filters * 4, num_filters * 2, kernel_size, batchnorm, activation)
        
        self.upconv4 = nn.ConvTranspose3d(num_filters * 2, num_filters * 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.drop_d4 = nn.Dropout(dropout)
        self.dec_conv4 = DoubleConv(num_filters * 2, num_filters * 1, kernel_size, batchnorm, activation)
        
        # Camada de saída
        self.out_conv = nn.Conv3d(num_filters, self.classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        c1 = self.enc_conv1(x)
        p1 = self.drop1(self.pool1(c1))
        
        c2 = self.enc_conv2(p1)
        p2 = self.drop2(self.pool2(c2))
        
        c3 = self.enc_conv3(p2)
        p3 = self.drop3(self.pool3(c3))
        
        c4 = self.enc_conv4(p3)
        p4 = self.drop4(self.pool4(c4))
        
        # Bottleneck
        bn = self.bottleneck(p4)
        
        # Decoder
        u1 = self.upconv1(bn)
        x = torch.cat([u1, c4], dim=1) 
        x = self.drop_d1(x)
        x = self.dec_conv1(x)
        
        u2 = self.upconv2(x)
        x = torch.cat([u2, c3], dim=1)
        x = self.drop_d2(x)
        x = self.dec_conv2(x)
        
        u3 = self.upconv3(x)
        x = torch.cat([u3, c2], dim=1)
        x = self.drop_d3(x)
        x = self.dec_conv3(x)
        
        u4 = self.upconv4(x)
        x = torch.cat([u4, c1], dim=1)
        x = self.drop_d4(x)
        x = self.dec_conv4(x)
        
        out = self.out_conv(x)
        return out
