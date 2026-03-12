import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(activation):
    if activation == 'leaky':
        return nn.LeakyReLU(0.1, inplace=True)
    return nn.ReLU(inplace=True)

def get_norm(out_channels, use_norm):
    if use_norm:
        # GroupNorm funciona muito bem para lotes pequenos. Usa 8 grupos como no seu original.
        groups = 8 if out_channels >= 8 and out_channels % 8 == 0 else 1
        return nn.GroupNorm(num_groups=groups, num_channels=out_channels)
    return nn.Identity()

class Conv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, use_norm=True, activation='leaky'):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=not use_norm)
        self.norm1 = get_norm(out_channels, use_norm)
        self.act1  = get_activation(activation)

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=not use_norm)
        self.norm2 = get_norm(out_channels, use_norm)
        self.act2  = get_activation(activation)

    def forward(self, x):
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.act2(self.norm2(self.conv2(x)))
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, use_norm=True, activation='leaky', dropout=0.1, pool_size=(2,2,2)):
        super().__init__()
        self.conv_block = Conv3DBlock(in_channels, out_channels, kernel_size, use_norm, activation)
        self.pool = nn.MaxPool3d(kernel_size=pool_size, stride=pool_size)
        self.dropout = nn.Dropout3d(dropout) # Equivalente ao SpatialDropout3D

    def forward(self, x):
        x_conv = self.conv_block(x)
        p = self.pool(x_conv)
        p = self.dropout(p)
        return x_conv, p

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, kernel_size=3, use_norm=True, activation='leaky', dropout=0.1, up_size=(2,2,2)):
        super().__init__()
        self.up_size = up_size
        
        # Upsampling (Nearest) seguido de Convolução para refinar
        self.up_conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2, bias=not use_norm)
        self.norm = get_norm(out_channels, use_norm)
        self.act = get_activation(activation)

        self.dropout = nn.Dropout3d(dropout)
        # O in_channels do conv_block será o out_channels atual + canais da conexão de salto (skip)
        self.conv_block = Conv3DBlock(out_channels + skip_channels, out_channels, kernel_size, use_norm, activation)

    def forward(self, x, skip):
        # Upsampling
        x = F.interpolate(x, scale_factor=self.up_size, mode='nearest')
        x = self.act(self.norm(self.up_conv(x)))

        # Concatenação: dim=1 representa os canais (Batch, Channels, Depth, Height, Width)
        x = torch.cat([x, skip], dim=1)
        x = self.dropout(x)

        x = self.conv_block(x)
        return x

class DilatedBottleneck(nn.Module):
    """
    Aumenta o campo receptivo sem mais pooling. 
    Usando expansões residuais dilatas.
    """
    def __init__(self, in_channels, out_channels, activation='leaky', use_norm=True):
        super().__init__()
        self.rates = [(2,2,2), (4,4,4)]
        self.blocks = nn.ModuleList()

        for rate in self.rates:
            # Em PyTorch, padding = dilation para manter as dimensões com kernel=3
            padding = rate[0] # Assumindo dilation simétrica para simplificar a lógica
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=rate, dilation=rate, bias=not use_norm),
                get_norm(out_channels, use_norm),
                get_activation(activation)
            )
            self.blocks.append(layer)

    def forward(self, x):
        for block in self.blocks:
            y = block(x)
            x = x + y # Conexão residual leve
        return x

class Unet3D_V2(nn.Module):
    def __init__(self, img_channels=1, classes=7, num_filters=16, kernel_size=3, batchnorm=True, activation='leaky', dropout=0.05):
        super(Unet3D_V2, self).__init__()
        self.classes = classes
        use_norm = bool(batchnorm)

        # -------- Encoder --------
        # Níveis 1-2: NÃO reduzem a profundidade D (pool (1,2,2))
        self.enc1 = EncoderBlock(img_channels, num_filters*1, kernel_size, use_norm, activation, dropout*0.25, pool_size=(1,2,2))
        self.enc2 = EncoderBlock(num_filters*1, num_filters*2, kernel_size, use_norm, activation, dropout*0.75, pool_size=(1,2,2))
        
        # Níveis 3-4: reduzem também D (pool cúbico)
        self.enc3 = EncoderBlock(num_filters*2, num_filters*4, kernel_size, use_norm, activation, dropout*1.5, pool_size=(2,2,2))
        self.enc4 = EncoderBlock(num_filters*4, num_filters*8, kernel_size, use_norm, activation, dropout*2.5, pool_size=(2,2,2))

        # -------- Bottleneck --------
        self.bot_conv = Conv3DBlock(num_filters*8, num_filters*16, kernel_size, use_norm, activation)
        self.bot_drop = nn.Dropout3d(dropout*3)
        self.bot_dilated = DilatedBottleneck(num_filters*16, num_filters*16, activation, use_norm)

        # -------- Decoder --------
        self.dec1 = DecoderBlock(num_filters*16, num_filters*8, num_filters*8, kernel_size, use_norm, activation, dropout, up_size=(2,2,2))
        self.dec2 = DecoderBlock(num_filters*8,  num_filters*4, num_filters*4, kernel_size, use_norm, activation, dropout, up_size=(2,2,2))
        self.dec3 = DecoderBlock(num_filters*4,  num_filters*2, num_filters*2, kernel_size, use_norm, activation, dropout, up_size=(1,2,2))
        self.dec4 = DecoderBlock(num_filters*2,  num_filters*1, num_filters*1, kernel_size, use_norm, activation, dropout, up_size=(1,2,2))

        # Saída
        self.out_conv = nn.Conv3d(num_filters*1, classes, kernel_size=1)

    def forward(self, x):
        # Lógica de padding automático herdada do seu primeiro script (opcional, mas seguro)
        d, h, w = x.shape[2:]
        pad_d = (16 - d % 16) % 16
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16
        
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))

        # Caminho descendente
        c1, p1 = self.enc1(x)
        c2, p2 = self.enc2(p1)
        c3, p3 = self.enc3(p2)
        c4, p4 = self.enc4(p3)

        # Gargalo (Bottleneck)
        bn = self.bot_conv(p4)
        bn = self.bot_drop(bn)
        bn = self.bot_dilated(bn)

        # Caminho ascendente
        d1 = self.dec1(bn, c4)
        d2 = self.dec2(d1, c3)
        d3 = self.dec3(d2, c2)
        d4 = self.dec4(d3, c1)

        # Saída pura (Logits)
        logits = self.out_conv(d4)

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            logits = logits[:, :, :d, :h, :w]

        return logits
