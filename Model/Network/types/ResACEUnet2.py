import torch
import torch.nn as nn
import torch.nn.functional as F

class EfficientChannelAttention3D(nn.Module):
    """Mecanismo de atenção de canal eficiente para focar nas features mais importantes."""
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, D, H, W = x.size()
        y = self.avg_pool(x).view(B, C)
        y = self.fc(y).view(B, C, 1, 1, 1)
        return x * y.expand_as(x)

class TransformerSelfAttention3D(nn.Module):
    """Mecanismo de Autoatenção (Transformer) para capturar contexto global no Bottleneck."""
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, D, H, W = x.size()
        N = D * H * W
        
        # Q, K, V Projections
        q = self.query(x).view(B, -1, N).permute(0, 2, 1) # [B, N, C/8]
        k = self.key(x).view(B, -1, N)                    # [B, C/8, N]
        v = self.value(x).view(B, -1, N).permute(0, 2, 1) # [B, N, C]

        # Scaled Dot-Product Attention
        attn = torch.bmm(q, k) / (x.size(1) ** 0.5)
        attn = F.softmax(attn, dim=-1)                    # [B, N, N]

        # Aplicar atenção aos valores
        out = torch.bmm(attn, v).permute(0, 2, 1).view(B, C, D, H, W)
        
        # Conexão residual com parâmetro de escala
        return x + self.gamma * out

class ResACEBlock3D(nn.Module):
    """Bloco Híbrido: Convolução Residual + Atenção."""
    def __init__(self, in_channels, out_channels, dropout_rate=0.0, use_transformer=False):
        super().__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.dropout = nn.Dropout3d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

        # Se as dimensões mudarem, precisamos de uma convolução 1x1 no skip connection
        self.skip = nn.Sequential()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_channels)
            )
            
        # Atenção Eficiente (Padrão) ou Transformer (Para o Bottleneck)
        if use_transformer:
            self.attention = TransformerSelfAttention3D(out_channels)
        else:
            self.attention = EfficientChannelAttention3D(out_channels)

    def forward(self, x):
        residual = self.skip(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Aplica Atenção
        out = self.attention(out)
        
        # Soma Residual
        out += residual
        out = self.relu(out)
        
        return out
    
class ResACE_Unet(nn.Module):
    def __init__(self, in_channels=1, num_classes=3, base_filters=16, dropout_rate=0.2):
        """
        ResACE Unet Híbrida 3D
        Args:
            in_channels (int): Canais de entrada (ex: 1 para amplitude sísmica).
            num_classes (int): Número de classes alvo (ex: 3 para multiclasse).
            base_filters (int): Número de filtros da primeira camada (cresce multiplicando por 2).
            dropout_rate (float): Taxa de dropout para regularização.
        """
        super().__init__()
        
        filters = [base_filters, base_filters * 2, base_filters * 4, base_filters * 8]
        
        # ENCODER (Downsampling)
        self.enc1 = ResACEBlock3D(in_channels, filters[0], dropout_rate)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.enc2 = ResACEBlock3D(filters[0], filters[1], dropout_rate)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.enc3 = ResACEBlock3D(filters[1], filters[2], dropout_rate)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # BOTTLENECK (Usa Transformer aqui porque o volume espacial é pequeno)
        self.bottleneck = ResACEBlock3D(filters[2], filters[3], dropout_rate, use_transformer=True)
        
        # DECODER (Upsampling)
        self.upconv3 = nn.ConvTranspose3d(filters[3], filters[2], kernel_size=2, stride=2)
        self.dec3 = ResACEBlock3D(filters[3], filters[2], dropout_rate) # filters[3] pq concatena
        
        self.upconv2 = nn.ConvTranspose3d(filters[2], filters[1], kernel_size=2, stride=2)
        self.dec2 = ResACEBlock3D(filters[2], filters[1], dropout_rate)
        
        self.upconv1 = nn.ConvTranspose3d(filters[1], filters[0], kernel_size=2, stride=2)
        self.dec1 = ResACEBlock3D(filters[1], filters[0], dropout_rate)
        
        # OUTPUT
        self.out_conv = nn.Conv3d(filters[0], num_classes, kernel_size=1)

    def forward(self, x):
        # Caminho de Codificação
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        # Gargalo (Transformer Attention atua aqui)
        b = self.bottleneck(p3)
        
        # Caminho de Decodificação com Skip Connections
        u3 = self.upconv3(b)
        u3 = torch.cat([u3, e3], dim=1) # Concatena com Encoder 3
        d3 = self.dec3(u3)
        
        u2 = self.upconv2(d3)
        u2 = torch.cat([u2, e2], dim=1) # Concatena com Encoder 2
        d2 = self.dec2(u2)
        
        u1 = self.upconv1(d2)
        u1 = torch.cat([u1, e1], dim=1) # Concatena com Encoder 1
        d1 = self.dec1(u1)
        
        # Saída (Logits)
        out = self.out_conv(d1)
        
        return out