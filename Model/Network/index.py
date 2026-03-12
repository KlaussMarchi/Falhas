import torch.nn.functional as F
import torch
import torch.optim as optim
from torchmetrics.classification import MulticlassJaccardIndex
from torchmetrics.classification import BinaryJaccardIndex
from monai.networks.nets import UNet, VNet, UNETR, SwinUNETR, SegResNet

from .types.UNet3D import UNet3D
from .types.Unet3D_V2 import Unet3D_V2
from .types.ResACEUnet import ResACEUNet2


class ModelNetwork:
    selected = None

    def __init__(self, network, img_size, classes=1, channels=1, lr=1e-4, dropout=0.1, num_filters=16):
        self.network = network
        self.img_size = img_size
        self.classes  = classes
        self.multiclass = (self.classes > 1)
        self.channels   = channels
        self.dropout    = dropout
        self.num_filters = num_filters
        self.lr = lr
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model  = self.get().to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

        if self.multiclass:
            self.iou = MulticlassJaccardIndex(num_classes=self.classes, average='macro').to(self.device)
        else:
            self.iou = BinaryJaccardIndex(threshold=0.5).to(self.device)
    
    def get(self):
        classes = self.classes

        if self.network == 'standard':
            return UNet3D(img_channels=self.channels, num_filters=self.num_filters, dropout=self.dropout, classes=classes)
        
        if self.network == 'unet3d_v2':
            return Unet3D_V2(img_channels=self.channels, classes=classes, num_filters=self.num_filters, dropout=self.dropout)

        if self.network == 'monai_unet':
            return UNet(spatial_dims=3, in_channels=self.channels, out_channels=self.classes, channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2)

        if self.network == 'vnet':
            return VNet(spatial_dims=3, in_channels=self.channels, out_channels=self.classes, dropout_prob_down=self.dropout, dropout_prob_up=(self.dropout, self.dropout))
        
        if self.network == 'segresnet':
            return SegResNet(spatial_dims=3, in_channels=self.channels, out_channels=self.classes, init_filters=self.num_filters, dropout_prob=self.dropout)
        
        if self.network == 'resaceunet':
            return ResACEUNet2(in_channels=self.channels, out_channels=self.classes, img_size=self.img_size[0] if isinstance(self.img_size, tuple) else self.img_size, feature_size=self.num_filters, hidden_size=256, num_heads=4, drop_rate=self.dropout, attn_drop_rate=self.dropout, depths=[3, 3, 3, 3], dims=[32, 64, 128, 256])
        
        return None
