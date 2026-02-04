import segmentation_models_pytorch as smp
import torch.nn.functional as F
from utils.Unet import UNet3D
import torch
import torch.optim as optim
from torchmetrics.classification import MulticlassJaccardIndex
from torchmetrics.classification import BinaryJaccardIndex
from monai.networks.nets import UNet, VNet, UNETR, SwinUNETR, SegResNet


class ModelNetwork:
    selected = None

    def __init__(self, name, img_size=128, classes=1, pretrained=True, channels=1):
        self.selected = name
        self.img_size = img_size
        self.classes  = classes
        self.multiclass = (self.classes > 1)
        self.pretrained = pretrained
        self.weights    = 'imagenet' if self.pretrained else None
        self.channels   = channels

        if self.multiclass: # considerar o fundo como +1 classe
            self.classes = (self.classes + 1)
        
        self.device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model     = self.getModel().to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=2e-4, weight_decay=1e-2)

        if self.multiclass:
            self.iou = MulticlassJaccardIndex(num_classes=self.classes, average='macro', ignore_index=0).to(self.device)
        else:
            self.iou = BinaryJaccardIndex(threshold=0.5).to(self.device)
    
    def getModel(self):
        classes = self.classes

        if self.selected == 'standard':
            return UNet3D(img_channels=self.channels, num_filters=16, dropout=0.1, classes=classes)

        if self.selected == 'monai_unet':
            return UNet(spatial_dims=3, in_channels=self.channels, out_channels=self.classes, channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2)

        if self.selected == 'vnet':
            return VNet(spatial_dims=3, in_channels=self.channels, out_channels=self.classes, dropout_prob_down=0.1, dropout_prob_up=(0.1, 0.1)  )
        
        if self.selected == 'segresnet':
            return SegResNet(spatial_dims=3, in_channels=self.channels, out_channels=self.classes, init_filters=16, dropout_prob=0.1)
        
        return None
    
    def info(self):
        return {
            'multiclass': self.multiclass,
            'model_network': self.selected,
            'model_weights': self.weights,
            'model_channels': self.channels,
            'pretrained': self.pretrained
        }