import segmentation_models_pytorch as smp
import torch.nn.functional as F
from utils.Unet import UNet3D
import torch
import torch.optim as optim
from torchmetrics.classification import MulticlassJaccardIndex
from torchmetrics.classification import BinaryJaccardIndex


class ModelNetwork:
    selected = None
    encoders = ['resnet34', 'resnet50', 'resnet101', 'efficientnet-b3', 'timm-res2net50_26w_4s']

    def __init__(self, name, encoder='resnet34', classes=1, pretrained=True, channels=1):
        self.selected = name
        self.encoder  = encoder
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
        encoder = self.encoder 

        if self.selected == 'standard':
            return UNet3D(img_channels=self.channels, num_filters=16, dropout=0.1, classes=classes)
        
        if self.selected == 'unet':
            return smp.UnetPlusPlus(encoder_name=encoder, encoder_weights=self.weights, in_channels=self.channels, classes=classes, activation=None)
        
        if self.selected == 'deep_lab':
            return smp.DeepLabV3Plus(encoder_name=encoder, encoder_weights=self.weights, in_channels=self.channels, classes=classes, activation=None)

        return None
    
    def info(self):
        return {
            'multiclass': self.multiclass,
            'model_network': self.selected,
            'model_encoder': self.encoder,
            'model_weights': self.weights,
            'model_channels': self.channels,
            'pretrained': self.pretrained
        }