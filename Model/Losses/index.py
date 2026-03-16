import torch
import torch.nn as nn
from monai import losses


# --- MULTICLASS WRAPPERS (MONAI) ---

class MultiClassDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._loss = losses.DiceLoss(softmax=True, to_onehot_y=True, include_background=True)

    def forward(self, predicted, target):
        return self._loss(predicted, target)

class MultiClassDiceCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._loss = losses.DiceCELoss(softmax=True, to_onehot_y=True, include_background=True)

    def forward(self, predicted, target):
        return self._loss(predicted, target)

class MultiClassDiceFocalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._loss = losses.DiceFocalLoss(softmax=True, to_onehot_y=True, include_background=True)

    def forward(self, predicted, target):
        return self._loss(predicted, target)

class MultiClassFocalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._loss = losses.FocalLoss(to_onehot_y=True, gamma=2.0, use_softmax=True)

    def forward(self, predicted, target):
        return self._loss(predicted, target)

class MultiClassTverskyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._loss = losses.TverskyLoss(softmax=True, to_onehot_y=True)
    
    def forward(self, predicted, target):
        return self._loss(predicted, target)

# --- BINARY WRAPPERS ---

class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._loss = losses.DiceLoss(to_onehot_y=False, sigmoid=True)
    def forward(self, p, t): return self._loss(p, t)

class BinaryDiceFocalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._loss = losses.DiceFocalLoss(to_onehot_y=False, sigmoid=True)
    def forward(self, p, t): return self._loss(p, t)

class CustomDiceBCELoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth
        
    def forward(self, logits, y):
        bce_loss = self.bce(logits, y.float())
        p        = torch.sigmoid(logits)
        
        spatial_dims = (2, 3, 4) if p.ndim == 5 else (2, 3)
        intersection = (p * y).sum(dim=spatial_dims)
        union = p.sum(dim=spatial_dims) + y.sum(dim=spatial_dims)
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean()
        return 0.5 * bce_loss + 0.5 * dice_loss


class Losses:
    binary = {
        'dice': BinaryDiceLoss(),
        'dice_bce':  CustomDiceBCELoss(),
        'dice_focal': BinaryDiceFocalLoss()
    }

    multiclass = {   
        'dice': MultiClassDiceLoss(),        
        'dice_focal': MultiClassDiceFocalLoss(),  
        'focal': MultiClassFocalLoss(),           
        'tversky': MultiClassTverskyLoss(),       
    }

    def __new__(cls, name, multiclass=False):
        selected = cls.multiclass if multiclass else cls.binary
        return selected[name]