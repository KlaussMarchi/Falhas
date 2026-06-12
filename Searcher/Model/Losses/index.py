import torch
import torch.nn as nn
from monai import losses


# --- MULTICLASS WRAPPERS (MONAI) ---

class MultiClassDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._loss = losses.DiceLoss(softmax=True, to_onehot_y=True, include_background=True)

    def forward(self, predicted, target):
        with torch.amp.autocast('cuda', enabled=False):
            return self._loss(predicted.float(), target.float())

class MultiClassDiceCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._loss = losses.DiceCELoss(softmax=True, to_onehot_y=True, include_background=True)

    def forward(self, predicted, target):
        with torch.amp.autocast('cuda', enabled=False):
            return self._loss(predicted.float(), target.float())

class MultiClassDiceFocalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._loss = losses.DiceFocalLoss(softmax=True, to_onehot_y=True, include_background=True)

    def forward(self, predicted, target):
        with torch.amp.autocast('cuda', enabled=False):
            return self._loss(predicted.float(), target.float())

class MultiClassFocalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._loss = losses.FocalLoss(to_onehot_y=True, gamma=2.0, use_softmax=True)

    def forward(self, predicted, target):
        with torch.amp.autocast('cuda', enabled=False):
            return self._loss(predicted.float(), target.float())

class MultiClassTverskyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._loss = losses.TverskyLoss(softmax=True, to_onehot_y=True)
    
    def forward(self, predicted, target):
        with torch.amp.autocast('cuda', enabled=False):
            return self._loss(predicted.float(), target.float())

# --- BINARY WRAPPERS ---

class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._loss = losses.DiceLoss(to_onehot_y=False, sigmoid=True)

    def forward(self, p, t):
        with torch.amp.autocast('cuda', enabled=False):
            return self._loss(p.float(), t.float())


class BinaryDiceFocalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._loss = losses.DiceFocalLoss(to_onehot_y=False, sigmoid=True)

    def forward(self, p, t):
        with torch.amp.autocast('cuda', enabled=False):
            return self._loss(p.float(), t.float())

class CustomDiceBCELoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss() 
        self.smooth = smooth
        
    def forward(self, logits, y):
        with torch.amp.autocast('cuda', enabled=False):
            logits = logits.float()
            y = y.float()
            
            if y.shape != logits.shape:
                y = y.view_as(logits)

            bce_loss = self.bce(logits, y)
            p = torch.sigmoid(logits)
            
            p_flat = p.contiguous().view(p.shape[0], -1)
            y_flat = y.contiguous().view(y.shape[0], -1)
            
            intersection = (p_flat * y_flat).sum(dim=1)
            union = p_flat.sum(dim=1) + y_flat.sum(dim=1)
            dice = (2. * intersection + self.smooth) / (union + self.smooth)
            dice_loss = 1 - dice.mean()
            return 0.5 * bce_loss + 0.5 * dice_loss


class BinaryDiceFocalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._loss = losses.DiceFocalLoss(to_onehot_y=False, sigmoid=True)

    def forward(self, logits, target):
        with torch.amp.autocast('cuda', enabled=False):
            logits = logits.float()
            target = target.float()
            
            # PROTEÇÃO DE SHAPE (OBRIGATÓRIA)
            if target.shape != logits.shape:
                target = target.view_as(logits)
                
            return self._loss(logits, target)


class Losses:
    binary = {
        'dice': BinaryDiceLoss(),
        'dice_bce':  CustomDiceBCELoss(),
        'dice_focal': BinaryDiceFocalLoss(),
        'focal': BinaryDiceFocalLoss()
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