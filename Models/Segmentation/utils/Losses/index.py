import torch.nn as nn
import torch
from monai import losses


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._loss = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, predictions, targets):
        return self._loss(predictions, targets.long())

class BinaryCrossEntropyWithLogits(nn.Module):
    def __init__(self):
        super().__init__()
        self._loss = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, predictions, targets):
        return self._loss(predictions, targets.float())

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
        self._loss = losses.FocalLoss(to_onehot_y=True, gamma=2.0)

    def forward(self, predicted, target):
        return self._loss(predicted, target)

class MultiClassTverskyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._loss = losses.TverskyLoss(softmax=True, to_onehot_y=True)
    
    def forward(self, predicted, target):
        return self._loss(predicted, target)

class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._loss = losses.DiceLoss(to_onehot_y=False, sigmoid=True)
    def forward(self, p, t): return self._loss(p, t)

class BinaryDiceCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._loss = losses.DiceCELoss(to_onehot_y=False, sigmoid=True)
    def forward(self, p, t): return self._loss(p, t)


class FocalLossProb(nn.Module):
    def __init__(self, alpha=0.9, gamma=2.0, eps=1e-7, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.reduction = reduction

    def forward(self, p, y):
        # ensure shapes and dtypes
        p = p.squeeze(1) if p.dim() == 4 and p.size(1) == 1 else p
        y = y.squeeze(1) if y.dim() == 4 and y.size(1) == 1 else y
        p = p.clamp(self.eps, 1.0 - self.eps)
        y = y.float()

        # pt = probability of the true class
        pt = torch.where(y > 0.5, p, 1.0 - p) # (N,H,W)
        alpha_t = torch.where(y > 0.5, self.alpha, 1.0 - self.alpha)
        loss    = - alpha_t * ((1.0 - pt) ** self.gamma) * torch.log(pt)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, eps=1e-7):
        super().__init__()
        self.alpha, self.beta, self.gamma, self.eps = alpha, beta, gamma, eps
        
    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        tp = (p*targets).sum(dim=(2,3))
        fp = ((1-targets)*p).sum(dim=(2,3))
        fn = (targets*(1-p)).sum(dim=(2,3))
        tversky = (tp + self.eps) / (tp + self.alpha*fp + self.beta*fn + self.eps)
        loss = (1 - tversky)**self.gamma
        return loss.mean()
            
class DiceFocalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._loss = losses.DiceFocalLoss(to_onehot_y=False, sigmoid=True)

    def forward(self, predicted, target):
        loss = self._loss(predicted, target)
        return loss

class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.bce = nn.BCELoss()
        self.smooth = smooth
        
    def forward(self, p, y):
        p = p.squeeze(1)
        y = y.squeeze(1).float()
        bce = self.bce(p, y)
        intersection = (p * y).sum(dim=(1,2))
        dice = (2.*intersection + self.smooth) / (p.sum(dim=(1,2)) + y.sum(dim=(1,2)) + self.smooth)
        dice_loss = 1 - dice.mean()
        return 0.5*bce + 0.5*dice_loss

class Losses:
    binary = {
        'cross_entropy': CrossEntropyLoss(), 
        'bce_logits': BinaryCrossEntropyWithLogits(),
        'dice': BinaryDiceLoss(),
        'diceCE': BinaryDiceCELoss(),
        'focal': FocalLossProb(alpha=0.90, gamma=2.0),
        'bce': nn.BCELoss(),
        'dice_bce':  DiceBCELoss(),
        'tversky': FocalTverskyLoss(),
        'dice_focal': DiceFocalLoss()
    }

    multiclass = {
        'cross_entropy': CrossEntropyLoss(),      # O padrão ouro para multiclasse
        'dice': MultiClassDiceLoss(),             # Dice com Softmax
        'diceCE': MultiClassDiceCELoss(),         # Soma de Dice + CE
        'dice_focal': MultiClassDiceFocalLoss(),  # Soma de Dice + Focal
        'focal': MultiClassFocalLoss(),           # Focal do MONAI adaptada
        'tversky': MultiClassTverskyLoss(),       # Tversky do MONAI
    }

    def __new__(cls, name, multiclass=False):
        return cls.multiclass[name] if multiclass else cls.binary[name]
