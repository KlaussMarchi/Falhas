import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


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
    def __init__(self, weights=0.5):
        super(DiceFocalLoss, self).__init__()
        self.dice_weight  = weights
        self.focal_weight = (1 - weights)
        
        self.dice_loss  = smp.losses.DiceLoss(mode='multiclass', from_logits=True)
        self.focal_loss = smp.losses.FocalLoss(mode='multiclass', gamma=2.0)

    def forward(self, preds, targets):
        dice  = self.dice_loss(preds, targets)
        focal = self.focal_loss(preds, targets)
        return (self.dice_weight * dice) + (self.focal_weight * focal)

