import torch
import torch.nn as nn
from monai import losses


class MonaiLoss(nn.Module):
    def __init__(self, multiclass=False):
        super().__init__()
        self.multiclass = multiclass

    def forward(self, logits, target):
        with torch.amp.autocast('cuda', enabled=False):
            logits = logits.float()
            target = target.float()

            # PROTEÇÃO DE SHAPE (OBRIGATÓRIA)
            if not self.multiclass:
                if target.shape != logits.shape:
                    target = target.view_as(logits)

            elif target.dim() == logits.dim() - 1:
                target = target.unsqueeze(1)

            return self._loss(logits, target)

class FocalLoss(MonaiLoss):
    def __init__(self, multiclass=False, gamma=2.0, alpha=None):
        super().__init__(multiclass)

        self._loss = losses.FocalLoss(
            to_onehot_y=multiclass,
            use_softmax=multiclass,
            include_background=True,
            gamma=gamma,
            alpha=alpha,
        )

class DiceFocalLoss(MonaiLoss):
    def __init__(self, multiclass=False, gamma=2.0, alpha=None):
        super().__init__(multiclass)

        self._loss = losses.DiceFocalLoss(
            to_onehot_y=multiclass,
            softmax=multiclass,
            sigmoid=not multiclass,
            include_background=True,
            gamma=gamma,
            alpha=alpha,
        )

class CrossEntropyLoss(nn.Module):
    def __init__(self, multiclass=False, weight=None):
        super().__init__()
        self.multiclass = multiclass
        self._loss = nn.CrossEntropyLoss(weight=weight) if multiclass else nn.BCEWithLogitsLoss()

    def forward(self, logits, target):
        with torch.amp.autocast('cuda', enabled=False):
            logits = logits.float()

            if not self.multiclass:
                target = target.float()

                if target.shape != logits.shape:
                    target = target.view_as(logits)

                return self._loss(logits, target)

            if target.dim() == logits.dim():
                target = target.squeeze(1)

            return self._loss(logits, target.long())

# DICE SUAVIZADO DO MACNN (GAO ET AL., 2022, EQ. 7): L = 1 - (2*sum(p*g) + 1) / (sum(p) + sum(g) + 1).
# A soma e global sobre o lote inteiro, como no artigo, e o suavizador e 1 (nao 1e-5)
class SmoothDiceLoss(nn.Module):
    SMOOTH = 1.0

    def __init__(self, multiclass=False):
        super().__init__()
        self.multiclass = multiclass

    def forward(self, logits, target):
        with torch.amp.autocast('cuda', enabled=False):
            logits = logits.float()
            target = target.float()

            if self.multiclass:
                probs  = torch.softmax(logits, dim=1)
                target = torch.zeros_like(probs).scatter_(1, target.long().view(probs.shape[0], 1, *probs.shape[2:]), 1.0)
            else:
                probs = torch.sigmoid(logits)

                if target.shape != probs.shape:
                    target = target.view_as(probs)

            overlap = (probs * target).sum()
            volume  = probs.sum() + target.sum()
            return 1.0 - (2.0 * overlap + self.SMOOTH) / (volume + self.SMOOTH)


class Losses:
    FOCAL_ALPHA = 0.93

    options = {
        'cross_entropy': CrossEntropyLoss,
        'dice_focal': DiceFocalLoss,
        'focal': FocalLoss,
        'smooth_dice': SmoothDiceLoss,
    }

    def __new__(cls, name, multiclass=False):
        kwargs = {'alpha': cls.FOCAL_ALPHA} if name == 'focal' else {}
        return cls.options[name](multiclass=multiclass, **kwargs)
