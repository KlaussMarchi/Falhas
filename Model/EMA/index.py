import copy
import torch


class ModelEMA:
    """Exponential moving average of the model weights, evaluated in place of the raw model.

    The decay ramps up as (1 + step) / (10 + step) capped at `decay`, so early updates track the
    live model closely and the average only becomes slow once training stabilises. Float tensors
    (parameters and BN running stats) are averaged; integer buffers are copied.

    Ligada por "ema": true no info.json; com false o Trainer nem cria o objeto e usa o modelo vivo.
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.step  = 0

        self.model = copy.deepcopy(model).eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.step += 1
        decay = min(self.decay, (1.0 + self.step) / (10.0 + self.step))

        ema_state = self.model.state_dict()
        for key, value in model.state_dict().items():
            target = ema_state[key]
            if target.dtype.is_floating_point:
                target.mul_(decay).add_(value.detach(), alpha=1.0 - decay)
            else:
                target.copy_(value)

    def state_dict(self):
        return self.model.state_dict()
