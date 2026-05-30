import torch
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F

a = torch.randn(2, 4, 16, 32768, device='cuda') * 0.01  # small values
a.requires_grad_(True)

scaler = GradScaler()
with autocast(dtype=torch.float16):
    b = F.normalize(a, dim=-1)
    loss = b.sum()

scaler.scale(loss).backward()
print(a.grad.max(), a.grad.min())
if torch.isnan(a.grad).any():
    print("NaN gradient!")
