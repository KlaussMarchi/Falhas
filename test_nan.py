import torch
import torch.nn as nn
from Model.Network.types.ResACEUnetWu import ResACEUnetWu
from monai.losses import DiceCELoss

torch.autograd.set_detect_anomaly(True)

model = ResACEUnetWu(
    in_channels=1, 
    out_channels=1, 
    img_size=128, 
    feature_size=16, 
    hidden_size=256, 
    num_heads=4, 
    drop_rate=0.1, 
    attn_drop_rate=0.1, 
    depths=[1, 1, 1, 1], 
    dims=[32, 64, 128, 256]
).cuda()

x = torch.randn(1, 1, 128, 128, 128).cuda() * 2.0
y = torch.randint(0, 2, (1, 1, 128, 128, 128)).float().cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = DiceCELoss(sigmoid=True)

for i in range(2):
    optimizer.zero_grad()
    logits = model(x)
    loss = loss_fn(logits, y)
    print(f"Iter {i}, loss: {loss.item()}")
    loss.backward()
    
    has_nan = False
    for name, p in model.named_parameters():
        if p.grad is not None and torch.isnan(p.grad).any():
            print(f"NaN gradient in {name}")
            has_nan = True
            break
            
    if has_nan:
        break
        
    optimizer.step()

