import torch
from Network.ICCRN import NET

net = NET()
net.load_state_dict(torch.load('ICCRN_SD.ckpt').state_dict)

mic = ...
farend = ...

with torch.no_grad():
    est_time = net(torch.stack([mic, farend], dim=1))

