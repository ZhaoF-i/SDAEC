import torch
from Network.ICCRN import NET
import soundfile as sf

net = NET()
# origin_name = ''
# net.load_state_dict(torch.load(origin_name).state_dict)
save_name = ''
# torch.save(net.state_dict(), save_name)
net.load_state_dict(torch.load(save_name))

mic, _ = sf.read('')
farend, _ = sf.read('')

mic = torch.Tensor(mic).unsqueeze(dim=0)
farend = torch.Tensor(farend).unsqueeze(dim=0)

with torch.no_grad():
    est_time = net(torch.stack([mic, farend], dim=1))

print(est_time.shape)