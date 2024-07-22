import torch
import soundfile as sf
from Network.ICCRN import NET
from Network.alpha_predictor import AlphaPredictor

def stft(x):
    b, m, t = x.shape[0], x.shape[1], x.shape[2],
    x = x.reshape(-1, t)
    X = torch.stft(x, n_fft=319, hop_length=160, window=torch.hamming_window(319))
    F, T = X.shape[1], X.shape[2]
    X = X.reshape(b, m, F, T, 2)
    X = torch.cat([X[..., 0], X[..., 1]], dim=1)
    return X


network = NET()
origin_name = '/ddnstor/imu_fanhaipeng/result/AEC-challenge-2024/ICCRN_E2E_splitAlphaPredictor10_batch8_OneOptimizer/checkpoints/net-best.ckpt'
network.load_state_dict(torch.load(origin_name).state_dict)
save_name = '/ddnstor/imu_fanhaipeng/result/AEC-challenge-2024/ICCRN_E2E_splitAlphaPredictor10_batch8_OneOptimizer/checkpoints/ICCRN.ckpt'
torch.save(network.state_dict(), save_name)
network.load_state_dict(torch.load(save_name))


alpha_predictor = AlphaPredictor(10)
origin_name = '/ddnstor/imu_fanhaipeng/result/AEC-challenge-2024/ICCRN_E2E_splitAlphaPredictor10_batch8_OneOptimizer/checkpoints/net-best.ckpt'
alpha_predictor.load_state_dict(torch.load(origin_name.replace('net-best', 'alpha-best')).state_dict)
save_name = '/ddnstor/imu_fanhaipeng/result/AEC-challenge-2024/ICCRN_E2E_splitAlphaPredictor10_batch8_OneOptimizer/checkpoints/alpha.ckpt'
torch.save(alpha_predictor.state_dict(), save_name)
alpha_predictor.load_state_dict(torch.load(save_name))

mic, _ = sf.read('gen_dataset/tmp/near_echo.wav')
farend, _ = sf.read('gen_dataset/tmp/near.wav')

mic = torch.Tensor(mic).unsqueeze(dim=0)
farend = torch.Tensor(farend).unsqueeze(dim=0)
with torch.no_grad():
    X0 = stft(torch.stack([mic, farend], dim=1))
    mix_comp = torch.stack([X0[:, 0], X0[:, 2]], dim=1)
    far_comp = torch.stack([X0[:, 1], X0[:, 3]], dim=1)
    alpha = alpha_predictor(mix_comp, far_comp, 10)
    far_comp = far_comp * alpha
    input = torch.cat([mix_comp, far_comp], dim=1)
    est_time = network(input)

print(est_time.shape)
