import torch.nn as nn
import torch

class AlphaPredictor(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.linear1 = nn.Linear(2, 1)
        self.linear2 = nn.Linear(k, 1)
        # self.ReLU = nn.ReLU()

    def forward(self, mix_comp, far_comp, k):
        # shape: B, C, F, T
        far_complex = far_comp
        mix_complex = mix_comp
        far_complex_padded = torch.nn.functional.pad(far_complex, (k - 1, 0))
        # 执行 unfold 操作，设置 size 参数为 (F, 4)，即 T+3 维度展开为大小为 4 的窗口
        far_complex_unfolded = far_complex_padded.unfold(3, k, 1)
        mix_complex_padded = torch.nn.functional.pad(mix_complex, (k - 1, 0))
        mix_complex_unfolded = mix_complex_padded.unfold(3, k, 1)

        # 等价于 abs(X) ** 2
        pow_far = far_complex_unfolded[:, 0] ** 2 + far_complex_unfolded[:, 1] ** 2
        pow_mix = mix_complex_unfolded[:, 0] ** 2 + mix_complex_unfolded[:, 1] ** 2

        input = torch.stack([pow_far, pow_mix], dim=-1).unsqueeze(dim=1)
        out = self.linear1(torch.sum(input, dim=2, keepdim=True)).squeeze(dim=-1)
        out = self.linear2(out).squeeze(dim=-1)
        # out = self.ReLU(out)
        out = torch.abs(out)
        return out

if __name__ == '__main__':
    from thop import profile, clever_format

    inputs = torch.randn(1, 2, 161, 100).cuda()
    model = AlphaPredictor(10).cuda()
    total_ops, total_params = profile(model, inputs=(inputs, inputs, 10), verbose=False)
    flops, params = clever_format([total_ops, total_params], "%.3f ")
    print(flops, params)