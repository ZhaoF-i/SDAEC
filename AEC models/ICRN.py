import torch.nn as nn
import torch

class convGLU(nn.Module):
    def __init__(self, in_channels=None, out_channels=None, kernel_size=(5, 1), stride=(1, 1),
                 padding=(2, 0), dilation=1, groups=1, bias=True):
        super(convGLU, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
    def forward(self, inputs):
        outputs = self.conv(inputs)
        return outputs


class convTransGLU(nn.Module):
    def __init__(self, in_channels=None, out_channels=None, kernel_size=(5, 1), stride=(1, 1),
                 padding=(2, 0), output_padding=(0, 0), dilation=1, groups=1, bias=True):
        super(convTransGLU, self).__init__()
        self.convTrans = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                            stride=stride, padding=padding, output_padding=output_padding,
                                            dilation=dilation, groups=groups, bias=bias)
    def forward(self, inputs):
        outputs = self.convTrans(inputs)
        return outputs

class ICRN(nn.Module):
    def __init__(self, in_ch=4, out_ch=2, channels=24, lstm_hiden=48):
        super(ICRN, self).__init__()
        self.act = nn.ELU()
        self.sigmod = nn.Sigmoid()

        self.e1 = convGLU(in_ch, channels)
        self.e2 = convGLU(channels, channels)
        self.e3 = convGLU(channels, channels)
        self.e4 = convGLU(channels, channels)
        self.e5 = convGLU(channels, channels)
        self.e6 = convGLU(channels, channels)


        self.d26 = convTransGLU(2 * channels, channels)
        self.d25 = convTransGLU(2 * channels, channels)
        self.d24 = convTransGLU(2 * channels, channels)
        self.d23 = convTransGLU(2 * channels, channels)
        self.d22 = convTransGLU(2 * channels, channels)
        self.d21 = convTransGLU(2 * channels, 2)

        self.BNe1 = nn.BatchNorm2d(channels)
        self.BNe2 = nn.BatchNorm2d(channels)
        self.BNe3 = nn.BatchNorm2d(channels)
        self.BNe4 = nn.BatchNorm2d(channels)
        self.BNe5 = nn.BatchNorm2d(channels)
        self.BNe6 = nn.BatchNorm2d(channels)


        self.BNd26 = nn.BatchNorm2d(channels)
        self.BNd25 = nn.BatchNorm2d(channels)
        self.BNd24 = nn.BatchNorm2d(channels)
        self.BNd23 = nn.BatchNorm2d(channels)
        self.BNd22 = nn.BatchNorm2d(channels)
        self.BNd21 = nn.BatchNorm2d(2)

        self.lstm = nn.LSTM(channels, lstm_hiden,
                            num_layers=2, batch_first=True, bidirectional=False)

        self.linear_lstm_out = nn.Linear(lstm_hiden, channels)
    def forward(self, mix_comp, far_comp):

        inputs = torch.cat((mix_comp, far_comp), 1).permute(0,1,3,2)

        e1 = self.act(self.BNe1(self.e1(inputs)))
        e2 = self.act(self.BNe2(self.e2(e1)))
        e3 = self.act(self.BNe3(self.e3(e2)))
        e4 = self.act(self.BNe4(self.e4(e3)))
        e5 = self.act(self.BNe5(self.e5(e4)))
        e6 = self.act(self.BNe6(self.e6(e5)))

        shape_in = e6.shape
        lstm_in = e6.permute(0, 2, 3, 1).reshape(-1, shape_in[3], shape_in[1])
        lstm_out, _ = self.lstm(lstm_in.float())
        lstm_out = self.linear_lstm_out(lstm_out)
        lstm_out = lstm_out.reshape(shape_in[0], shape_in[2], shape_in[3], shape_in[1]).permute(0, 3, 1, 2)

        d26 = self.act(self.BNd26(self.d26(torch.cat([e6, lstm_out], dim=1))))
        d25 = self.act(self.BNd25(self.d25(torch.cat([e5, d26], dim=1))))
        d24 = self.act(self.BNd24(self.d24(torch.cat([e4, d25], dim=1))))
        d23 = self.act(self.BNd23(self.d23(torch.cat([e3, d24], dim=1))))
        d22 = self.act(self.BNd22(self.d22(torch.cat([e2, d23], dim=1))))
        out = self.act(self.BNd21(self.d21(torch.cat([e1, d22], dim=1))))

        out = out.permute(0, 1, 3, 2)  # [B,C,T,F]
        return out, far_comp

def vector_unitization(x):
    amp = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True) + 1e-10)
    out = x / (amp + 1e-10)
    return out

