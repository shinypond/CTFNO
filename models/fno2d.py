import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_act(name):
    if name.lower() == 'relu':
        return nn.ReLU()
    elif name.lower() == 'leakyrelu':
        return nn.LeakyReLU()
    elif name.lower() == 'gelu':
        return nn.GELU()
    elif name.lower() == 'silu':
        return nn.SiLU()
    else:
        raise ValueError(f'Activation function {name} is not supported.')


class SpectralConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, modes1, modes2):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = 1 / (in_ch * out_ch)
        self.weights1 = nn.Parameter(
            self.scale * torch.view_as_real(
                torch.rand(in_ch, out_ch, self.modes1, self.modes2, dtype=torch.cfloat)
            )
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.view_as_real(
                torch.rand(in_ch, out_ch, self.modes1, self.modes2, dtype=torch.cfloat)
            )
        )

    def compl_mul2d(self, input, weights):
        weights = torch.view_as_complex(weights)
        return torch.einsum('bixy,ioxy->boxy', input, weights)

    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        x_ft = torch.fft.rfft2(x, dim=(-2, -1))
        out_ft = torch.zeros(B, self.out_ch, x.shape[-2], x.shape[-1] // 2 + 1, dtype=torch.cfloat).to(x)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )
        x = torch.fft.irfft2(out_ft, s=(x.shape[-2], x.shape[-1]))
        return x


class FNO2d(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        data_ch = cfg.data.ch
        _cfg = cfg.fno2d
        self.modes1 = _cfg.modes[0]
        self.modes2 = _cfg.modes[1]
        self.width = _cfg.width
        self.pad_x_ratio = _cfg.pad_ratio[0]
        self.pad_y_ratio = _cfg.pad_ratio[1]
        self.num_blocks = _cfg.num_blocks
        self.act = _cfg.act

        self.fc0 = nn.Linear(data_ch + 2, self.width)
        modules = []
        for _ in range(self.num_blocks):
            modules.append(SpectralConv2d(self.width, self.width, self.modes1, self.modes2))
            modules.append(nn.Conv2d(self.width, self.width, 1))
        self.main = nn.ModuleList(modules)
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, data_ch)

    def get_grid(self, shape):
        B, X, Y, _ = shape
        gridx = torch.from_numpy(np.linspace(0, 1, X)).type(torch.float32)
        gridx = gridx.reshape(1, X, 1, 1).repeat(B, 1, Y, 1)
        gridy = torch.from_numpy(np.linspace(0, 1, Y)).type(torch.float32)
        gridy = gridy.reshape(1, 1, Y, 1).repeat(B, X, 1, 1)
        return torch.cat([gridx, gridy], dim=-1)

    def forward(self, x: torch.Tensor):
        # x shape : (B, C, X, Y) (If Y-axis is time, C indicates the number of T_in)
        x = x.permute(0, 2, 3, 1)
        grid = self.get_grid(x.shape).to(x)       
        x = torch.cat([x, grid], dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        pad_x = max(1, x.shape[2] // self.pad_x_ratio) # 1024 // 16 -> 64 ---> 1088
        pad_y = max(1, x.shape[3] // self.pad_y_ratio) # 51 // 16 -> 3 ---> 54
        x = F.pad(x, [0, pad_y, 0, pad_x])

        for i in range(self.num_blocks):
            x = self.main[2*i](x) + self.main[2*i+1](x)
            x = get_act(self.act)(x)

        x = x[..., :-pad_x, :-pad_y]
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = get_act(self.act)(x)
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)
        return x