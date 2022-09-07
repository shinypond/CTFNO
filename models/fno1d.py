import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import init_network_weights


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


class SpectralConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, modes1):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.modes1 = modes1

        self.scale = 1 / (in_ch * out_ch)
        self.weights1 = nn.Parameter(
            self.scale * torch.view_as_real(
                torch.rand(in_ch, out_ch, self.modes1, dtype=torch.cfloat)
            )
        )

    def compl_mul1d(self, input, weights):
        weights = torch.view_as_complex(weights)
        return torch.einsum('bix,iox->box', input, weights)

    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        x_ft = torch.fft.rfft(x, dim=-1)
        out_ft = torch.zeros(B, self.out_ch, x.shape[-1] // 2 + 1, dtype=torch.cfloat).to(x)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)
        x = torch.fft.irfft(out_ft, n=x.shape[-1])
        return x


class FNO1d(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        data_ch = cfg.data.ch
        _cfg = cfg.fno1d
        self.modes1 = _cfg.modes[0]
        self.width = _cfg.width
        self.pad_x_ratio = _cfg.pad_ratio[0]
        self.num_blocks = _cfg.num_blocks
        self.act = _cfg.act

        self.fc0 = nn.Linear(data_ch + 1, self.width)
        modules = []
        for _ in range(self.num_blocks):
            modules.append(SpectralConv1d(self.width, self.width, self.modes1))
            modules.append(nn.Conv1d(self.width, self.width, 1))
        self.main = nn.ModuleList(modules)
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, data_ch)

        init_network_weights(self.fc0, std=0.01)
        init_network_weights(self.fc1, std=0.01)
        init_network_weights(self.fc2, std=0.01)

    def get_grid(self, shape):
        B, X, _ = shape
        grid = torch.from_numpy(np.linspace(0, 1, X)).type(torch.float32)
        grid = grid.reshape(1, X, 1).repeat(B, 1, 1)
        return grid

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 1)
        grid = self.get_grid(x.shape).to(x)       
        x = torch.cat([x, grid], dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        pad_x = max(1, x.shape[2] // self.pad_x_ratio)
        x = F.pad(x, [0, pad_x])

        for i in range(self.num_blocks):
            x = self.main[2*i](x) + self.main[2*i+1](x)
            x = get_act(self.act)(x)

        x = x[..., :-pad_x]
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = get_act(self.act)(x)
        x = self.fc2(x)
        x = x.permute(0, 2, 1)
        return x


        
