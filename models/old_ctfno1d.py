import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


def get_act(name):
    if name.lower() == 'relu': return nn.ReLU()
    if name.lower() == 'gelu': return nn.GELU()
    if name.lower() == 'silu': return nn.SiLU()


############################################
# Embedding
############################################
class TimeEmbedding(nn.Module):
    def __init__(self, freq_ch, midch):
        super().__init__()        
        half_dim = freq_ch // 2
        emb = np.log(10000) / (half_dim - 1)
        self.emb = torch.exp(-emb * torch.arange(half_dim, dtype=torch.float32)) # ch//2
        self.mlp = nn.Sequential(
            nn.Linear(freq_ch, midch),
            nn.SiLU(),
            nn.Linear(midch, midch),
            nn.SiLU(),
            nn.Linear(midch, midch),
            nn.SiLU(),
        )
        self.initialize()
        
    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, t):
        emb = self.emb.to(t)
        t = t[:, None] * emb[None, :]
        t = torch.cat([torch.sin(t), torch.cos(t)], dim=1)
        temb_ch = self.mlp(t)
        return temb_ch # shape (T, midch)


class KernelEmbedding1d(nn.Module):
    def __init__(self, temb, ch, midch, modes1, heads=1):
        super().__init__()
        assert ch % heads == 0
        self.temb = temb
        self.modes1 = modes1
        self.ch = ch
        self.heads = heads
        self.midch = midch
        scale = 1 / (midch * modes1)
        self.weights = Parameter(scale * torch.rand(heads, midch//heads, 2*modes1, dtype=torch.float32))
        self.bias = Parameter(scale * torch.zeros(1, 1, 2*modes1, dtype=torch.float32))
        self.initialize()
        
    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
                
    def forward(self, t):
        temb_modes = self.temb(t)
        T = t.shape[0]
        temb_modes = temb_modes.reshape(T, self.heads, self.midch//self.heads)
        temb_modes = torch.einsum("thc,hcm->thm", temb_modes, self.weights) 
        temb_modes = temb_modes.repeat(1, self.ch//self.heads, 1) + self.bias
        temb_modes = temb_modes.reshape(T, self.ch, self.modes1, 2)
        return torch.view_as_complex(temb_modes) # shape (T, C, M1)


class ConvEmbedding1d(nn.Module):
    def __init__(self, temb, ch, midch):
        super().__init__()
        self.temb = temb
        self.ch = ch
        self.mlp = nn.Linear(midch, 2*ch)
        self.initialize()
        
    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, t):
        temb_ch = self.temb(t)
        T = t.shape[0]
        temb_modes = self.mlp(temb_ch).reshape(T, self.ch, 2)
        return temb_modes # shape (T, C, 2)


############################################
# Spectral / Conv Blocks 
############################################
class SpectralFreqTimeConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, modes1, emb_heads, mid_channels, temb, c):
        super().__init__()

        self.out_ch = out_ch
        self.modes1 = modes1
        self.c = c
        self.scale = 1 / (in_ch * out_ch) 
        self.weights1 = Parameter(self.scale * torch.view_as_real(torch.rand(in_ch, out_ch, modes1, dtype=torch.cfloat)))
        self.temb = KernelEmbedding1d(temb, in_ch, mid_channels, modes1, emb_heads)
        
    def get_Norm(self, w_temb, weights):
        # w_temb : shape (T, I, M1)
        # weights : shape (I, O, M1)
        norm = torch.einsum('tix,iox->tox', torch.abs(w_temb), torch.abs(weights)) + 1e-12
        norm = torch.relu(norm / self.c - 1) + 1.
        return norm # shape (T, O, M1)

    def compl_mul1d_freq(self, input, weights, w_temb):
        B, T, C, M = input.shape
        weights = torch.view_as_complex(weights)
        out = input * w_temb[None, :, :, :]
        out = out.reshape(B, T, C, M) 
        out = torch.einsum('btix,iox->btox', out, weights) 
        norm = self.get_Norm(w_temb, weights)
        out = out / norm[None, :, :, :]
        return out # shape (B, T, O, M1)
    
    def forward(self, x, t):
        B = x.shape[0]
        T = t.shape[0]
        temb_modes = self.temb(t) # shape (T, C, M1)

        x_ft = torch.fft.rfft(x)

        out_ft = torch.zeros(B, T, self.out_ch, x.size(-1)//2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :, :self.modes1] = self.compl_mul1d_freq(x_ft[:, :, :, :self.modes1], self.weights1, temb_modes)

        out = torch.fft.irfft(out_ft, n=x.size(-1))

        return out


class TimeConv1d_1x1(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch, temb, c):
        super().__init__()
        self.c = c
        self.scale = 1 / (in_ch * out_ch)
        self.weights = Parameter(self.scale * torch.rand(1, in_ch, out_ch, dtype=torch.float32))
        self.temb = ConvEmbedding1d(temb, in_ch, mid_ch)

    def get_Norm(self, w):
        norm = torch.norm(w, p=1, dim=1) + 1e-12
        norm = torch.relu(norm / self.c - 1) + 1.
        return norm
    
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        temb = self.temb(t)
        temb_weights = temb[:, :, :1]
        temb_bias = temb[:, :, 1]
        w = temb_weights * self.weights
        out = torch.einsum("btix,tio->btox", x, w)
        norm = self.get_Norm(w)
        out = out / norm[None, :, :, None] + temb_bias[None, :, :, None]
        return out # shape (B, T, O, X)


class Conv1d_1x1(nn.Module):
    def __init__(self, in_ch, out_ch, c):
        super().__init__()
        self.c = c
        self.scale = math.sqrt(1 / in_ch)
        self.weights = Parameter(self.scale * torch.randn(in_ch, out_ch, dtype=torch.float32))
        self.bias = Parameter(self.scale * torch.zeros(1, 1, out_ch, 1, dtype=torch.float32))

    def get_Norm(self, w):
        norm = torch.norm(w, p=1, dim=0) + 1e-12
        norm = torch.relu(norm / self.c - 1) + 1.
        return norm
    
    def forward(self, x: torch.Tensor):
        out = torch.einsum("btix,io->btox", x, self.weights)
        norm = self.get_Norm(self.weights)
        out = out / norm[None, None, :, None] + self.bias
        return out # shape (B, T, O, X, Y)


class SpectralConv1d_1x1(nn.Module):
    def __init__(self, in_ch, out_ch, modes1, c):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.modes1 = modes1
        self.c = c
        self.scale = 1 / (in_ch * out_ch)
        self.weights1 = Parameter(self.scale * torch.view_as_real(torch.rand(in_ch, out_ch, dtype=torch.cfloat)))
    
    def get_Norm(self, weights):
        norm = torch.sum(torch.abs(weights), dim=0)  #
        norm = torch.relu(norm / self.c - 1) + 1.
        return norm # shape (T, O, M1)
    
    def compl_mul1d_freq(self, input, weights):
        # input : shape (B, T, I, M1)
        w = torch.view_as_complex(weights)
        out = torch.einsum('btix,io->btox', input, w)
        norm = self.get_Norm(w)
        out = out / norm[None, None, :, None]
        return out # (B, T, O, M1)
    
    def forward(self, x):
        B, T = x.shape[0], x.shape[1]
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(B, T, self.out_ch, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :, :self.modes1] = self.compl_mul1d_freq(x_ft[:, :, :, :self.modes1], self.weights1)
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


############################################
# [Main] CTFNO 
############################################
class CTFNO1d(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        data_ch = cfg.data.ch
        _cfg = cfg.ctfno1d
        modes1 = _cfg.modes[0]
        self.num_blocks = _cfg.num_blocks
        self.width = _cfg.width
        self.pad_x_ratio = _cfg.pad_ratio[0]
        c = _cfg.C
        midch = _cfg.mid_ch
        freqch = _cfg.freq_ch
        emb_heads = _cfg.emb_heads
        self.act = _cfg.act
        
        data_name = cfg.data.name.lower()
        if data_name in ['pv', 'hopper', 'physionet']:
            self.fc0 = nn.Sequential(
                Conv1d_1x1(data_ch+1, self.width, c),
                get_act(self.act),
                SpectralConv1d_1x1(self.width, self.width, modes1, c),
            )
        else:
            self.fc0 = Conv1d_1x1(data_ch+1, self.width, c)
        temb1 = TimeEmbedding(freqch, midch)
        temb2 = TimeEmbedding(freqch, midch)
        modules = []
        for _ in range(self.num_blocks):
            modules.append(SpectralFreqTimeConv1d(self.width, self.width, modes1, emb_heads, midch, temb1, c))
            modules.append(TimeConv1d_1x1(self.width, self.width, midch, temb2, c))
        self.Fmodules = nn.ModuleList(modules)
        
        self.fc1 = Conv1d_1x1(self.width, data_ch, c)
        
    def forward(self, x: torch.Tensor, t: torch.Tensor):

        # x : shape (B, T, C, X)
        # t : shape (T,)

        # Lifting Layer
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=2)
        x = self.fc0(x)
        
        # Padding
        pad_x = max(1, x.shape[3] // self.pad_x_ratio)
        x = F.pad(x, [0, pad_x])
        
        # Main Layer
        for i in range(self.num_blocks):
            x = self.Fmodules[2*i](x, t) + self.Fmodules[2*i+1](x, t)
            x = get_act(self.act)(x)
        
        # Unpadding
        x = x[..., :-pad_x]
        
        # Projection Layer
        x = self.fc1(x)
        
        return x

    def get_grid(self, shape, device):
        B, size_t, size_x, = shape[0], shape[1], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, 1, 1, size_x).repeat([B, size_t, 1, 1])
        return gridx.to(device) # shape (B, T, 1, X)

