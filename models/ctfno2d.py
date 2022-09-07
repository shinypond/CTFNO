import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np


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


class KernelEmbedding2d(nn.Module):
    def __init__(self, temb, ch, midch, modes1, modes2, heads=1):
        super().__init__()
        assert ch % heads == 0
        self.temb = temb
        self.modes1 = modes1
        self.modes2 = modes2
        self.ch = ch
        self.heads = heads
        self.midch = midch
        
        scale = math.sqrt(2 / (midch + 4 * modes1 * modes2))
        self.weights = Parameter(scale * torch.randn(heads, midch//heads, 4*modes1*modes2, dtype=torch.float32))
        self.bias = Parameter(scale * torch.zeros(1, 1, 4*modes1*modes2, dtype=torch.float32))
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
        temb_modes = temb_modes.reshape(T, self.ch, self.modes1, self.modes2, 2, 2)
        return torch.view_as_complex(temb_modes) # shape (T, C, M1, M2, 2)


class ConvEmbedding(nn.Module):
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
class SpectralReducedTimeConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, modes1, modes2, emb_heads, mid_channels, temb, weight_heads, c):
        super().__init__()
        assert in_ch % weight_heads == 0 and out_ch % weight_heads == 0
        
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.weight_heads = weight_heads
        self.modes1 = modes1
        self.modes2 = modes2
        self.c = c

        scale = 1 / (in_ch * out_ch)
        # mode R
        self.weights1 = Parameter(
            scale * torch.view_as_real(
                torch.rand(in_ch//weight_heads, out_ch//weight_heads, weight_heads, modes1, modes2, dtype=torch.cfloat)
            )
        )
        self.weights2 = Parameter(
            scale * torch.view_as_real(
                torch.rand(in_ch//weight_heads, out_ch//weight_heads, weight_heads, modes1, modes2, dtype=torch.cfloat)
            )
        )
        # channel R
        self.weights3 = Parameter(
            scale * torch.view_as_real(torch.rand(in_ch, out_ch, dtype=torch.cfloat))
        )
        # time embedding
        self.temb = KernelEmbedding2d(temb, in_ch, mid_channels, modes1, modes2, emb_heads)
        self.conv = ConvEmbedding(temb, in_ch, mid_channels)
        
    def get_Norm1(self, w_temb, weights):
        # w_temb : shape (T, I/H, H, M1, M2)
        # weights: shape (I/H, O/H, H, M1, M2)
        norm = torch.einsum('tihxy,iohxy->tohxy', torch.abs(w_temb), torch.abs(weights)) + 1e-12
        norm = norm.reshape(norm.shape[0], -1, self.modes1, self.modes2)
        norm = torch.relu(norm / self.c - 1) + 1.
        return norm # shape (T, O, M1, M2)

    def get_Norm2(self, w_temb, weights):
        # w_temb : shape (T, I)
        # weights : shape (I, O)
        norm = torch.einsum('ti,io->to', torch.abs(w_temb), torch.abs(weights))
        norm = torch.relu(norm / self.c - 1) + 1.
        return norm[:, :, None, None] # shape (T, O, 1, 1)
    
    def compl_mul2d1(self, input, weights, w_temb):
        # input : shape (B, T, I, M1, M2)
        # w_temb : shape (T, I, M1, M2)
        B, T = input.shape[0], input.shape[1]
        input = input.reshape(B, T, -1, self.weight_heads, self.modes1, self.modes2)
        w_temb = w_temb.reshape(T, -1, self.weight_heads, self.modes1, self.modes2)
        weights = torch.view_as_complex(weights)
        out = input * w_temb.unsqueeze(0)
        out = torch.einsum('btihxy,iohxy->btohxy', out, weights)
        out = out.reshape(B, T, -1, self.modes1, self.modes2)
        norm = self.get_Norm1(w_temb, weights)
        out = out / norm.unsqueeze(0)
        return out # shape (B, T, O, M1, M2)
    
    def compl_mul2d2(self, input, weights, w_temb):
        # input : shape (B, T, I, M1, M2)
        # w_temb : shape (T, I)
        weights = torch.view_as_complex(weights) 
        out = input * w_temb[None, :, :, None, None]
        out = torch.einsum('btixy,io->btoxy', out, weights)
        norm = self.get_Norm2(w_temb, weights)
        out = out / norm.unsqueeze(0)
        return out # shape (B, T, O, M1, M2)
    
    def forward(self, x, t):
        B = x.shape[0]
        T = t.shape[0]
        temb_modes = self.temb(t)
        temb_modes1 = temb_modes[:, :, :, :, 0] # shape (T, C, M1, M2)
        temb_modes2 = temb_modes[:, :, :, :, 1] # shape (T, C, M1, M2)
        temb_modes3 = torch.view_as_complex(self.conv(t)) # shape (T, C)
        
        x_ft = torch.fft.rfft2(x)

        out_ft1 = torch.zeros(B, T, self.out_ch, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat).to(x)
        out_ft1[:, :, :, :self.modes1, :self.modes2]  = self.compl_mul2d1(x_ft[:, :, :, :self.modes1, :self.modes2], self.weights1, temb_modes1)
        out_ft1[:, :, :, -self.modes1:, :self.modes2] = self.compl_mul2d1(x_ft[:, :, :, -self.modes1:, :self.modes2], self.weights2, temb_modes2)
        x = torch.fft.irfft2(out_ft1, s=(x.size(-2), x.size(-1)))

        out_ft2 = torch.zeros(B, T, self.out_ch,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft2  = self.compl_mul2d2(x_ft, self.weights3, temb_modes3)
        x = x + torch.fft.irfft2(out_ft2, s=(x.size(-2), x.size(-1)))
        
        return x


class TimeConv2d_1x1(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch, temb, c):
        super().__init__()
        self.c = c
        self.scale = 1 / (in_ch * out_ch)
        self.weights = Parameter(self.scale * torch.rand(1, in_ch, out_ch, dtype=torch.float32))
        self.temb = ConvEmbedding(temb, in_ch, mid_ch)

    def get_Norm(self, w):
        norm = torch.norm(w, p=1, dim=1) + 1e-12
        norm = torch.relu(norm / self.c - 1) + 1.
        return norm
    
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        temb = self.temb(t) 
        temb_weights, temb_bias = temb[:, :, :1], temb[:, :, 1]
        w = temb_weights * self.weights
        out = torch.einsum("btixy,tio->btoxy", x, w)
        norm = self.get_Norm(w)
        out = out / norm[None, :, :, None, None] + temb_bias[None, :, :, None, None]
        return out # shape (B, T, O, X, Y)


class Conv2d_1x1(nn.Module):
    def __init__(self, in_ch, out_ch, c):
        super().__init__()
        self.c = c
        self.scale = math.sqrt(1 / in_ch)
        self.weights = Parameter(self.scale * torch.randn(in_ch, out_ch, dtype=torch.float32))
        self.bias = Parameter(self.scale * torch.zeros(1, 1, out_ch, 1, 1, dtype=torch.float32))

    def get_Norm(self, w):
        norm = torch.norm(w, p=1, dim=0) + 1e-12
        norm = torch.relu(norm / self.c - 1) + 1.
        return norm
    
    def forward(self, x: torch.Tensor):
        out = torch.einsum("btixy,io->btoxy", x, self.weights)
        norm = self.get_Norm(self.weights)
        out = out / norm[None, None, :, None, None] + self.bias
        return out # shape (B, T, O, X, Y)


############################################
# [Main] CTFNO 
############################################
class CTFNO2d(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        data_ch = cfg.data.ch
        _cfg = cfg.ctfno2d
        modes1 = _cfg.modes[0]
        modes2 = _cfg.modes[1]
        self.num_blocks = _cfg.num_blocks
        self.width = _cfg.width
        self.pad_x_ratio = _cfg.pad_ratio[0]
        self.pad_y_ratio = _cfg.pad_ratio[1]
        c = _cfg.C
        midch = _cfg.mid_ch
        freqch = _cfg.freq_ch
        emb_heads = _cfg.emb_heads
        self.act = _cfg.act
        weight_heads = _cfg.weight_heads
        
        self.fc0 = Conv2d_1x1(data_ch+2, self.width, c)
        temb1 = TimeEmbedding(freqch, midch)
        temb2 = TimeEmbedding(freqch, midch)
        modules = []
        for _ in range(self.num_blocks):
            modules.append(
                SpectralReducedTimeConv2d(
                    self.width, self.width, modes1, modes2, emb_heads, midch, temb1, weight_heads, c
                )
            )
            modules.append(
                TimeConv2d_1x1(
                    self.width, self.width, midch, temb2, c
                )
            )
        self.Fmodules = nn.ModuleList(modules)
        
        self.fc1 = Conv2d_1x1(self.width, 1, c)
        
    def forward(self, x: torch.Tensor, t: torch.Tensor):

        # x : shape (B, T, C, X, Y)
        # t : shape (T,)

        # Lifting Layer
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=2)
        x = self.fc0(x)
        
        # Padding
        pad_x = max(1, x.shape[3] // self.pad_x_ratio)
        pad_y = max(1, x.shape[4] // self.pad_y_ratio)
        x = F.pad(x, [0, pad_y, 0, pad_x])
        
        # Main Layer
        for i in range(self.num_blocks):
            x1 = self.Fmodules[2*i](x, t)
            x2 = self.Fmodules[2*i+1](x, t)
            x = x1 + x2
            x = get_act(self.act)(x)
        
        # Unpadding
        x = x[..., :-pad_x, :-pad_y]
        
        # Projection Layer
        x = self.fc1(x)
        
        return x

    def get_grid(self, shape, device):
        B, size_t, size_x, size_y = shape[0], shape[1], shape[3], shape[4]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, 1, 1, size_x, 1).repeat([B, size_t, 1, 1, size_y])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, 1, 1, size_y).repeat([B, size_t, 1, size_x, 1])
        return torch.cat((gridx, gridy), dim=2).to(device) # shape (B, T, 2, X, Y)