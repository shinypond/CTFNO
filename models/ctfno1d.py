import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np

################################################################
# Activation
################################################################
def get_act(name):
    if name.lower() == 'relu': return nn.ReLU()
    if name.lower() == 'gelu': return nn.GELU()
    if name.lower() == 'silu': return nn.SiLU()


################################################################
# Time & Freq Embedding
################################################################
class TimeEmbedding1d(nn.Module):
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
        T = t.shape[0]
        emb = self.emb.to(t.device)
        t = t[:,None]*emb[None,:] # (T, ch//2)
        t = torch.cat([torch.sin(t), torch.cos(t)], dim=1) # (T, freq_ch)
        temb_ch = self.mlp(t) # (T, midch)
        return temb_ch 


class KernelEmbedding1d(nn.Module):
    def __init__(self, temb, ch, midch, modes1, heads=1):
        super().__init__()
        assert ch%heads == 0
        self.temb = temb
        self.modes1 = modes1
        self.ch = ch
        self.heads = heads
        self.midch = midch
        scale = 1 / (midch * modes1)
        self.weights = Parameter(scale*torch.rand(heads, midch//heads, modes1*2, dtype=torch.float32))
        self.bias = Parameter(scale*torch.zeros(1, 1, modes1*2, dtype=torch.float32))
        
        self.initialize()
        
    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
                
    def forward(self, t):
        temb_modes = self.temb(t) # (T, midch)
        T = t.shape[0]
        temb_modes = temb_modes.reshape(T,self.heads, self.midch//self.heads) # (T, head, midch//head)
        temb_modes = torch.einsum("thc,hcm->thm", temb_modes, self.weights) # (T, head, 2*mode)
        temb_modes = temb_modes.repeat(1, self.ch//self.heads, 1) + self.bias # (T, ch, 2*mode)
        temb_modes = temb_modes.reshape(T, self.ch, self.modes1, 2)
        return torch.view_as_complex(temb_modes)


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
        temb_ch = self.temb(t) # (T, midch)
        T = t.shape[0]
        temb_modes = self.mlp(temb_ch).reshape(T, self.ch, 2) # (T, ch, 2)
        return temb_modes
        

################################################################
# Fourier Convolution Networks
################################################################
class SpectralFreqTimeConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, heads, mid_channels, temb, c):
        super(SpectralFreqTimeConv1d, self).__init__()
        self.out_channels = out_channels
        self.modes1 = modes1
        self.c = c
        self.scale = 1/(in_channels*out_channels) 
        self.weights1 = Parameter(self.scale * torch.view_as_real(torch.rand(in_channels, out_channels, modes1, dtype=torch.cfloat)))
        
        self.temb = KernelEmbedding1d(temb, in_channels, mid_channels, modes1, heads)
        
    def get_Norm(self, w_temb, weights):
        T, _, M = w_temb.shape
        norm = torch.einsum('bix,iox->box',torch.abs(w_temb),torch.abs(weights)) + 1e-12
        norm = torch.relu(norm / self.c - 1) + 1.
        return norm # (T,C,M)

    def compl_mul1d_freq(self, input, weights, w_temb):
        """Complex multiplication with time"""
        # input : (B, T, in_channel, modes1)
        B, T, C, M = input.shape
        weights = torch.view_as_complex(weights) # (in_channel, out_channel)
        out = input * w_temb[None, :, :, :] # (B, T, in_channel, modes1)
        out = out.reshape(B, T, C, M) # (B, T, in_channels, modes1)
        out = torch.einsum('btix,iox->btox', out, weights) # (B, T, out_channel, modes1)
        norm = self.get_Norm(w_temb, weights) # (T, out_channel, modes1)
        out = out / norm[None, :, :, :]
        return out # (B, T, out_channel, modes1)
    
    def forward(self, x, t):
        B = x.shape[0]
        T = t.shape[0]
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(B, T, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        temb_modes = self.temb(t)
        out_ft[:, :, :, :self.modes1] = self.compl_mul1d_freq(x_ft[:, :, :, :self.modes1], self.weights1, temb_modes)
        x1 = torch.fft.irfft(out_ft, n=x.size(-1))
        return x1


class TimeConv1d_1x1(nn.Module):
    def __init__(self, in_channels, out_channels, mid_ch, temb, c):
        super().__init__()
        self.c = c
        self.scale = 1 / (in_channels * out_channels)
        self.weights = Parameter(self.scale * torch.rand(1, in_channels, out_channels, dtype=torch.float32))
        self.temb = ConvEmbedding1d(temb, in_channels, mid_ch)

    def get_Norm(self, w):
        norm = torch.norm(w, p=1, dim=1) + 1e-12
        norm = torch.relu(norm / self.c - 1) + 1.
        return norm
    
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        temb = self.temb(t) # (T, ch, 2)
        temb_weights, temb_bias = temb[:, :, :1], temb[:, :, 1]
        w = temb_weights * self.weights # (T, in_channel, out_channel)
        out = torch.einsum("btix,tio->btox", x, w)
        norm = self.get_Norm(w)
        out = out / norm[None, :, :, None] + temb_bias[None, :, :, None]
        return out


class Conv1d_1x1(nn.Module):
    def __init__(self, in_channels, out_channels, c):
        super().__init__()
        self.c = c
        self.scale = math.sqrt(1 / in_channels)
        self.weights = Parameter(self.scale * torch.randn(in_channels, out_channels, dtype=torch.float32))
        self.bias = Parameter(self.scale * torch.zeros(1, 1, out_channels, 1, dtype=torch.float32))

    def get_Norm(self, w):
        norm = torch.norm(w, p=1, dim=0) + 1e-12
        norm = torch.relu(norm / self.c - 1) + 1.
        return norm
    
    def forward(self, x: torch.Tensor):
        out = torch.einsum("btix,io->btox", x, self.weights)
        norm = self.get_Norm(self.weights)
        out = out/norm[None, None, :, None] + self.bias
        return out


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
        norm = torch.sum(torch.abs(weights), dim=0) 
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


################################################################
# Fourier layers
################################################################
class CTFNO1d(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        data_ch = cfg.data.ch
        _cfg = cfg.ctfno1d
        self.act = _cfg.act
        modes1 = _cfg.modes[0]
        self.num_blocks = _cfg.num_blocks
        self.width = _cfg.width
        self.pad_x_ratio = _cfg.pad_ratio[0]
        c = _cfg.C
        midch = _cfg.mid_ch
        freqch = _cfg.freq_ch
        num_heads = _cfg.emb_heads

        self.preprocess, self.postprocess = get_process(cfg)
        
        data_name = cfg.data.name.lower()
        if data_name in ['pv', 'hopper', 'physionet']:
            self.fc0 = nn.Sequential(
                Conv1d_1x1(data_ch+1, self.width, c),
                get_act(self.act),
                SpectralConv1d_1x1(self.width, self.width, modes1, c),
            )
        else:
            self.fc0 = Conv1d_1x1(data_ch+1, self.width, c)
        
        modules = []
        
        temb1 = TimeEmbedding1d(freqch, midch)
        temb2 = TimeEmbedding1d(freqch, midch)
        
        for _ in range(self.num_blocks):
            modules.append(SpectralFreqTimeConv1d(self.width, self.width, modes1, num_heads, midch, temb1, c))
            modules.append(TimeConv1d_1x1(self.width, self.width, midch, temb2, c))
        self.Fmodules = nn.ModuleList(modules)
        
        self.fc1 = Conv1d_1x1(self.width, 1, c)
        
    def forward(self, x: torch.Tensor, t: torch.Tensor):

        # Preprocess Layer
        x = self.preprocess(x)

        # Lifting Layer
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=2)
        x = self.fc0(x)

        # Padding
        pad_x = max(1, x.shape[-1] // self.pad_x_ratio)
        x = F.pad(x, [0, pad_x])

        # Main Layer
        for i in range(self.num_blocks):
            x = self.Fmodules[2*i](x, t) + self.Fmodules[2*i+1](x, t)
            x = get_act(self.act)(x)

        # Unpadding
        x = x[..., :-pad_x]

        # Projection Layer
        x = self.fc1(x)

        # Postprocess Layer
        x = self.postprocess(x)

        return x

    def get_grid(self, shape, device):
        batchsize, size_t, size_x, = shape[0], shape[1], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, 1, 1, size_x).repeat([batchsize, size_t, 1, 1])
        return gridx.to(device)


def get_process(cfg):
    _cfg = cfg.ctfno1d
    if hasattr(_cfg, 'process'):
        if _cfg.process.lower() == 'linear':
            preprocess = Linear_PreProcess(cfg)
            postprocess = Linear_PostProcess(cfg)    
        else:
            raise ValueError(f'Process {_cfg.process} is not supported.')
    else:
        preprocess = nn.Identity()
        postprocess = nn.Identity()
    return preprocess, postprocess


class Linear_PreProcess(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.input_dim = cfg.encoder.input_dim
        self.latent_dim = cfg.encoder.latent_dim
        self.linear = nn.Linear(self.input_dim+1, self.latent_dim)
        
    def forward(self, x):
        x = self.linear(x)
        return x


class Linear_PostProcess(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.input_dim = cfg.encoder.input_dim
        self.latent_dim = cfg.encoder.latent_dim
        self.linear = nn.Linear(self.latent_dim, self.input_dim)
        
    def forward(self, x):
        x = self.linear(x)
        return x
