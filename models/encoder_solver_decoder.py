##########################################
# Referred to (Latent ODE & NeuralFlow)
# https://github.com/YuliaRubanova/latent_ode
# https://github.com/mbilos/stribor
##########################################
from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.rnn import GRU
from torchdiffeq import odeint
import stribor as st
from utils import split_last_dim, init_network_weights, check_mask, reverse, linspace_vector
from .ctfno1d import CTFNO1d


##########################################
# Auxiliary classes 
##########################################
class TimeEmbedding(nn.Module):
    """Time Embedding Layer"""
    def __init__(self, ch, tdim):
        super().__init__()
        half_dim = ch // 2
        emb = np.log(10000) / (half_dim - 1)
        self.emb = torch.exp(-emb * torch.arange(half_dim, dtype=torch.float32))
        
        self.mlp = nn.Sequential(
            nn.Linear(ch, ch*4),
            nn.SiLU(),
            nn.Linear(ch*4, tdim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, t: torch.Tensor):
        emb = self.emb.to(t.device)
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        emb = self.mlp(emb) # (B, dim)
        return emb


class GRU_unit(nn.Module):
    def __init__(
        self,
        latent_dim,
        input_dim, 
        update_gate=None,
        reset_gate=None,
        new_state_net=None,
        n_units=100,
    ):
        super(GRU_unit, self).__init__()

        if update_gate is None:
            self.update_gate = nn.Sequential(
               nn.Linear(latent_dim * 2 + input_dim, n_units),
               nn.Tanh(),
               nn.Linear(n_units, latent_dim),
               nn.Sigmoid(),
            )
            init_network_weights(self.update_gate)
        else: 
            self.update_gate = update_gate

        if reset_gate is None:
            self.reset_gate = nn.Sequential(
               nn.Linear(latent_dim * 2 + input_dim, n_units),
               nn.Tanh(),
               nn.Linear(n_units, latent_dim),
               nn.Sigmoid(),
            )
            init_network_weights(self.reset_gate)
        else: 
            self.reset_gate = reset_gate

        if new_state_net is None:
            self.new_state_net = nn.Sequential(
               nn.Linear(latent_dim * 2 + input_dim, n_units),
               nn.Tanh(),
               nn.Linear(n_units, latent_dim * 2),
            )
            init_network_weights(self.new_state_net)
        else: 
            self.new_state_net = new_state_net


    def forward(self, y_mean, y_std, x, masked_update=True):

        y_concat = torch.cat([y_mean, y_std, x], dim=-1)

        update_gate = self.update_gate(y_concat)
        reset_gate = self.reset_gate(y_concat)
        concat = torch.cat([y_mean * reset_gate, y_std * reset_gate, x], dim=-1)
        
        new_state, new_state_std = split_last_dim(self.new_state_net(concat))
        new_state_std = new_state_std.abs()

        new_y = (1 - update_gate) * new_state + update_gate * y_mean
        new_y_std = (1 - update_gate) * new_state_std + update_gate * y_std

        assert not torch.isnan(new_y).any()

        if masked_update:
            # IMPORTANT: assumes that x contains both data and mask
            # update only the hidden states for hidden state only if at least one feature is present for the current time point
            n_data_dims = x.size(-1) // 2
            mask = x[:, :, n_data_dims:]
            check_mask(x[:, :, :n_data_dims], mask)
            
            mask = (torch.sum(mask, -1, keepdim = True) > 0).float()

            assert not torch.isnan(mask).any()

            new_y = mask * new_y + (1 - mask) * y_mean
            new_y_std = mask * new_y_std + (1 - mask) * y_std

            if torch.isnan(new_y).any():
                print("new_y is nan!")
                print(mask)
                print(y_mean)
                print(new_y)
                exit()

        new_y_std = new_y_std.abs()
        return new_y, new_y_std


##########################################
# Encoders 
##########################################
class Encoder_z0_RNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.input_dim = cfg.encoder.input_dim
        self.rec_dim = cfg.encoder.rec_dim
        self.latent_dim = cfg.encoder.latent_dim
        self.use_delta_t = True
        enc_input_dim = 2 * self.input_dim

        self.hiddens_to_z0 = nn.Sequential(
           nn.Linear(self.rec_dim, 100),
           nn.Tanh(),
           nn.Linear(100, self.latent_dim * 2),
        )
        init_network_weights(self.hiddens_to_z0)
        if self.use_delta_t:
            enc_input_dim += 1
        self.gru_rnn = GRU(enc_input_dim, self.rec_dim)

    def forward(self, data, time_steps, run_backwards=True):
        # IMPORTANT: assumes that 'data' already has mask concatenated to it 
        # data shape: [n_traj, n_tp, n_dims]
        # shape required for rnn: (seq_len, batch, input_size)

        n_traj = data.size(0)

        assert not torch.isnan(data).any()
        assert not torch.isnan(time_steps).any()

        data = data.permute(1, 0, 2) 

        if run_backwards:
            # Look at data in the reverse order: from later points to the first
            data = reverse(data)

        if self.use_delta_t:
            delta_t = time_steps[1:] - time_steps[:-1]
            if run_backwards:
                # we are going backwards in time with
                delta_t = reverse(delta_t)
            # append zero delta t in the end
            delta_t = torch.cat([delta_t, torch.zeros(1).to(delta_t)], dim=0)
            delta_t = delta_t.unsqueeze(1).repeat(1, n_traj).unsqueeze(-1)
            data = torch.cat([delta_t, data], dim=-1)

        outputs, _ = self.gru_rnn(data)

        # GRU output shape: (seq_len, batch, num_directions * hidden_size)
        last_output = outputs[-1]

        mean, std = self.hiddens_to_z0(last_output).chunk(2, dim=-1)
        std = F.softplus(std)

        assert not torch.isnan(mean).any()
        assert not torch.isnan(std).any()

        return mean.unsqueeze(0), std.unsqueeze(0)


class Encoder_z0_ODERNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.input_dim = cfg.encoder.input_dim
        self.rec_dim = cfg.encoder.rec_dim
        self.latent_dim = cfg.encoder.latent_dim
        _cfg = cfg.odernn
        self.rec_layers = _cfg.rec_layers
        self.hidden_units = _cfg.hidden_units

        self.GRU_update = GRU_unit(
            self.rec_dim,
            2*self.input_dim, 
            n_units=self.hidden_units, 
        )

        ode_func = ODEFunc(
            n_input=self.rec_dim,
            n_output=self.rec_dim,
            n_layers=self.rec_layers,
            n_hidden=cfg.node.hidden_units,
            nonlinear=nn.Tanh,
        )
        self.z0_diffeq_solver = DiffeqSolver(
            ode_func,
            'euler',
            rtol=1e-3,
            atol=1e-4,
        )

        self.transform_z0 = nn.Sequential(
           nn.Linear(self.rec_dim * 2, 100),
           nn.Tanh(),
           nn.Linear(100, self.latent_dim * 2),
        )
        init_network_weights(self.transform_z0)

    def forward(self, data, time_steps, run_backwards=True):
        # data, time_steps -- observations and their time stamps
        # IMPORTANT: assumes that 'data' already has mask concatenated to it 
        assert not torch.isnan(data).any()
        assert not torch.isnan(time_steps).any()

        n_traj, n_tp, n_dims = data.size()

        if len(time_steps) == 1:
            prev_y = torch.zeros(1, n_traj, self.rec_dim).to(data)
            prev_std = torch.zeros(1, n_traj, self.rec_dim).to(data)
            xi = data[:, 0, :].unsqueeze(0)
            last_yi, last_yi_std = self.GRU_update(prev_y, prev_std, xi)

        else:
            last_yi, last_yi_std, _ = self.run_odernn(
                data,
                time_steps,
                run_backwards=run_backwards,
            )

        means_z0 = last_yi.reshape(1, n_traj, self.rec_dim)
        std_z0 = last_yi_std.reshape(1, n_traj, self.rec_dim)

        mean_z0, std_z0 = self.transform_z0(torch.cat([means_z0, std_z0], dim=-1)).chunk(2, dim=-1)
        std_z0 = F.softplus(std_z0)

        return mean_z0, std_z0

    def run_odernn(self, data, time_steps, run_backwards=True):
        # IMPORTANT: assumes that 'data' already has mask concatenated to it 
        n_traj, n_tp, n_dims = data.size()

        prev_y = torch.zeros(1, n_traj, self.rec_dim).to(data)
        prev_std = torch.zeros(1, n_traj, self.rec_dim).to(data)

        prev_t, t_i = time_steps[-1] + 0.01, time_steps[-1]

        interval_length = time_steps[-1] - time_steps[0]
        minimum_step = interval_length / 50

        assert not torch.isnan(data).any()
        assert not torch.isnan(time_steps).any()

        latent_ys = []

        # Run ODE backwards and combine the y(t) estimates using gating
        time_points_iter = range(0, len(time_steps))
        if run_backwards:
            time_points_iter = reversed(time_points_iter)

        for i in time_points_iter:
            if prev_t - t_i < minimum_step:
                time_points = torch.stack([prev_t, t_i], dim=0)
                inc = (t_i - prev_t) * self.z0_diffeq_solver.ode_func(prev_t, prev_y)

                assert not torch.isnan(inc).any()

                ode_sol = prev_y + inc
                ode_sol = torch.stack((prev_y, ode_sol), dim=2).to(data)

                assert not torch.isnan(ode_sol).any()
            else:
                n_intermediate_tp = max(2, ((prev_t - t_i) / minimum_step).int())

                time_points = linspace_vector(prev_t, t_i, n_intermediate_tp)
                ode_sol = self.z0_diffeq_solver(prev_y, time_points)

                assert not torch.isnan(ode_sol).any()

            if torch.mean(ode_sol[:, :, 0, :]  - prev_y) >= 0.001:
                print("Error: first point of the ODE is not equal to initial value")
                print(torch.mean(ode_sol[:, :, 0, :]  - prev_y))
                exit()

            yi_ode = ode_sol[:, :, -1, :]
            xi = data[:, i, :].unsqueeze(0)
            
            yi, yi_std = self.GRU_update(yi_ode, prev_std, xi)

            prev_y, prev_std = yi, yi_std			
            prev_t, t_i = time_steps[i], time_steps[i-1]

            latent_ys.append(yi)

        latent_ys = torch.stack(latent_ys, 1)

        assert not torch.isnan(yi).any()
        assert not torch.isnan(yi_std).any()

        return yi, yi_std, latent_ys


class Encoder_z0_NFRNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.input_dim = cfg.encoder.input_dim
        self.rec_dim = cfg.encoder.rec_dim
        self.latent_dim = cfg.encoder.latent_dim
        _cfg = cfg.nfrnn

        self.gru = nn.GRUCell(2*self.input_dim, self.rec_dim)

        self.z0_diffeq_solver = CouplingFlow(
            self.rec_dim,
            _cfg.layers,
            [_cfg.hidden_dim] * _cfg.hidden_layers,
            _cfg.time_net,
            _cfg.time_hidden_dim,
        )
        self.transform_z0 = nn.Sequential(
           nn.Linear(self.rec_dim, 100),
           nn.Tanh(),
           nn.Linear(100, self.latent_dim * 2),
        )
        init_network_weights(self.transform_z0)

    def forward(self, data, time_steps, run_backwards=True):
        assert(not torch.isnan(data).any())
        assert(not torch.isnan(time_steps).any())

        n_traj, n_tp, n_dims = data.size()

        latent = self.run_odernn(data, time_steps, run_backwards)

        latent = latent.reshape(1, n_traj, self.rec_dim)

        mean_z0, std_z0 = self.transform_z0(latent).chunk(2, dim=-1)
        std_z0 = F.softplus(std_z0)

        return mean_z0, std_z0

    def run_odernn(self, data, time_steps, run_backwards=True):
        batch_size, n_tp, n_dims = data.size()
        prev_t, t_i = time_steps[-1] + 0.01, time_steps[-1]

        time_points_iter = range(0, time_steps.shape[0])
        if run_backwards:
            time_points_iter = reversed(time_points_iter)

        h = torch.zeros(batch_size, self.rec_dim).to(data)

        for i in time_points_iter:
            dt = (t_i - prev_t).unsqueeze(0)

            dt = dt.repeat(h.shape[0]).unsqueeze(-1)
            h = self.z0_diffeq_solver(h, dt)

            xi = data[:, i, :]
            h_ = self.gru(xi, h)
            mask = self.get_mask(xi)

            h = mask * h_ + (1 - mask) * h

            prev_t, t_i = time_steps[i], time_steps[i-1]

        return h

    def get_mask(self, x):
        x = x.unsqueeze(0)
        n_data_dims = x.size(-1) // 2
        mask = x[:, :, n_data_dims:]
        check_mask(x[:, :, :n_data_dims], mask)
        mask = (torch.sum(mask, dim=-1, keepdim=True) > 0).float()
        assert not torch.isnan(mask).any()
        return mask.squeeze(0)


##########################################
# Solvers 
##########################################
class Solver_CTFNO(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.solver = CTFNO1d(cfg)
        self.t_multiplier = float(cfg.ctfno1d.t_multiplier)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        assert len(t.shape) == 1
        Traj, B, Z = x.shape
        T = t.shape[0]
        x = x.reshape(-1, 1, 1, Z).repeat(1, T, 1, 1) 
        t = t * self.t_multiplier # for better time embedding
        y = self.solver(x, t)
        y = y.squeeze(-2) # remove channel dim
        y = y.reshape(Traj, B, T, Z)
        return y


class Solver_NODE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.latent_dim = cfg.encoder.latent_dim
        _cfg = cfg.node
        self.gen_layers = _cfg.gen_layers
        self.hidden_units = _cfg.hidden_units
        ode_func = ODEFunc(
            n_input=self.latent_dim,
            n_output=self.latent_dim,
            n_layers=self.gen_layers,
            n_hidden=self.hidden_units,
            nonlinear=nn.Tanh,
        )
        self.solver = DiffeqSolver(
            ode_func,
            'dopri5',
            rtol=1e-3,
            atol=1e-4,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        return self.solver(x, t)


class Solver_NF(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.latent_dim = cfg.encoder.latent_dim
        _cfg = cfg.nf
        self.solver = CouplingFlow(
            self.latent_dim,
            _cfg.layers,
            [_cfg.hidden_dim] * _cfg.hidden_layers,
            _cfg.time_net,
            _cfg.time_hidden_dim, 
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        assert len(t.shape) == 1
        x = x.unsqueeze(-2) # (Traj, B, 1, Z)
        t = t[None, None, :, None].repeat(x.shape[0], x.shape[1], 1, 1) # (Traj, B, T, 1)
        y = self.solver(x, t)
        return y


##########################################
# Decoder
##########################################
class Decoder(nn.Module):
    # Decode data from latent space where we are solving an ODE back to the data space
    def __init__(self, cfg):
        super().__init__()
        self.input_dim = cfg.encoder.input_dim
        self.latent_dim = cfg.encoder.latent_dim
        decoder = nn.Sequential(
           nn.Linear(self.latent_dim, self.input_dim),
        )

        init_network_weights(decoder)	
        self.decoder = decoder

    def forward(self, data):
        return self.decoder(data)


##########################################
# LatentODE (only for MuJoco & PhysioNet)
##########################################
class DiffeqSolver(nn.Module):
    def __init__(
        self,
        ode_func,
        ode_method,
        rtol=1e-4,
        atol=1e-4,
    ):
        super().__init__()
        self.ode_func = ode_func
        self.ode_method = ode_method
        self.rtol = rtol
        self.atol = atol

    def forward(self, first_point: torch.Tensor, time_steps_to_predict: torch.Tensor):
        Traj, B = first_point.shape[0], first_point.shape[1]
        T, = time_steps_to_predict.shape
        pred = odeint(
            self.ode_func,
            first_point,
            time_steps_to_predict,
            rtol=self.rtol,
            atol=self.atol,
            method=self.ode_method,
        )
        pred = pred.permute(1, 2, 0, 3)
        assert torch.mean(pred[:, :, 0, :] - first_point) < 1e-3
        assert pred.shape[0] == Traj
        assert pred.shape[1] == B
        return pred


class ODEFunc(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_layers: int,
        n_hidden: int,
        nonlinear: nn.Module,
    ):
        super().__init__()
        self.gradient_net = self.create_ode_func_net(
            n_input, n_output, n_layers, n_hidden, nonlinear,
        )
        init_network_weights(self.gradient_net)

    def create_ode_func_net(self, n_input, n_output, n_layers, n_hidden, nonlinear):
        layers = [nn.Linear(n_input, n_hidden)]
        for _ in range(n_layers):
            layers.append(nonlinear())
            layers.append(nn.Linear(n_hidden, n_hidden))
        layers.append(nonlinear())
        layers.append(nn.Linear(n_hidden, n_output))
        return nn.Sequential(*layers)

    def forward(self, t_local, y, backwards=False):
        grad = self.gradient_net(y)
        if backwards:
            return -grad
        else:
            return grad


##########################################
# NeuralFlow (only for MuJoco & PhysioNet)
##########################################
class CouplingFlow(nn.Module):
    def __init__(
        self,
        dim: int,
        n_layers: int,
        hidden_dims: List[int],
        time_net: nn.Module,
        time_hidden_dim: Optional[int] = None,
        **kwargs
    ):
        super().__init__()

        transforms = []
        for i in range(n_layers):
            transforms.append(st.ContinuousAffineCoupling(
                latent_net=st.net.MLP(dim + 1, hidden_dims, 2 * dim),
                time_net=getattr(st.net, time_net)(2 * dim, hidden_dim=time_hidden_dim),
                mask='none' if dim == 1 else f'ordered_{i % 2}'))

        self.flow = st.NeuralFlow(transforms=transforms)

    def forward(
        self,
        x: torch.Tensor, # Initial conditions, (..., 1, dim)
        t: torch.Tensor, # Times to solve at, (..., seq_len, dim)
        t0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor: # Solutions to IVP given x at t, (..., times, dim)

        if x.shape[-2] == 1:
            x = x.repeat_interleave(t.shape[-2], dim=-2) # (..., 1, dim) -> (..., seq_len, 1)

        # If t0 not 0, solve inverse first
        if t0 is not None:
            x = self.flow.inverse(x, t=t0)[0]

        return self.flow(x, t=t)

