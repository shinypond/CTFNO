import torch
import torch.nn as nn
from torchdiffeq import odeint
from utils import init_network_weights


#####################################
# Neural ODE  
#####################################
class ANODE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        input_dim = cfg.data.ch
        _cfg = cfg.anode
        aug_dim = _cfg.aug
        n_layers = _cfg.n_layers
        n_hidden = _cfg.n_hidden
        ode_func = ODEFunc(
            n_input=input_dim,
            n_aug=aug_dim,
            n_output=input_dim,
            n_layers=n_layers,
            n_hidden=n_hidden,
            nonlinear=nn.Tanh,
        )
        self.diffeq_solver = DiffeqSolver(input_dim, ode_func, ode_method='euler', n_aug=aug_dim)

    def forward(self, x: torch.Tensor, time_steps_to_predict: torch.Tensor):
        B, _, X, Y = x.shape
        return self.diffeq_solver(x, time_steps_to_predict)


#####################################
# DiffeqSolver
#####################################
class DiffeqSolver(nn.Module):
    def __init__(
        self,
        input_dim,
        ode_func,
        ode_method,
        n_aug,
        rtol=1e-4,
        atol=1e-4,
    ):
        super().__init__()
        self.ode_func = ode_func
        self.ode_method = ode_method
        self.rtol = rtol
        self.atol = atol
        self.n_aug = n_aug
        self.linear = nn.Linear(input_dim+n_aug, 1)

    def forward(self, x: torch.Tensor, time_steps_to_predict: torch.Tensor):
        B, _, X, Y = x.shape
        if self.n_aug:
            aug_shape = (B, self.n_aug, X, Y)
            aug_x = torch.zeros(aug_shape).to(x.device)
            aug_x = torch.cat([x, aug_x], dim=1) # (B, 1+aug, X, Y)
        
        T, = time_steps_to_predict.shape
        pred = odeint(
            self.ode_func,
            aug_x,
            time_steps_to_predict,
            rtol=self.rtol,
            atol=self.atol,
            method=self.ode_method,
        )
        
        pred = pred.permute(1, 0, 3, 4, 2) # (B, T, X, Y, C)
        pred = self.linear(pred).squeeze(-1)

        assert pred.shape == (B, T, X, Y)
        return pred


#####################################
# ODE FuncNet 
#####################################
class ODEFunc(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_aug: int,
        n_output: int,
        n_layers: int,
        n_hidden: int,
        nonlinear: nn.Module,
    ):
        super().__init__()
        
        modules = [nn.Conv2d(n_input+n_aug, n_hidden, kernel_size=1, padding=0), nonlinear()]
        
        for _ in range(n_layers):
            modules.append(nn.Conv2d(n_hidden, n_hidden, kernel_size=3, padding=1))
            modules.append(nonlinear())
        modules.append(nn.Conv2d(n_hidden, n_output+n_aug, kernel_size=1, padding=0))
        
        self.gradient_net = nn.Sequential(*modules)
        
        init_network_weights(self.gradient_net)

    def forward(self, t_local, y, backwards=False):
        grad = self.gradient_net(y)
        if backwards:
            return -grad
        else:
            return grad
