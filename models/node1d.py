import torch
import torch.nn as nn
from torchdiffeq import odeint
from utils import init_network_weights


#####################################
# Neural ODE  
#####################################
class NODE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        input_dim = cfg.data.x_resolution
        _cfg = cfg.node
        n_layers = _cfg.n_layers
        n_hidden = _cfg.n_hidden
        ode_func = ODEFunc(
            n_input=input_dim,
            n_output=input_dim,
            n_layers=n_layers,
            n_hidden=n_hidden,
            nonlinear=nn.Tanh
        )
        self.diffeq_solver = DiffeqSolver(ode_func, ode_method='euler')

    def forward(self, x: torch.Tensor, time_steps_to_predict: torch.Tensor):
        return self.diffeq_solver(x, time_steps_to_predict)


#####################################
# DiffeqSolver
#####################################
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

    def forward(self, x: torch.Tensor, time_steps_to_predict: torch.Tensor):
        B, _, X = x.shape
        T, = time_steps_to_predict.shape
        pred = odeint(
            self.ode_func,
            x,
            time_steps_to_predict,
            rtol=self.rtol,
            atol=self.atol,
            method=self.ode_method,
        )
        pred = pred.transpose(0, 1).squeeze(2)
        assert pred.shape == (B, T, X)
        # assert torch.mean(pred[:, 0, :] - x) < 1e-3
        return pred


#####################################
# ODE FuncNet 
#####################################
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
        init_network_weights(self.gradient_net, std=0.01)

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
