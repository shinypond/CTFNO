import torch
import torch.nn as nn
import stribor as st


class CouplingFlow(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        _cfg = cfg.nf
        input_dim = cfg.data.x_resolution
        n_layers = _cfg.n_layers
        n_hidden = [_cfg.n_hidden_dims] * _cfg.n_hidden_layers
        time_net = getattr(st.net, _cfg.time_net)
        time_hidden = _cfg.time_hidden
        act = _cfg.activation
        final_act = _cfg.final_activation

        transforms = []
        for i in range(n_layers):
            transforms.append(
                st.ContinuousAffineCoupling(
                    latent_net=st.net.MLP(input_dim+1, n_hidden, 2*input_dim, activation=act, final_activation=final_act),
                    time_net=time_net(2*input_dim, hidden_dim=time_hidden),
                    mask='none',
                )
            )

        self.flow = st.NeuralFlow(transforms=transforms)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        B, _, X = x.shape
        B, T = t.shape
        x = x.repeat(1, T, 1)
        t = t.unsqueeze(-1)
        return self.flow(x, t=t)
