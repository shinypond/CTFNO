from models.ctfno1d import CTFNO1d
from models.ctfno2d import CTFNO2d
from models.fno1d import FNO1d
from models.fno2d import FNO2d
from models.nf import CouplingFlow
from models.anode2d import ANODE
from models.node1d import NODE
from models.encoder_solver_decoder import Encoder_z0_RNN, Encoder_z0_NFRNN, Encoder_z0_ODERNN
from models.encoder_solver_decoder import Solver_CTFNO, Solver_NF, Solver_NODE, Decoder
from models.latent_baseline import LatentODE


def get_model(cfg, model_name, encoder_name=None):
    # (1) Requires No Encoding. (ex: Heat, Burgers, ODE, PV, ScalarFlow)
    if encoder_name is None:

        data_name = cfg.data.name.lower()

        if model_name == 'ctfno':
            if data_name != 'scalarflow':
                model = CTFNO1d(cfg)
            else:
                model = CTFNO2d(cfg)

        elif model_name == 'nf':
            assert data_name in ['heat', 'burgers', 'ode']
            model = CouplingFlow(cfg)

        elif model_name == 'node':
            assert data_name in ['heat', 'burgers', 'ode'] 
            model = NODE(cfg)

        elif model_name == 'anode':
            assert data_name in ['scalarflow']
            model = ANODE(cfg)

        elif model_name == 'fnoseq':
            assert data_name in ['heat', 'burgers']
            model = FNO1d(cfg)

        elif model_name == 'fno2d':
            assert data_name in ['heat', 'burgers']
            model = FNO2d(cfg)

        else:
            raise ValueError(f'Model {model_name} is not supported.')

        setattr(model, 'encode', False)

    # (2) Requires Encoding. (ex: MuJoCo, PhysioNet)
    else:
        # (a) Encoder (input to latent)
        if encoder_name == 'rnn':
            assert model_name in ['ctfno', 'nf', 'lode']
            encoder = Encoder_z0_RNN(cfg)

        elif encoder_name == 'nfrnn':
            assert model_name == 'nf'
            encoder = Encoder_z0_NFRNN(cfg)

        elif encoder_name == 'odernn':
            assert model_name == 'lode'
            encoder = Encoder_z0_ODERNN(cfg)

        else:
            raise ValueError(f'Encoder {encoder_name} is not supported.')

        # (b) Diffeq_Solver (solve ODE in the latent space)
        if model_name == 'ctfno':
            solver = Solver_CTFNO(cfg)

        elif model_name == 'nf':
            solver = Solver_NF(cfg)

        elif model_name == 'lode':
            solver = Solver_NODE(cfg)

        else:
            raise ValueError(f'Model {model_name} is not supported.')

        # (c) Decoder (latent to output)
        decoder = Decoder(cfg)

        # LatentODE (as a wrapper)
        model = LatentODE(cfg, encoder=encoder, solver=solver, decoder=decoder)

        setattr(model, 'encode', True)

    model = model.cuda(cfg.gpu)
    return model 

