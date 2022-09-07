import os
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from utils import get_data_obj, load_config, load_and_copy_config, get_logger, load_ckpt, get_optim, get_sched
from get_model import get_model
from runner import train_model, eval_model


def main(args):
    # Fix a random seed
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    # Logging directory
    os.makedirs('logs/', exist_ok=True)
    model_enc = f'{args.model}' if args.encoder is None else f'{args.model}_{args.encoder}'
    args.logdir = os.path.join('logs', f'{args.data}_{model_enc}_{args.exp_name}')
    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(os.path.join(args.logdir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(args.logdir, 'tensorboard'), exist_ok=True)
    os.makedirs(os.path.join(args.logdir, 'gif'), exist_ok=True)

    # Run
    if args.mode == 'train':

        # Load config 
        if args.resume:
            cfg = load_config(os.path.join(args.logdir, 'cfg_copy.yml'))
        else:
            cfg = load_and_copy_config(f'configs/{args.data}.yml', os.path.join(args.logdir, 'cfg_copy.yml'))

        # Logger
        logger = get_logger(os.path.join(args.logdir, 'train_log.log'), os.path.abspath(__file__))

        # Tensorboard Writer
        tb_path = os.path.join(args.logdir, 'tensorboard')
        writer = SummaryWriter(log_dir=tb_path)

        # Load model
        model = get_model(cfg, args.model, args.encoder)
        logger.info(datetime.now())
        logger.info(f'Model Params: {sum(p.numel() for p in model.parameters()) / 1024 / 1024:.4f}M')

        # Optimizer, Scheduler
        optimizer = get_optim(cfg, model)
        scheduler, step_unit = get_sched(cfg, optimizer)

        # Load checkpoint
        if args.resume:
            ckpt_path = os.path.join(args.logdir, 'checkpoints', 'ckpt.pt')
            init_epoch, model, optimizer, scheduler = load_ckpt(
                ckpt_path, model, optimizer, scheduler, gpu_num=cfg.gpu, mode='train',
            )
        else:
            init_epoch = 0

        # Load data object
        data_obj = get_data_obj(cfg)

        # Train
        train_model(cfg, args, data_obj, init_epoch, model, optimizer, scheduler, step_unit, logger, writer)

    elif args.mode == 'eval':

        # Load config
        cfg = load_config(os.path.join(args.logdir, 'cfg_copy.yml'))

        # Load model
        model = get_model(cfg, args.model, args.encoder)

        # Load checkpoint
        ckpt_path = os.path.join(args.logdir, 'checkpoints', 'ckpt.pt')
        epoch, model = load_ckpt(
            ckpt_path, model, None, None, gpu_num=cfg.gpu, mode='eval',
        )

        # Load data object
        data_obj = get_data_obj(cfg)

        # Evaluate
        eval_model(cfg, args, data_obj, epoch, model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode', type=str, default='train', choices=['train', 'eval'],
        help='Choose train or eval. Default to train.',
    )
    parser.add_argument(
        '--data', type=str, required=True,
        help='Dataset name. Available datasets are described in README.md',
    )
    parser.add_argument(
        '--exp-name', type=str, required=True,
        help='Experiment name. Required when making a logging directory.',
    )
    parser.add_argument(
        '--model', type=str, required=True, choices=['ctfno', 'nf', 'lode', 'node', 'anode', 'fnoseq', 'fno2d'],
        help='Model name for experiments. Available models for each dataset are described in README.md',
    )
    parser.add_argument(
        '--encoder', type=str, default=None, choices=['rnn', 'nfrnn', 'odernn'],
        help='Encoder name for MuJoCo/PhysioNet experiments. Available encoders for each dataset are described in README.md',
    )
    parser.add_argument(
        '--random-seed', type=int, default=1997, 
        help='Random seed.',
    )
    parser.add_argument(
        '--resume', action='store_true',
        help='Resume training',
    )

    args = parser.parse_args()
    main(args)