from typing import Optional
import os
from argparse import Namespace
from logging import Logger
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensorboard as tensorboard
from utils import get_loss_fn, get_next_batch, save_ckpt, compute_loss_all_batches, make_gif, make_gif2


def train_model(
    cfg: Namespace,
    args: Namespace,
    data_obj: dict,
    init_epoch: int, 
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    step_unit: str,
    logger: Logger,
    writer: tensorboard.SummaryWriter,
):
    data_name = cfg.data.name.lower()
    model_name = args.model
    start_time = datetime.now()

    # (1) Requires No Encoding. (ex: Heat, Burgers, ODE, etc.)
    if not model.encode:
        num_batches = data_obj['n_train_batches']
        criterion = get_loss_fn(cfg)
        cum_loss = 0.
        for itr in range(init_epoch * num_batches, cfg.train.epochs * num_batches + 1):
            model.train()
            batch = next(data_obj['train_loader'])
            x = batch[0].cuda(cfg.gpu)
            y = batch[1].cuda(cfg.gpu)
            B = x.shape[0]
            T_in = x.shape[1]
            T_out = y.shape[1]

            # CTFNO
            if model_name == 'ctfno':
                if data_name not in ['pv', 'scalarflow']: # CTFNO1d
                    x = x.unsqueeze(1).repeat(1, T_out, 1, 1)
                    t_y = batch[3][0].cuda(cfg.gpu)
                    pred = model(x, t_y) 
                    pred = pred.reshape(B, T_out, x.shape[-1])
                    loss = criterion(pred, y)

                elif data_name == 'pv': # CTFNO1d
                    t_x = batch[2].cuda(cfg.gpu)
                    t_x = torch.cumsum(t_x, dim=1)/64
                    x_ = torch.cat([x,t_x[:,:,None]], dim=-1)
                    x_ = x_.unsqueeze(1).repeat(1, T_out, 1, 1)
                    t_y = batch[3][0].cuda(cfg.gpu)
                    pred = model(x_, t_y)
                    pred = pred.reshape(B, T_out, x.shape[-1])
                    loss = criterion(pred, y)

                else: # CTFNO2d (only for scalarflow dataset)
                    x = x.unsqueeze(1).repeat(1, T_out, 1, 1, 1)
                    t_y = batch[3][0].cuda(cfg.gpu)
                    pred = model(x, t_y)
                    pred = pred.reshape(B, T_out, x.shape[-2], x.shape[-1])
                    loss = criterion(pred, y)

            # NF
            elif model_name == 'nf':
                t_y = batch[3].cuda(cfg.gpu)
                pred = model(x, t_y)
                loss = criterion(pred, y)

            # NODE & ANODE
            elif model_name == 'node' or model_name == 'anode':
                t_y = batch[3][0].cuda(cfg.gpu)
                if model_name == 'anode':
                    t_y = t_y * 0.1 # only for ScalarFlow
                pred = model(x, t_y)
                loss = criterion(pred, y)

            # FNOseq: FNO1d & Autoregressive
            elif model_name == 'fnoseq':
                loss = 0.
                xx = x
                for t in range(T_out):
                    _pred = model(xx)
                    if t == 0:
                        pred = _pred
                    else:
                        pred = torch.cat([pred, _pred], dim=1)
                    xx = _pred
                loss = criterion(pred, y)

            # FNO2d: (x, t) - time as a grid
            elif model_name == 'fno2d':
                x = x.unsqueeze(-1).repeat(1, T_in, 1, T_out) # shape (B, 1, X, T_out)
                pred = model(x) # shape (B, 1, X, T_out)
                pred = pred.transpose(2, 3).squeeze(1) # shape (B, T_out, X)
                loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cum_loss += loss.item()
            if step_unit == 'batch':
                scheduler.step()

            # When an epoch ends...
            if itr % num_batches == 0 and itr > 0:
                epoch = itr // num_batches
                avg_loss = cum_loss / data_obj['n_train_data']
                cum_loss = 0.

                # Logging
                elapsed = datetime.now() - start_time
                logger.info(f'[Epoch {epoch:04d}] Train Loss {avg_loss:.3e} | elapsed {elapsed}')
                writer.add_scalar('train_loss', avg_loss, epoch)

                # Evaluate
                if epoch % cfg.train.eval_freq == 0:
                    eval_model(cfg, args, data_obj, epoch, model, logger, writer)

                # Save (save as ckpt.pt)
                if epoch % cfg.train.save_freq == 0:
                    save_path = os.path.join(args.logdir, 'checkpoints', 'ckpt.pt')
                    save_ckpt(save_path, epoch, model, optimizer, scheduler)

                # Archive (save as ckpt_xxxx.pt)
                if epoch % cfg.train.archive_freq == 0 and epoch > 0:
                    save_path = os.path.join(args.logdir, 'checkpoints', f'ckpt_{epoch}.pt')
                    save_ckpt(save_path, epoch, model, optimizer, scheduler)

                if step_unit == 'epoch':
                    scheduler.step()

    # (2) Requires Encoding. (ex: MuJoCo, PhysioNet)
    else:
        num_batches = data_obj['n_train_batches']
        for itr in range(init_epoch * num_batches, cfg.train.epochs * num_batches + 1):
            model.train()
            optimizer.zero_grad()

            # Scehduling for coefficient of KL-divergence term
            wait_until_kl_inc = 10
            if itr // num_batches < wait_until_kl_inc:
                kl_coef = 0.
            else:
                kl_coef = (1 - 0.99 ** (itr // num_batches - wait_until_kl_inc))

            batch_dict = get_next_batch(data_obj['train_loader'], device=torch.device(f'cuda:{cfg.gpu}'))
            train_result = model.compute_all_losses(batch_dict, n_traj_samples=1, kl_coef=kl_coef, train=True)
            train_result['loss'].backward()
            optimizer.step()
            if step_unit == 'batch':
                scheduler.step()

            # When an epoch ends...
            if itr % num_batches == 0:
                epoch = itr // num_batches

                # Logging
                elapsed = datetime.now() - start_time
                logger.info(f'[Epoch {epoch:04d}] Train Likelihood (one batch) {train_result["likelihood"]:.4e} | elapsed {elapsed}')
                writer.add_scalar(f'train_likelihood (one batch)', train_result["likelihood"], epoch)

                # Evaluate
                if epoch % cfg.train.eval_freq == 0:
                    eval_model(cfg, args, data_obj, epoch, model, logger, writer, kl_coef=kl_coef)

                # Save (save as ckpt.pt)
                if epoch % cfg.train.save_freq == 0:
                    save_path = os.path.join(args.logdir, 'checkpoints', 'ckpt.pt')
                    save_ckpt(save_path, epoch, model, optimizer, scheduler)

                # Archive (save as ckpt_xxxx.pt)
                if epoch % cfg.train.archive_freq == 0 and epoch > 0:
                    save_path = os.path.join(args.logdir, 'checkpoints', f'ckpt_{epoch}.pt')
                    save_ckpt(save_path, epoch, model, optimizer, scheduler)

                if step_unit == 'epoch':
                    scheduler.step()


def eval_model(
    cfg: Namespace,
    args: Namespace,
    data_obj: dict,
    epoch: int,
    model: nn.Module,
    logger: Optional[Logger] = None,
    writer: Optional[tensorboard.SummaryWriter] = None,
    **kwargs,
):
    data_name = cfg.data.name.lower()
    model_name = args.model

    # (1) Requires No Encoding. (ex: Heat, Burgers, ODE)
    if not model.encode:
        num_batches = data_obj['n_test_batches']
        criterion = get_loss_fn(cfg)
        cum_loss = 0.
        model.eval()

        with torch.no_grad():
            for itr in range(num_batches):
                batch = next(data_obj['test_loader'])
                x = batch[0].cuda(cfg.gpu)
                y = batch[1].cuda(cfg.gpu)
                t_y = batch[3].cuda(cfg.gpu)
                B = x.shape[0]
                T_in = x.shape[1]
                T_out = y.shape[1]

                # CTFNO
                if model_name == 'ctfno':
                    if data_name not in ['pv', 'scalarflow']: # CTFNO1d
                        x = x.unsqueeze(1).repeat(1, T_out, 1, 1)
                        t_y = batch[3][0].cuda(cfg.gpu)
                        pred = model(x, t_y) 
                        pred = pred.reshape(B, T_out, x.shape[-1])
                        loss = criterion(pred, y)
                        
                    elif data_name == 'pv': # CTFNO1d
                        t_x = batch[2].cuda(cfg.gpu)
                        t_x = torch.cumsum(t_x, dim=1)/64
                        x_ = torch.cat([x,t_x[:,:,None]], dim=-1)
                        x_ = x_.unsqueeze(1).repeat(1, T_out, 1, 1)
                        t_y = batch[3][0].cuda(cfg.gpu)
                        pred = model(x_, t_y)
                        pred = pred.reshape(B, T_out, x.shape[-1])
                        loss = criterion(pred, y)

                    else: # CTFNO2d (only for scalarflow dataset)
                        x = x.unsqueeze(1).repeat(1, T_out, 1, 1, 1)
                        t_y = t_y[0]
                        pred = model(x, t_y)
                        pred = pred.reshape(B, T_out, x.shape[-2], x.shape[-1])
                        loss = criterion(pred, y)

                # NF
                elif model_name == 'nf':
                    pred = model(x, t_y)
                    loss = criterion(pred, y)

                # NODE & ANODE
                elif model_name == 'node' or model_name == 'anode':
                    t_y = t_y[0]
                    if model_name == 'anode':
                        t_y = t_y * 0.1 # only for ScalarFlow
                    pred = model(x, t_y)
                    loss = criterion(pred, y)

                # FNOseq: FNO1d & Autoregressive
                elif model_name == 'fnoseq':
                    loss = 0.
                    xx = x
                    for t in range(T_out):
                        _pred = model(xx)
                        if t == 0:
                            pred = _pred
                        else:
                            pred = torch.cat([pred, _pred], dim=1)
                        xx = _pred
                    loss = criterion(pred, y)

                # FNO2d: (x, t) - time as a grid
                elif model_name == 'fno2d':
                    x = x.unsqueeze(-1).repeat(1, T_in, 1, T_out) # shape (B, 1, X, T_out)
                    pred = model(x) # shape (B, 1, X, T_out)
                    pred = pred.transpose(2, 3).squeeze(1) # shape (B, T_out, X)
                    loss = criterion(pred, y)

                cum_loss += loss.item()

        avg_loss = cum_loss / data_obj['n_test_data']
        if logger is not None:
            logger.info(f'<Epoch {epoch:04d}> Eval Loss {avg_loss:.3e}')
        if writer is not None:
            writer.add_scalar(f'eval_loss', avg_loss, epoch)

        # Save gif
        if (writer is None) or (epoch % cfg.train.gif_freq == 0):
            idx = np.random.randint(low=0, high=len(y))
            t_y = batch[3]
            save_gif(
                os.path.join(args.logdir, 'gif'),
                t_y[idx].numpy(),
                y[idx].cpu().numpy(),
                pred[idx].cpu().numpy(),
                epoch,
                cfg.data.gif_value_lim,
            )

    # (2) Requires Encoding. (ex: MuJoCo, PhysioNet)
    else:
        with torch.no_grad():
            test_result = compute_loss_all_batches(
                args,
                model,
                data_obj['test_loader'],
                data_obj['n_test_batches'],
                n_traj_samples=1,
                kl_coef=kwargs['kl_coef'],
                device=torch.device(f'cuda:{cfg.gpu}'),
                classify=(cfg.train.latent_loss_type != 'iwae'), # PhysioNet (for AUC), Activity
            )
            message = f'<Epoch {epoch:04d}> TEST MSE {test_result["mse"]:.4e}'
            if logger is not None:
                logger.info(message)
            if writer is not None:
                writer.add_scalar(f'eval_mse', test_result['mse'], epoch)


def save_gif(gif_dir, t, true, pred, epoch, value_lim=None):
    if value_lim == 'none':
        value_lim = None
    # make_gif(true, t, save_path=os.path.join(gif_dir, f'Epoch{epoch}_true.gif'), value_lim=value_lim)
    # make_gif(pred, t, save_path=os.path.join(gif_dir, f'Epoch{epoch}_pred.gif'), value_lim=value_lim)
    make_gif2(true, pred, t, save_path=os.path.join(gif_dir, f'Epoch{epoch}_true_pred.gif'), value_lim=value_lim)

