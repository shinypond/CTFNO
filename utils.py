from typing import Optional, List
import os
import argparse
import logging
import yaml
import shutil
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
logging.getLogger('matplotlib.animation').disabled = True
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from einops import repeat
from mujoco_physics import HopperPhysics
from physionet import PhysioNet, variable_time_collate_fn, get_data_min_max


#######################################
# Logging tools
#######################################
def get_logger(logpath, filepath, debug=False):
    logger = logging.getLogger()

    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logger.setLevel(level)
    info_file_handler = logging.FileHandler(logpath, mode='w')
    info_file_handler.setLevel(level)
    logger.addHandler(info_file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    logger.info(filepath)

    return logger


#######################################
# Save / Load tools
#######################################
def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def load_config(config_path):
    with open(config_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    cfg = dict2namespace(cfg_dict)
    return cfg


def load_and_copy_config(config_path, copy_path):
    cfg = load_config(config_path)
    shutil.copy(config_path, copy_path)
    return cfg


def save_ckpt(save_path, epoch, model, optimizer, scheduler):
    '''Save checkpoint at save_path'''
    ckpt = {
        'epoch': int(epoch),
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'sched': scheduler.state_dict(),
    }
    torch.save(ckpt, save_path)


def load_ckpt(load_path, model, optimizer, scheduler, gpu_num=0, mode='train'):
    '''Load checkpoint from load_path'''
    # Load checkpoint
    try:
        ckpt = torch.load(load_path, map_location=torch.device(f'cuda:{gpu_num}'))
    except:
        print('No checkpoint is found.')
        return 0, model, optimizer, scheduler
    
    # For training
    if mode == 'train':
        init_epoch = ckpt['epoch'] + 1
        print(f'Checkpoint is found! Start from epoch {init_epoch}')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optim'])
        scheduler.load_state_dict(ckpt['sched'])
        return init_epoch, model, optimizer, scheduler

    # For evaluation
    elif mode == 'eval':
        epoch = ckpt['epoch']
        print(f'Checkpoint is found! Epoch is {epoch}')
        model.load_state_dict(ckpt['model'])
        return epoch, model

    else:
        raise ValueError(f'Mode {mode} is not supported.')


#######################################
# Loss function tools
#######################################
def get_loss_fn(cfg):
    _loss_fn = cfg.train.loss_fn
    if _loss_fn.lower() == 'rel_l2':
        return Relative_Lp_error(p=2, reduction='sum')
    elif _loss_fn.lower() == 'mse':
        return MSE()
    else:
        raise ValueError(f'Loss function {_loss_fn} is not supported.')


class MSE:
    def __init__(self):
        self.mse = nn.MSELoss(reduction='none')

    def __call__(self, pred, true):
        B, T = pred.shape[0], pred.shape[1]
        return self.mse(pred, true).reshape(B, T, -1).mean(2).mean(1).sum(0)


class Relative_Lp_error:
    def __init__(self, p=2, reduction='sum'):
        assert p > 0
        self.p = p
        self.reduction = reduction

    def __call__(self, pred: torch.Tensor, true: torch.Tensor):
        B, T = pred.shape[0], pred.shape[1]
        assert B == true.shape[0]
        assert T == true.shape[1]

        diff_norms = torch.norm(pred.reshape(B, -1) - true.reshape(B, -1), p=self.p, dim=1)
        true_norms = torch.norm(true.reshape(B, -1), p=self.p, dim=1)

        if self.reduction == None:
            return diff_norms / true_norms
        elif self.reduction == 'mean':
            return torch.mean(diff_norms / true_norms)
        elif self.reduction == 'sum':
            return torch.sum(diff_norms / true_norms)
        else:
            raise ValueError(f'Reduction option {self.reduction} is invalid.')


def compute_loss_all_batches(
    args,
    model,
    test_dataloader,
    n_batches,
    n_traj_samples=1,
    kl_coef=1., 
    device=torch.device('cpu'),
    classify=False,
):

    total = {
        'loss': 0,
        'likelihood': 0,
        'mse': 0,
        'kl_first_p': 0,
        'std_first_p': 0,
    }

    n_test_batches = 0
    classif_predictions = torch.Tensor([]).to(device)
    all_test_labels = torch.Tensor([]).to(device)
    
    for i in range(n_batches):
        
        batch_dict = get_next_batch(test_dataloader, device=device)

        results = model.compute_all_losses(batch_dict, n_traj_samples=n_traj_samples, kl_coef=kl_coef, train=False)

        if classify:
            n_labels = batch_dict['labels'].size(-1)
            n_traj_samples = results['label_predictions'].size(0)
            classif_predictions = torch.cat((
                classif_predictions,
                results['label_predictions'].reshape(n_traj_samples, -1, n_labels)
            ), dim=1)
            all_test_labels = torch.cat((
                all_test_labels,
                batch_dict['labels'].reshape(-1, n_labels)
            ), dim=0)

        for key in total.keys(): 
            if key in results:
                var = results[key]
                if isinstance(var, torch.Tensor):
                    var = var.detach()
                total[key] += var

        n_test_batches += 1

    if n_test_batches > 0:
        for key, value in total.items():
            total[key] = value / n_test_batches

    if classify:
        if args.data == 'physionet':
            all_test_labels = all_test_labels.repeat(n_traj_samples, 1, 1)
            idx_not_nan = ~torch.isnan(all_test_labels)
            classif_predictions = classif_predictions[idx_not_nan]
            all_test_labels = all_test_labels[idx_not_nan]
            total['auc'] = 0.
            if torch.sum(all_test_labels) != 0:
                print(f'Number of labeled examples: {len(all_test_labels.reshape(-1))}')
                print(f'Number of examples with mortality 1: {torch.sum(all_test_labels == 1)}')
                total['auc'] = roc_auc_score(
                    all_test_labels.cpu().numpy().reshape(-1),
                    classif_predictions.cpu().numpy().reshape(-1),
                )
            else:
                print("Warning: Couldn't compute AUC -- all examples are from the same class.")

        if args.data == 'activity':
            all_test_labels = all_test_labels.repeat(n_traj_samples, 1, 1)
            labeled_tp = torch.sum(all_test_labels, -1) > 0
            all_test_labels = all_test_labels[labeled_tp]
            classif_predictions = classif_predictions[labeled_tp]
            _, pred_class_id = torch.max(classif_predictions, -1)
            _, class_labels = torch.max(all_test_labels, -1)
            pred_class_id = pred_class_id.reshape(-1)
            total['accuracy'] = accuracy_score(
                class_labels.cpu().numpy(),
                pred_class_id.cpu().numpy(),
            )

    return total
        

#######################################
# Optimizer tools
#######################################
def get_optim(cfg, model: nn.Module):
    if cfg.train.optimizer.name.lower() == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.train.optimizer.lr, weight_decay=cfg.train.optimizer.weight_decay,
        )
    elif cfg.train.optimizer.name.lower() == 'adamax':
        optimizer = torch.optim.Adamax(
            model.parameters(), lr=cfg.train.optimizer.lr, weight_decay=cfg.train.optimizer.weight_decay,
        )
    else:
        raise ValueError(f'Optimizer {cfg.train.optimizer} is not supported.')
    return optimizer


def get_sched(cfg, optimizer: torch.optim):
    if cfg.train.optimizer.scheduler.lower() == 'steplr':
        scheduler = CustomStepLR(
            optimizer, step_size=cfg.train.optimizer.step_size,
            gamma=cfg.train.optimizer.gamma, lowest_lr=cfg.train.optimizer.lowest_lr,
        )
        step_unit = cfg.train.optimizer.step_unit # 'epoch' or 'batch'
    else:
        raise ValueError(f'Scheduler {cfg.train.optimizer.scheduler} is not supported.')
    return scheduler, step_unit


class CustomStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, step_size, gamma, lowest_lr=0.):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.lowest_lr = lowest_lr
        self.count = 0
        super().__init__(optimizer)

    def get_lr(self):
        return None

    def step(self):
        self.count += 1
        if self.count == self.step_size:
            self.count = 0
            for param_group in self.optimizer.param_groups:
                lr = param_group['lr']
                lr = max(lr * self.gamma, self.lowest_lr)
                param_group['lr'] = lr


#######################################
# Dataset / DataLoader tools
#######################################
class Custom_1D_Dataset(Dataset):
    def __init__(self, cfg, mode):
        super().__init__()

        # Load raw data
        if mode == 'train':
            u = np.load(os.path.join(cfg.data.root, f'{cfg.data.train_name}.npz'))['u']
        elif mode == 'test':
            u = np.load(os.path.join(cfg.data.root, f'{cfg.data.test_name}.npz'))['u']
        else:
            raise ValueError(f'Mode {mode} is invalid. It should be one of ["train", "test"].')
        sort_str = ('btxyz')[:len(cfg.data.sorted_as)]
        u = u.transpose([cfg.data.sorted_as.find(k) for k in sort_str])

        # Grid of t
        self.grid_t = np.arange(cfg.data.t_resolution)
        if hasattr(cfg.train, 't_sub'):
            t_sub = int(getattr(cfg.train.t_sub, mode))
            self.grid_t = self.grid_t[::t_sub]

        # Grid of x (if subsampling)
        if u.shape[-1] != cfg.data.x_resolution:
            x_sub = u.shape[-1] // cfg.data.x_resolution
            u = u[:, :, ::x_sub]

        # To PyTorch tensor
        u = torch.from_numpy(u).type(torch.float32)
        self.grid_t = torch.from_numpy(self.grid_t).type(torch.float32)

        # Get indices of time for input / output
        T_in_start_idx, T_in_end_idx = cfg.train.T_in
        T_out_start_idx, T_out_end_idx = cfg.train.T_out

        # x: input, y: ouptut, t_x: time for x, t_y: time for y
        self.x = u[:, T_in_start_idx:T_in_end_idx, :]
        self.y = u[:, T_out_start_idx:T_out_end_idx, :]
        self.t_x = self.grid_t[T_in_start_idx:T_in_end_idx]
        self.t_y = self.grid_t[T_out_start_idx:T_out_end_idx]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.t_x, self.t_y


################################################################
# Please Modify!!!!!!
################################################################
class PV_Dataset(Dataset):
    def __init__(self, x, t, y):
        super().__init__()
        self.x = x # (B, T, C)
        self.t = t # (B, T)
        self.y = y # (B, 8, 1, C)
        self.yt = np.arange(8)

    def __len__(self):
        return self.x.shape[0]
        
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.t[idx], self.yt
    
class EmptyClass:
    pass

def lab_to_dat(lab, data, input_len, forecast_len=30):
    lab_size = len(lab)
    obs_size = lab_size // (input_len + 1)
    obs_lab = lab[:obs_size * (input_len + 1)].reshape(obs_size, input_len + 1)
    obs_lab = obs_lab.transpose()
    t = np.diff(obs_lab, axis=0)
    dat = data[obs_lab]
    x = dat[:input_len]
    y = dat[1:input_len + 1]
    fore_lab = obs_lab[-1]
    fore_lab = repeat(fore_lab, 'b -> t b', t=forecast_len)
    fore_lab = fore_lab + np.arange(forecast_len).reshape(-1, 1)
    z = data[fore_lab]
    out = (x, y, t, z)
    return (torch.Tensor(i).transpose(0,1) for i in out)

def pv(path='pv.csv', batch_size=64, input_len=64, tvt_ratio=(2, 1, 1), throwout_rate=0.1, error_var=0.0, verbose=True, forecast_len=8):
    np.random.seed(1) # For the same dataset as HeavyBallNODE
    data = np.genfromtxt(path, delimiter=',')[1:, :-2]
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    data = data + error_var * np.random.randn(*(data.shape))

    full_len = len(data) - 100
    eff_len = int(full_len * (1 - throwout_rate))
    assert eff_len > 0
    label_set = np.random.choice(range(full_len), eff_len, replace=False)
    label_set.sort()

    tvt_ratio = np.array(tvt_ratio) / np.sum(tvt_ratio)
    trlen, valen, _ = (tvt_ratio * eff_len).astype(int)
    tslen = eff_len - valen - trlen
    trlab, valab, tslab = label_set[:trlen], label_set[trlen:-tslen], label_set[-tslen:]

    output = EmptyClass()
    output.train_x, output.train_y, output.train_times, output.trext = lab_to_dat(trlab, data, input_len, forecast_len)
    #output.valid_x, output.valid_y, output.valid_times, output.vaext = lab_to_dat(valab, data, input_len, forecast_len)
    output.test_x, output.test_y, output.test_times, output.tsext = lab_to_dat(tslab, data, input_len, forecast_len)

    if verbose:
        print('Train-validation-test ratio: {}'.format(tvt_ratio))
        print('Input {} tp | forecast {} tp'.format(input_len, forecast_len))
        print('Full {} tp | using {} tp'.format(full_len, eff_len))
        print('Train {} tp | Validation {} tp | Test {} tp'.format(trlen, valen, tslen))
    
    trainds = PV_Dataset(output.train_x, output.train_times, output.trext)
    testds = PV_Dataset(output.test_x, output.test_times, output.tsext)
    return trainds, testds


class ScalarFlow_Dataset(Dataset):
    def __init__(self, cfg, mode):
        super().__init__()

        y_sub = cfg.data.y_sub
        x_crop, y_crop = cfg.data.crop

        # Load raw data
        if mode == 'train':
            u = np.load(os.path.join(cfg.data.root, f'{cfg.data.train_name}.npz'))['u']
        elif mode == 'test':
            u = np.load(os.path.join(cfg.data.root, f'{cfg.data.test_name}.npz'))['u']
        else:
            raise ValueError(f'Mode {mode} is invalid. It should be one of ["train", "test"].')
        sort_str = ('btxyz')[:len(cfg.data.sorted_as)]
        u = u.transpose([cfg.data.sorted_as.find(k) for k in sort_str])

        # Grid of t
        self.grid_t = np.arange(cfg.data.t_resolution)

        # To PyTorch tensor
        u = torch.from_numpy(u).type(torch.float32)
        self.grid_t = torch.from_numpy(self.grid_t).type(torch.float32)

        # Crop & Resize (only for ScalarFlow)
        u = u[:, :, x_crop[0]:x_crop[1], y_crop[0]:y_crop[1]]
        u = nn.AdaptiveAvgPool2d((cfg.data.x_resolution, cfg.data.y_resolution))(u)
        u = (u - 0.1533) / 0.4865

        # Break into time pieces
        _u = []
        for i in range(cfg.data.t_start, u.shape[1]-cfg.data.t_resolution+1, cfg.data.t_stride):
            _u.append(u[:, i:i+cfg.data.t_resolution])
        u = torch.cat(_u, dim=0)

        # Get indices of time for input / output
        T_in_start_idx, T_in_end_idx = cfg.train.T_in
        T_out_start_idx, T_out_end_idx = cfg.train.T_out

        # x: input, y: ouptut, t_x: time for x, t_y: time for y
        self.x = u[:, T_in_start_idx:T_in_end_idx, :]
        self.y = u[:, T_out_start_idx:T_out_end_idx, :][:, ::y_sub]
        self.t_x = self.grid_t[T_in_start_idx:T_in_end_idx]
        self.t_y = self.grid_t[T_out_start_idx:T_out_end_idx][::y_sub]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.t_x, self.t_y


def inf_generator(iterable):
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()

    
def split_train_test(data, ratio=0.8):
    n_samples = data.shape[0]
    data_train = data[:int(n_samples * ratio)]
    data_test = data[int(n_samples * ratio):]
    return data_train, data_test


def basic_collate_fn(batch, time_steps, cfg, data_type='train'):
    batch = torch.stack(batch)
    data_dict = {
        'data': batch,
        'time_steps': time_steps,
    }
    data_dict = split_and_subsample_batch(cfg, data_dict, data_type)
    return data_dict


def get_data_obj(cfg):

    train_collate_fn = None
    test_collate_fn = None

    if cfg.data.name.lower() in ['heat', 'burgers', 'ode']:
        train_dataset = Custom_1D_Dataset(cfg, 'train')
        test_dataset = Custom_1D_Dataset(cfg, 'test')

    elif cfg.data.name.lower() == 'pv':
        train_dataset, test_dataset = pv(os.path.join(cfg.data.root, 'pv.csv'))

    elif cfg.data.name.lower() == 'scalarflow':
        train_dataset = ScalarFlow_Dataset(cfg, 'train')
        test_dataset = ScalarFlow_Dataset(cfg, 'test')

    elif cfg.data.name.lower() == 'hopper':
        dataset_obj = HopperPhysics(
            root='./data',
            download=True,
            generate=False,
        )
        dataset = dataset_obj.get_dataset()[:cfg.data.num]
        n_tp_data = dataset.shape[1]
        time_steps = torch.arange(start=0, end=n_tp_data, step=1).float()
        time_steps = time_steps / len(time_steps)

        if not cfg.data.extrap:
            n_traj = len(dataset)
            n_reduced_tp = cfg.data.timepoints
            start_ind = np.random.randint(0, high=n_tp_data-n_reduced_tp+1, size=n_traj)
            end_ind = start_ind + n_reduced_tp
            sliced = []
            for i in range(n_traj):
                sliced.append(dataset[i, start_ind[i]:end_ind[i], :])
            dataset = torch.stack(sliced)
            time_steps = time_steps[:n_reduced_tp]

        train_dataset, test_dataset = split_train_test(dataset, ratio=0.8)
        train_collate_fn = lambda batch: basic_collate_fn(batch, time_steps, cfg, data_type='train')
        test_collate_fn = lambda batch: basic_collate_fn(batch, time_steps, cfg, data_type='test')

    elif cfg.data.name.lower() == 'physionet':
        train_dataset_obj = PhysioNet(
            root='./data/',
            train=True,
            quantization=cfg.data.quantization,
            download=True,
            n_samples=cfg.data.num,
        )
        test_dataset_obj = PhysioNet(
            root='./data/',
            train=False,
            quantization=cfg.data.quantization,
            download=True,
            n_samples=cfg.data.num,
        )
        total_dataset = train_dataset_obj
        total_dataset.data = total_dataset.data + test_dataset_obj.data
        total_dataset.labels = total_dataset.labels + test_dataset_obj.labels # unused
        train_dataset, test_dataset = train_test_split(total_dataset, train_size=0.8, random_state=42, shuffle=True)
        data_min, data_max = get_data_min_max(total_dataset)
        train_collate_fn = lambda batch: variable_time_collate_fn(batch, cfg, data_min, data_max, data_type='train')
        test_collate_fn = lambda batch: variable_time_collate_fn(batch, cfg, data_min, data_max, data_type='test')

    else:
        raise ValueError(f'Dataset {cfg.data.name.lower()} is not supported.')

    # Make dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.train_batch_size,
        shuffle=True,
        collate_fn=train_collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.data.test_batch_size,
        shuffle=False,
        collate_fn=test_collate_fn
    )
    data_obj = {
        'train_loader': inf_generator(train_loader),
        'test_loader': inf_generator(test_loader),
        'n_train_batches': len(train_loader),
        'n_test_batches': len(test_loader),
        'n_train_data': len(train_loader.dataset),
        'n_test_data': len(test_loader.dataset),
    }
    return data_obj


def get_next_batch(dataloader, device=torch.device('cpu')):
    # Make the union of all time points and perform normalization across the whole dataset
    data_dict = dataloader.__next__()
    batch_dict = {
        'observed_data': None,
        'observed_tp': None,
        'observed_mask': None,
        'data_to_predict': None,
        'tp_to_predict': None,
        'mask_predicted_data': None,
    }

    # remove the time points where there are no observations in this batch
    non_missing_tp = torch.sum(data_dict['observed_data'],(0,2)) != 0.
    batch_dict['observed_data'] = data_dict['observed_data'][:, non_missing_tp].to(device)
    batch_dict['observed_tp'] = data_dict['observed_tp'][non_missing_tp].to(device)

    if ('observed_mask' in data_dict) and (data_dict['observed_mask'] is not None):
        batch_dict['observed_mask'] = data_dict['observed_mask'][:, non_missing_tp].to(device)

    non_missing_tp = torch.sum(data_dict['data_to_predict'], dim=(0, 2)) != 0.
    batch_dict['data_to_predict'] = data_dict['data_to_predict'][:, non_missing_tp].to(device)
    batch_dict['tp_to_predict'] = data_dict['tp_to_predict'][non_missing_tp].to(device)

    if ('mask_predicted_data' in data_dict) and (data_dict['mask_predicted_data'] is not None):
        batch_dict['mask_predicted_data'] = data_dict['mask_predicted_data'][:, non_missing_tp].to(device)

    if ('labels' in data_dict) and (data_dict['labels'] is not None):
        batch_dict['labels'] = data_dict['labels']

    batch_dict['mode'] = data_dict['mode']
    return batch_dict


def split_data_extrap(data_dict, dataset):

    if dataset.lower() == 'hopper':
        n_observed_tp = data_dict['data'].size(1) // 3
    elif dataset.lower() == 'physionet':
        n_observed_tp = data_dict['data'].size(1) // 2
    else:
        raise ValueError

    split_dict = {
        'observed_data': data_dict['data'][:, :n_observed_tp, :].clone(),
        'observed_tp': data_dict['time_steps'][:n_observed_tp].clone(),
        'data_to_predict': data_dict['data'][:, n_observed_tp:, :].clone(),
        'tp_to_predict': data_dict['time_steps'][n_observed_tp:].clone(),
    }

    split_dict['observed_mask'] = None 
    split_dict['mask_predicted_data'] = None 
    split_dict['labels'] = None 

    if ('mask' in data_dict) and (data_dict['mask'] is not None):
        split_dict['observed_mask'] = data_dict['mask'][:, :n_observed_tp].clone()
        split_dict['mask_predicted_data'] = data_dict['mask'][:, n_observed_tp:].clone()

    if ('labels' in data_dict) and (data_dict['labels'] is not None):
        split_dict['labels'] = data_dict['labels'].clone()

    split_dict['mode'] = 'extrap'
    return split_dict


def split_data_interp(data_dict):

    split_dict = {
        'observed_data': data_dict['data'].clone(),
        'observed_tp': data_dict['time_steps'].clone(),
        'data_to_predict': data_dict['data'].clone(),
        'tp_to_predict': data_dict['time_steps'].clone(),
    }

    split_dict['observed_mask'] = None 
    split_dict['mask_predicted_data'] = None 
    split_dict['labels'] = None 

    if 'mask' in data_dict and data_dict['mask'] is not None:
        split_dict['observed_mask'] = data_dict['mask'].clone()
        split_dict['mask_predicted_data'] = data_dict['mask'].clone()

    if 'labels' in data_dict and data_dict['labels'] is not None:
        split_dict['labels'] = data_dict['labels'].clone()

    split_dict['mode'] = 'interp'
    return split_dict


def split_and_subsample_batch(cfg, data_dict, data_type='train'):

    if data_type == 'train':
        if cfg.data.extrap:
            processed_dict = split_data_extrap(data_dict, cfg.data.name)
        else:
            processed_dict = split_data_interp(data_dict)
    else:
        if cfg.data.extrap:
            processed_dict = split_data_extrap(data_dict, cfg.data.name)
        else:
            processed_dict = split_data_interp(data_dict)

    # add mask
    processed_dict = add_mask(processed_dict)

    # Subsampling
    if cfg.data.sample_tp is not None:
        processed_dict = subsample_observed_data(
            processed_dict, 
            n_tp_to_sample=cfg.data.sample_tp, 
        )

    return processed_dict


def add_mask(data_dict):
    data = data_dict['observed_data']
    mask = data_dict['observed_mask']

    if mask is None:
        mask = torch.ones_like(data) # 0 -> masked, 1 -> unmasked

    data_dict['observed_mask'] = mask
    return data_dict


def check_mask(data, mask):
    n_zeros = torch.sum(mask == 0.).cpu().numpy()
    n_ones = torch.sum(mask == 1.).cpu().numpy()

    # mask should contain only zeros and ones
    assert (n_zeros + n_ones) == np.prod(list(mask.size()))

    # all masked out elements should be zeros
    assert torch.sum(data[mask == 0.] != 0.) == 0


def normalize_data(data):
    reshaped = data.reshape(-1, data.size(-1))
    att_min = torch.min(reshaped, 0)[0]
    att_max = torch.max(reshaped, 0)[0]

    # we don't want to divide by zero
    att_max[att_max == 0.] = 1.
    if (att_max != 0.).all():
        data_norm = (data - att_min) / att_max
    else:
        raise Exception("Zero!")
    if torch.isnan(data_norm).any():
        raise Exception("Nans!")

    return data_norm, att_min, att_max


def normalize_masked_data(data, mask, att_min, att_max):
    # we don't want to divide by zero
    att_max[att_max == 0.] = 1.
    if (att_max != 0.).all():
        data_norm = (data - att_min) / att_max
    else:
        raise Exception("Zero!")
    if torch.isnan(data_norm).any():
        raise Exception("Nans!")

    # set masked out elements back to zero 
    data_norm[mask == 0] = 0

    return data_norm, att_min, att_max


def subsample_observed_data(data_dict, n_tp_to_sample=None):
    # If n_tp_to_sample is not None, randomly subsample the time points.
    # The resulting timeline has n_tp_to_sample points
    if n_tp_to_sample is not None:
        data, time_steps, mask = subsample_timepoints(
            data=data_dict['observed_data'].clone(), 
            time_steps=data_dict['observed_tp'].clone(), 
            mask=None if data_dict['observed_mask'] is None else data_dict['observed_mask'].clone(),
            n_tp_to_sample=n_tp_to_sample,
        )

    new_data_dict = {}
    for key in data_dict.keys():
        new_data_dict[key] = data_dict[key]

    new_data_dict['observed_data'] = data.clone()
    new_data_dict['observed_tp'] = time_steps.clone()
    new_data_dict['observed_mask'] = mask.clone()

    return new_data_dict


def subsample_timepoints(data, time_steps, mask, n_tp_to_sample=None):
    # n_tp_to_sample: number of time points to subsample. If not None, sample exactly n_tp_to_sample points
    if n_tp_to_sample is None:
        return data, time_steps, mask
    n_tp_in_batch = len(time_steps)

    if n_tp_to_sample > 1:
        # Subsample exact number of points
        assert n_tp_to_sample <= n_tp_in_batch
        n_tp_to_sample = int(n_tp_to_sample)

        for i in range(data.size(0)):
            missing_idx = sorted(np.random.choice(np.arange(n_tp_in_batch), n_tp_in_batch-n_tp_to_sample, replace=False))

            data[i, missing_idx] = 0.
            if mask is not None:
                mask[i, missing_idx] = 0.
    
    elif (n_tp_to_sample <= 1) and (n_tp_to_sample > 0):
        # Subsample percentage of points from each time series
        percentage_tp_to_sample = n_tp_to_sample
        for i in range(data.size(0)):
            # Take mask for current training sample and sum over all features
            # Figure out which time points don't have any measurements at all in this batch
            current_mask = mask[i].sum(-1).cpu()
            non_missing_tp = np.where(current_mask > 0)[0]
            n_tp_current = len(non_missing_tp)
            n_to_sample = int(n_tp_current * percentage_tp_to_sample)
            subsampled_idx = sorted(np.random.choice(non_missing_tp, n_to_sample, replace=False))
            tp_to_set_to_zero = np.setdiff1d(non_missing_tp, subsampled_idx)

            data[i, tp_to_set_to_zero] = 0.
            if mask is not None:
                mask[i, tp_to_set_to_zero] = 0.

    return data, time_steps, mask


#######################################
# Visualization tools
#######################################
def make_gif(u: np.ndarray, t: np.ndarray, save_path: str, value_lim: Optional[List[float]] = None):
    def animate(i):
        plt.cla()
        if len(u.shape) == 2: # shape (T, X): 1D time-series (ex: Heat, Burgers, ODE)
            plt.plot(u[i])
            if value_lim is not None:
                plt.ylim(value_lim)
            plt.xlabel('X')
            plt.ylabel('Y')
        elif len(u.shape) == 3: # shape (T, X, Y): 2D time-series (ex: scalarflow)
            plt.imshow(u[i], cmap='viridis')
        else:
            raise ValueError(f'The shape of u should be of length 2 or 3.')
        plt.title(f't = {t[i]}')
        plt.tight_layout()
    ani = FuncAnimation(plt.gcf(), animate, frames=len(t), interval=1)
    ani.save(save_path, fps=10)
    return


def make_gif2(true: np.ndarray, pred: np.ndarray, t: np.ndarray, save_path: str, value_lim: Optional[List[float]] = None):
    assert true.shape == pred.shape
    def animate(i):
        plt.cla()
        if len(pred.shape) == 2: # shape (T, X): 1D time-series (ex: Heat, Burgers, ODE)
            plt.plot(pred[i], c='blue', label='pred')
            plt.plot(true[i], c='orange', linestyle=(0, (3, 2)), label='true')
            if value_lim is not None:
                plt.ylim(value_lim)
            plt.legend(loc='lower left')
        else:
            raise ValueError(f'The shape of u should be of length 2 or 3.')
        plt.title(f't = {t[i]}')
        plt.tight_layout()
    ani = FuncAnimation(plt.gcf(), animate, frames=len(t), interval=1)
    ani.save(save_path, fps=10)
    return


#######################################
# Other tools
#######################################
def split_last_dim(data):
    last_dim = data.size()[-1]
    last_dim = last_dim//2
    return data[..., :last_dim], data[..., last_dim:]


def init_network_weights(net, std=0.1):
    for m in net.modules():
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, mean=0, std=std)
            nn.init.constant_(m.bias, val=0)


def reverse(tensor):
    idx = [i for i in range(tensor.size(0)-1, -1, -1)]
    return tensor[idx]


def sample_standard_gaussian(mu: torch.Tensor, sigma: torch.Tensor):
    d = torch.distributions.normal.Normal(torch.Tensor([0.]).to(mu.device), torch.Tensor([1.]).to(sigma.device))
    r = d.sample(mu.size()).squeeze(-1)
    return r * sigma.float() + mu.float()


def linspace_vector(start, end, n_points):
    # start is either one value or a vector
    size = np.prod(start.size())

    assert start.size() == end.size()
    if size == 1:
        # start and end are 1d-tensors
        res = torch.linspace(start, end, n_points)
    else:
        # start and end are vectors
        res = torch.Tensor()
        for i in range(0, start.size(0)):
            res = torch.cat(
                [res, torch.linspace(start[i], end[i], n_points)], dim=0)
        res = torch.t(res.reshape(start.size(0), n_points))
    return res