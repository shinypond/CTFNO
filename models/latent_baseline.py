###########################
# Referred to (Latent ODE)
# https://github.com/YuliaRubanova/latent_ode
###########################

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
from utils import init_network_weights, sample_standard_gaussian
from .likelihood_eval import compute_binary_CE_loss, compute_multiclass_CE_loss, masked_gaussian_log_density, compute_mse


def create_classifier(z0_dim, n_labels):
    return nn.Sequential(
        nn.Linear(z0_dim, 300),
        nn.ReLU(),
        nn.Linear(300, 300),
        nn.ReLU(),
        nn.Linear(300, n_labels),
    )


class VAE_Baseline(nn.Module):
    def __init__(
        self,
        input_dim,
        latent_dim, 
        z0_prior,
        device,
        obsrv_std=0.01, 
        loss_type='iwae',
        classify_per_tp=False,
        linear_classifier=False,
        n_labels=1,
    ):
        super(VAE_Baseline, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.device = device
        self.obsrv_std = torch.Tensor([obsrv_std]).to(device)
        self.z0_prior = z0_prior
        assert loss_type in ['iwae', 'iwae+100*ce', 'ce'] # Hopper, PhysioNet, Activity
        self.loss_type = loss_type
        self.classify_per_tp = classify_per_tp
        if loss_type != 'iwae':
            if linear_classifier:
                self.classifier = nn.Linear(latent_dim, n_labels)
            else:
                self.classifier = create_classifier(latent_dim, n_labels)
            init_network_weights(self.classifier)


    def get_gaussian_likelihood(self, truth, pred_y, mask=None):

        truth_repeated = truth.repeat(pred_y.size(0), 1, 1, 1)
        
        if mask is not None:
            mask = mask.repeat(pred_y.size(0), 1, 1, 1)

        log_density_data = masked_gaussian_log_density(
            pred_y,
            truth_repeated, 
            obsrv_std=self.obsrv_std,
            mask=mask
        )
        log_density_data = log_density_data.permute(1, 0)
        log_density = torch.mean(log_density_data, dim=1)

        return log_density # Shape (n_traj_samples,)

    def get_mse(self, truth, pred_y, mask=None):
        truth_repeated = truth.repeat(pred_y.size(0), 1, 1, 1)
        
        if mask is not None:
            mask = mask.repeat(pred_y.size(0), 1, 1, 1)

        log_density_data = compute_mse(pred_y, truth_repeated, mask=mask)

        return torch.mean(log_density_data)

    def get_reconstruction(self, *args, **kwargs):
        raise NotImplementedError

    def compute_all_losses(self, batch_dict, n_traj_samples=1, kl_coef=1., train=True):

        pred_y, info = self.get_reconstruction(
            batch_dict["tp_to_predict"], 
            batch_dict["observed_data"],
            batch_dict["observed_tp"], 
            mask=batch_dict["observed_mask"],
            n_traj_samples=n_traj_samples,
        )
        
        fp_mu, fp_std, _ = info["first_point"]
        
        fp_std = fp_std.abs()
        fp_distr = Normal(fp_mu, fp_std)

        assert torch.sum(fp_std < 0) == 0.

        kldiv_z0 = kl_divergence(fp_distr, self.z0_prior)

        if torch.isnan(kldiv_z0).any():
            print(fp_mu)
            print(fp_std)
            raise Exception("kldiv_z0 is Nan!")

        kldiv_z0 = torch.mean(kldiv_z0, dim=(1, 2))

        # Compute likelihood of all the points
        if train:
            rec_likelihood = self.get_gaussian_likelihood(
                batch_dict["data_to_predict"],
                pred_y,
                mask=batch_dict["mask_predicted_data"],
            )
        else:
            rec_likelihood = torch.zeros(1).to(kldiv_z0)

        # Compute MSE loss
        if not train:
            mse = self.get_mse(
                batch_dict["data_to_predict"],
                pred_y,
                mask=batch_dict["mask_predicted_data"],
            )
        else:
            mse = torch.zeros(1).to(kldiv_z0)

        # IWAE loss
        iwae_loss = -torch.logsumexp(rec_likelihood - kl_coef * kldiv_z0, dim=0)
        if torch.isnan(iwae_loss):
            iwae_loss = -torch.mean(rec_likelihood - kl_coef * kldiv_z0, dim=0)

        # Cross Entropy loss
        if self.loss_type != 'iwae' and batch_dict['labels'] is not None:
            if (batch_dict['labels'].size(-1) == 1) or (len(batch_dict['labels'].size()) == 1):
                ce_loss = compute_binary_CE_loss(
                    info['label_predictions'],
                    batch_dict['labels'],
                )
            else:
                ce_loss = compute_multiclass_CE_loss(
                    info['label_predictions'],
                    batch_dict['labels'],
                    mask=batch_dict['mask_predicted_data'],
                )

        # Loss
        if self.loss_type == 'iwae':
            loss = iwae_loss
        elif self.loss_type == 'iwae+100*ce':
            loss = iwae_loss + 100 * ce_loss
        elif self.loss_type == 'ce':
            loss = ce_loss
        else:
            raise ValueError

        results = {
            'loss': torch.mean(loss),
            'likelihood': torch.mean(rec_likelihood).detach(),
            'mse': torch.mean(mse).detach(),
            'kl_first_p': torch.mean(kldiv_z0).detach(),
            'std_first_p': torch.mean(fp_std).detach(),
        }

        return results


class LatentODE(VAE_Baseline):
    def __init__(self, cfg, encoder, solver, decoder):
        device = torch.device(f'cuda:{cfg.gpu}') if torch.cuda.is_available() else torch.device('cpu')
        input_dim = cfg.encoder.input_dim
        latent_dim = cfg.encoder.latent_dim
        z0_prior = Normal(torch.Tensor([0.]).to(device), torch.Tensor([1.]).to(device))
        obsrv_std = 0.001 if cfg.data.name.lower() == 'hopper' else 0.01
        self.loss_type = cfg.train.latent_loss_type
        self.classify_per_tp = cfg.train.classify_per_tp
        linear_classifier = cfg.train.linear_classifier 
        n_labels = cfg.data.n_labels
        super().__init__(
            input_dim, latent_dim, z0_prior, device, obsrv_std,
            self.loss_type, self.classify_per_tp, linear_classifier, n_labels,
        )

        self.encoder = encoder
        self.solver = solver
        self.decoder = decoder

    def get_reconstruction(
        self,
        time_steps_to_predict,
        truth,
        truth_time_steps, 
        mask=None,
        n_traj_samples=1,
        run_backwards=True,
    ):
        truth_w_mask = truth
        if mask is not None:
            truth_w_mask = torch.cat((truth, mask), -1)
        first_point_mu, first_point_std = self.encoder(
            truth_w_mask, truth_time_steps, run_backwards=run_backwards
        )
        
        means_z0 = first_point_mu.repeat(n_traj_samples, 1, 1)
        sigma_z0 = first_point_std.repeat(n_traj_samples, 1, 1)
        first_point_enc = sample_standard_gaussian(means_z0, sigma_z0)

        assert torch.sum(first_point_std <= 0) == 0.

        assert not torch.isnan(time_steps_to_predict).any()
        assert not torch.isnan(first_point_enc).any()

        sol_y = self.solver(first_point_enc, time_steps_to_predict)
        pred_x = self.decoder(sol_y)

        all_extra_info = {
            "first_point": (first_point_mu, first_point_std, first_point_enc),
            "latent_traj": sol_y.detach(),
        }

        if self.loss_type != 'iwae':
            if self.classify_per_tp:
                all_extra_info['label_predictions'] = self.classifier(sol_y)
            else:
                all_extra_info['label_predictions'] = self.classifier(first_point_enc).squeeze(-1)

        return pred_x, all_extra_info


