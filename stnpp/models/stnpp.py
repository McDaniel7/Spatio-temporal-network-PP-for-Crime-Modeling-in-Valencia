from __future__ import annotations
import numpy as np
import torch
from shapely.affinity import affine_transform

from .base import BasePointProcess
from typing import Tuple


class MultivariateHawkesNetworkDistNeuralBaseIntensity(BasePointProcess):
    """Multivariate Hawkes on a road network with neural base intensity.

    Implements log-likelihood and numerical integration tailored to
    network-distance kernels, mirroring the original logic.
    """
    def __init__(self, T, S, mu, neural_BI, kernel, data_dim, device,
                 int_grid, int_grid_nwd, int_grid_slabel, min_dist: float = 0.01, neural_base: bool = True):
        super().__init__(T=T, S=S, mu=mu, data_dim=data_dim, device=device)
        self.kernel = kernel
        self.base = neural_BI
        self.min_dist = min_dist
        self.int_grid = int_grid
        self.int_grid_nwd = int_grid_nwd
        self.int_grid_slabel = int_grid_slabel
        self.unit_vol = affine_transform(self.S, np.array([111.320*0.772, 0, 0, 110.574, 0, 0])).area / self.int_grid.shape[0]
        locs = self.int_grid.clone().repeat(3, 1)
        types = torch.arange(3).reshape(-1, 1).repeat(1, self.int_grid.shape[0]).flatten().reshape(-1, 1).to(device)
        self.base_loc_type_grid = torch.concat((locs, types), dim=-1)
        self.neural_base = neural_base

    def log_likelihood(self, X):
        B, L, _ = X.shape
        ts = X[:, :, 0].clone()
        mask_all = ts > 0

        tts1 = ts.unsqueeze(1).repeat(1, L, 1)
        tts2 = ts.unsqueeze(-1).repeat(1, 1, L)
        taus = tts1 - tts2
        mask_t = (taus > 0).bool()
        taus[~mask_t] = 100.0

        hist_idx = X[:, :, 1][:, :, None].repeat(1, 1, L).long()
        cur_idx = X[:, :, 1][:, None, :].repeat(1, L, 1).long()
        alpha_val = (self.kernel.get_alpha() * self.kernel._alpha_mask)[cur_idx, hist_idx]

        if self.kernel._beta.shape[0] == 1:
            beta_val = torch.ones_like(X[:, :, 1][:, :, None]) * self.kernel._beta
            sigma_val = torch.ones_like(X[:, :, 1][:, :, None]) * self.kernel._sigma
        else:
            beta_val = self.kernel._beta[X[:, :, 1].long()][:, :, None]
            sigma_val = self.kernel._sigma[X[:, :, 1].long()][:, :, None]

        lls1 = X[:, :, 2:4].unsqueeze(1).repeat(1, L, 1, 1)
        lls2 = X[:, :, 2:4].unsqueeze(2).repeat(1, 1, L, 1)
        on_same_edge = (X[:, :, 6][:, None, :] == X[:, :, 6][:, :, None])

        nwds = torch.zeros_like(taus)
        nwds[on_same_edge] = torch.norm((lls1[on_same_edge] - lls2[on_same_edge]) * torch.tensor([111.320*0.772, 110.574], device=X.device), dim=-1)
        A = X[:, :, 5][:, None, :].repeat(1, L, 1)[~on_same_edge] +             X[:, :, 5][:, :, None].repeat(1, 1, L)[~on_same_edge] +             self.kernel.AllSPL[X[:, :, 4][:, None, :].repeat(1, L, 1)[~on_same_edge].long(),
                               X[:, :, 4][:, :, None].repeat(1, 1, L)[~on_same_edge].long()]
        nwds[~on_same_edge] = A / 1000
        mask_s = (nwds > self.min_dist).bool()

        trigger = (alpha_val * beta_val * torch.exp(- beta_val * taus - nwds**2/2/sigma_val**2) / (2*np.pi*sigma_val**2) * mask_t * mask_s * mask_all[:, :, None]).sum(-2)

        if self.neural_base:
            norm_loc = X[:, :, 2:4].clone()
            norm_loc[:, :, 0] = (norm_loc[:, :, 0] - self.S.bounds[0]) / (self.S.bounds[2] - self.S.bounds[0])
            norm_loc[:, :, 1] = (norm_loc[:, :, 1] - self.S.bounds[1]) / (self.S.bounds[3] - self.S.bounds[1])
            baserate = self.base(norm_loc, (X[:, :, 1] / 7).long())
        else:
            baserate = self._mu[X[:, :, 1].long()]
        lams = trigger + baserate

        integral = self.numerical_integral(X)
        loglik = (torch.log(torch.clamp(lams, min=1e-5)) * mask_all).sum() - integral
        return loglik, lams, integral

    def numerical_integral(self, X):
        B, L, _ = X.shape
        ts = X[:, :, 0]
        mask = ts > 0

        if self.neural_base:
            norm_loc = self.base_loc_type_grid[:, :2].clone()
            norm_loc[..., 0] = (norm_loc[..., 0] - self.S.bounds[0]) / (self.S.bounds[2] - self.S.bounds[0])
            norm_loc[..., 1] = (norm_loc[..., 1] - self.S.bounds[1]) / (self.S.bounds[3] - self.S.bounds[1])
            grid_val = self.base(norm_loc, self.base_loc_type_grid[:, 2].long())
            baserate = grid_val.sum() * self.unit_vol * (self.T[1] - self.T[0]) * B
        else:
            baserate = torch.stack([self._mu[self.int_grid_slabel.long() + i*7] for i in range(3)], dim=0).sum() * self.unit_vol * (self.T[1] - self.T[0]) * B

        if self.kernel._beta.shape[0] == 1:
            beta_val = torch.ones_like(X[:, :, 1]) * self.kernel._beta
            sigma_val = torch.ones_like(X[:, :, 1][None, :, :]) * self.kernel._sigma
        else:
            beta_val = self.kernel._beta[X[:, :, 1].long()]
            sigma_val = self.kernel._sigma[X[:, :, 1].long()][None, :, :]

        time_int = (1 - torch.exp(-beta_val * (self.T[1] - ts)))
        spa_int = 0
        his_idx = X[:, :, 1].long()
        nwds = self.int_grid_nwd[:, X[:, :, -1].long()]
        nwds[nwds < self.min_dist] = self.min_dist
        mask_s = (nwds >= 0.).bool()
        for i in range(3):
            cur_idx = (self.int_grid_slabel + 7 * i).long()
            alpha_val = (self.kernel.get_alpha() * self.kernel._alpha_mask)[cur_idx][:, his_idx]
            spa_int = spa_int + (torch.exp(-nwds**2/2/sigma_val**2) / (2*np.pi*sigma_val**2) * alpha_val * mask_s).sum(0) * self.unit_vol
        integral = (time_int * spa_int * mask).sum()
        return integral + baserate
