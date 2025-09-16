from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from typing import Optional
from ..data import generate_onehot_enc


class Multivariate_Exponential_StdDiffusion_Kernel_NWD(torch.nn.Module):
    """Exponentially decaying kernel with network distance."""
    def __init__(self, alpha, beta, sigma, sigma_l, alpha_mask, AllSPL):
        super().__init__()
        self._alpha = torch.nn.Parameter(torch.tensor(alpha), requires_grad=True)
        self._beta = torch.nn.Parameter(torch.tensor(beta), requires_grad=True)
        self._sigma = torch.nn.Parameter(torch.tensor(sigma), requires_grad=True)
        self._sigma_l = torch.nn.Parameter(torch.tensor(sigma_l), requires_grad=False)
        self._alpha_mask = torch.nn.Parameter(torch.tensor(alpha_mask), requires_grad=False)
        self.AllSPL = AllSPL

    def get_alpha(self):
        return self._alpha

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        alpha_masked = torch.where(self._alpha_mask.bool(), self._alpha, torch.zeros_like(self._alpha))
        alphas = alpha_masked[x[:, 1].long(), y[:, 1].long()]
        mask = x[:, 0] > 0
        tds = (x[:, 0] - y[:, 0]) * mask
        tds[~mask] = 1.0
        on_same_edge = (x[:, 6] == y[:, 6])
        nwds = torch.zeros_like(x[:, 0])
        nwds[on_same_edge] = torch.norm((x[on_same_edge, 2:4] - y[on_same_edge, 2:4]) * torch.tensor([111.320*0.772, 110.574], device=x.device), dim=-1)
        A = x[~on_same_edge, 5] + y[~on_same_edge, 5] + self.AllSPL[x[~on_same_edge, 4].long(), y[~on_same_edge, 4].long()]
        nwds[~on_same_edge] = A / 1000
        return alphas * self._beta * torch.exp(- self._beta * tds - nwds ** 2 / 2 / self._sigma ** 2 / tds) / (2 * np.pi * tds * self._sigma ** 2)


class Multivariate_Exponential_Gaussian_Kernel_NWD(torch.nn.Module):
    def __init__(self, alpha, beta, sigma, sigma_l, alpha_mask, AllSPL, device):
        super().__init__()
        self._alpha = torch.nn.Parameter(torch.tensor(alpha), requires_grad=True)
        self._beta = torch.nn.Parameter(torch.tensor(beta), requires_grad=True)
        self._sigma = torch.nn.Parameter(torch.tensor(sigma), requires_grad=True)
        self._sigma_l = torch.nn.Parameter(torch.tensor(sigma_l), requires_grad=False)
        self._alpha_mask = torch.nn.Parameter(torch.tensor(alpha_mask), requires_grad=False)
        self.AllSPL = AllSPL

    def get_alpha(self, threshold: float = 0.0):
        alphas = self._alpha.clone()
        return torch.where(alphas >= threshold, alphas, 0.0)


class Multivariate_Exponential_Gaussian_GAT_Kernel_NWD(torch.nn.Module):
    def __init__(self, n_head, out_channel, alpha_coef, alpha, beta, sigma, sigma_l, alpha_mask, AllSPL, device, alpha_prior=None):
        super().__init__()
        self._beta = torch.nn.Parameter(torch.tensor(beta), requires_grad=True)
        self._sigma = torch.nn.Parameter(torch.tensor(sigma), requires_grad=True)
        self._sigma_l = torch.nn.Parameter(torch.tensor(sigma_l), requires_grad=False)
        self._alpha_mask = torch.nn.Parameter(torch.tensor(alpha_mask), requires_grad=False)
        self.AllSPL = AllSPL

        self.n_head = n_head
        self.alpha_coef = torch.nn.Parameter(torch.tensor(alpha_coef)[:, None, None], requires_grad=True)
        self.edge_indices = torch.stack(torch.where(self._alpha_mask != 0)).to(device)
        if alpha_prior is not None:
            self.edge_attrs = torch.Tensor(alpha_prior)[self._alpha_mask != 0].reshape(-1, 1).to(device)
            self.GATconv = GATConv(3+7, out_channel, n_head, edge_dim=1)
        else:
            self.edge_attrs = None
            self.GATconv = GATConv(3+7, out_channel, n_head, edge_dim=None)
        self.onehot_enc = torch.FloatTensor(generate_onehot_enc(3, 7)).to(device)

    def get_alpha(self):
        shape = (self._alpha_mask.shape[0], self._alpha_mask.shape[0])
        _, (edges, att_weights) = self.GATconv(self.onehot_enc, edge_index=self.edge_indices, edge_attr=self.edge_attrs, return_attention_weights=True)
        alphas = torch.stack([torch.sparse_coo_tensor(edges, att_weights[:, i], size=shape).to_dense().T for i in range(self.n_head)], dim=0)
        alpha_coefs = F.softmax(self.alpha_coef, dim=0)
        alphas = (alphas * alpha_coefs).sum(0)
        return alphas * self._alpha_mask


class Multivariate_Exponential_Gaussian_latent_GAT_Kernel_NWD(torch.nn.Module):
    def __init__(self, n_head, out_channel, alpha_coef, alpha, beta, sigma, sigma_l, alpha_mask, AllSPL, device, alpha_prior=None):
        super().__init__()
        self._alpha = torch.nn.Parameter(torch.tensor(alpha), requires_grad=True)
        self._beta = torch.nn.Parameter(torch.tensor(beta), requires_grad=True)
        self._sigma = torch.nn.Parameter(torch.tensor(sigma), requires_grad=True)
        self._sigma_l = torch.nn.Parameter(torch.tensor(sigma_l), requires_grad=False)
        self._alpha_mask = torch.nn.Parameter(torch.tensor(alpha_mask), requires_grad=False)
        self.AllSPL = AllSPL
        self.n_head = n_head
        self.alpha_coef = torch.nn.Parameter((torch.ones(n_head) / n_head)[:, None, None], requires_grad=True)
        self.edge_indices = torch.stack(torch.where(self._alpha_mask != 0)).to(device)
        if alpha_prior is not None:
            self.edge_attrs = torch.Tensor(alpha_prior)[self._alpha_mask != 0].reshape(-1, 1).to(device)
            self.GATconv = GATConv(3+7, out_channel, n_head, edge_dim=1)
        else:
            self.edge_attrs = None
            self.GATconv = GATConv(3+7, out_channel, n_head, edge_dim=None)
        self.onehot_enc = torch.FloatTensor(generate_onehot_enc(3, 7)).to(device)

    def get_alpha(self):
        shape = (self._alpha_mask.shape[0], self._alpha_mask.shape[0])
        _, (edges, att_weights) = self.GATconv(self.onehot_enc, edge_index=self.edge_indices, edge_attr=self.edge_attrs, return_attention_weights=True)
        alphas_latent = torch.stack([torch.sparse_coo_tensor(edges, att_weights[:, i], size=shape).to_dense().T for i in range(self.n_head)], dim=0)
        alpha_coefs = F.softmax(self.alpha_coef, dim=0)
        alphas_latent = (alphas_latent * alpha_coefs).sum(0)
        return self._alpha * self._alpha_mask * alphas_latent


class Multivariate_Exponential_Gaussian_L3Net_Kernel_NWD(torch.nn.Module):
    def __init__(self, order_list, adj_mat, alpha_coef, alpha, beta, sigma, sigma_l, alpha_mask, AllSPL, device):
        super().__init__()
        self._beta = torch.nn.Parameter(torch.tensor(beta), requires_grad=True)
        self._sigma = torch.nn.Parameter(torch.tensor(sigma), requires_grad=True)
        self._sigma_l = torch.nn.Parameter(torch.tensor(sigma_l), requires_grad=False)
        self._alpha_mask = torch.nn.Parameter(torch.tensor(alpha_mask), requires_grad=False)
        self.AllSPL = AllSPL
        self._A = torch.Tensor(adj_mat)
        self.alpha_coef = torch.nn.Parameter(torch.tensor(alpha_coef)[:, None, None], requires_grad=True)

        self.n_basis = len(order_list)
        masks = []
        bases = []
        for order in order_list:
            if order == 0:
                b_mask = torch.eye(self._A.shape[1]).float()
            else:
                b_mask = (self._A.matrix_power(order) != 0).float()
            masks.append(b_mask)
            bases.append(self._init_local_filter(b_mask))
        self.basiss = torch.nn.Parameter(torch.stack(bases, dim=0), requires_grad=True)
        self.basis_masks = torch.nn.Parameter(torch.stack(masks, dim=0), requires_grad=False)

    def _init_local_filter(self, mask: torch.Tensor) -> torch.Tensor:
        in_size = self.n_basis ** 2 * self._A.sum(1).mean()
        std_ = torch.sqrt(1. / in_size)
        return torch.randn((mask.shape[0], mask.shape[0])) * std_ * mask

    def get_alpha(self):
        alphas = self.basiss * self.basis_masks
        alphas = (F.softplus(alphas, beta=100) * F.softplus(self.alpha_coef, beta=100)).sum(0)
        return alphas
