from __future__ import annotations
import numpy as np
import torch
from shapely.affinity import affine_transform
from typing import Tuple, Sequence


class NeuralBaseIntensity(torch.nn.Module):
    """Neural base intensity \mu(s, c) using an embedding + MLP.

    Mirrors the original implementation but with clearer naming and docstrings.
    """
    def __init__(self, n_class: int, embed_dim: int = 8, mlp_layer: int = 2, mlp_dim: int = 32):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.class_encoder = torch.nn.Embedding(n_class, embed_dim)
        self.input_layer = torch.nn.Linear(2 + embed_dim, mlp_dim)
        self.mlp = torch.nn.ModuleList([torch.nn.Linear(mlp_dim, mlp_dim) for _ in range(mlp_layer)])
        self.output_layer = torch.nn.Linear(mlp_dim, 1)

    def forward(self, s: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Compute \mu for locations and classes.

        Args:
            s: Locations of shape (..., 2) as normalized x,y in [0,1].
            c: Event classes of shape (...,) in [0, n_class).
        Returns:
            Tensor of shape (...) with base intensities.
        """
        enc = self.class_encoder(c)
        x = torch.concat((s, enc), dim=-1)
        h = torch.nn.functional.softplus(self.input_layer(x), beta=100)
        for layer in self.mlp:
            h = torch.nn.functional.softplus(layer(h), beta=100)
        out = torch.nn.functional.softplus(self.output_layer(h), beta=100)
        return out.squeeze(-1)


class BasePointProcess(torch.nn.Module):
    """Abstract base for conditional intensity models over sequences."""
    def __init__(self, T, S, mu, data_dim: int, int_res: int = 100, eval_res: int = 100, device: str = "cpu"):
        super().__init__()
        self.data_dim = data_dim
        self.T = T
        self.S = S
        self.int_res = int_res
        self.eval_res = eval_res
        self.device = device
        self.n_class = len(mu)
        self._mu = torch.nn.Parameter(torch.tensor(mu), requires_grad=False)

    def cond_lambda(self, xi: torch.Tensor, hti: torch.Tensor) -> torch.Tensor:
        """Conditional intensity at each event given its history.

        Args:
            xi: Current events, [B, D].
            hti: History events, [B, L, D].
        Returns:
            [B] intensities.
        """
        B, L, _ = hti.shape
        if L == 0:
            return self._mu[xi[:, 1].long()]
        xi2 = xi.unsqueeze(-2).repeat(1, L, 1)
        K = self.kernel(xi2.reshape(-1, self.data_dim), hti.reshape(-1, self.data_dim))
        l = K.reshape(B, L).sum(-1) + self._mu[xi[:, 1].long()]
        return l

    def log_likelihood(self, X: torch.Tensor) -> torch.Tensor:
        """Compute sequence log-likelihood (default naive form)."""
        B, L, _ = X.shape
        ts = X[:, :, 0].clone()
        lams = [self.cond_lambda(X[:, i, :].clone(), X[:, :i, :].clone()) for i in range(L)]
        lams = torch.stack(lams, dim=0).T
        mask = ts > 0
        sumlog = (torch.log(lams + 1e-5) * mask).sum()
        baserate = torch.sum(self._mu * (self.T[1] - self.T[0])) *                    affine_transform(self.S, np.array([111.320*0.772, 0, 0, 110.574, 0, 0])).area * B
        # Integral of kernel is model-specific; subclasses may override.
        loglik = sumlog - baserate
        return loglik

    def forward(self, X: torch.Tensor):
        return self.log_likelihood(X)
