# STNPP: Spatio‑Temporal Neural Point Process

A clean, modular Python package that implements multivariate Hawkes-style spatio‑temporal point processes on networks with neural base intensity and learnable kernels (GAT/L3Net variants).

## Install

```bash
git clone [https://github.com/yourusername/stnpp_package.git](https://github.com/McDaniel7/Spatio-temporal-network-PP-for-Crime-Modeling-in-Valencia.git)
cd stnpp_package
pip install -e .
```

## Quick start

The CLI allows training from a YAML configuration file.

```bash
stnpp --config examples/config.example.yaml
### Or with your own config file
# stnpp --config path/to/config.yaml
```

The CLI will:
1) load data & precompute tensors; 2) build the model (base intensity, kernel, wrapper); 3) train with logging & optional regularization; 4) save checkpoints under `results/saved_models/<modelname>/`.

See `examples/config.example.yaml` for a minimal configuration.

## Usage as a library

You can also use STNPP in your own Python scripts:

```python
import torch
from stnpp import (
    data_pre_formatting, NeuralBaseIntensity,
    Multivariate_Exponential_Gaussian_latent_GAT_Kernel_NWD,
    MultivariateHawkesNetworkDistNeuralBaseIntensity
)

# Example: create a neural base intensity and a latent-GAT kernel
neural_BI = NeuralBaseIntensity(n_class=3, embed_dim=8, mlp_layer=2, mlp_dim=32)
kernel = Multivariate_Exponential_Gaussian_latent_GAT_Kernel_NWD(
    n_head=8, out_channel=64, alpha_coef=torch.ones(8),
    alpha=torch.rand(21, 21), beta=torch.tensor([1.0]),
    sigma=torch.tensor([0.1]), sigma_l=torch.tensor([0.5]),
    alpha_mask=torch.ones((21, 21)), AllSPL=torch.rand(21, 21),
    device=torch.device("cpu")
)

# Wrap into the Hawkes process model (with placeholder arguments for T, S, etc.)
model = MultivariateHawkesNetworkDistNeuralBaseIntensity(
    T=[0.0, 100.0], S=None, mu=torch.zeros(21),
    neural_BI=neural_BI, kernel=kernel, data_dim=9, device="cpu",
    int_grid=torch.rand(10, 2), int_grid_nwd=torch.rand(10, 10),
    int_grid_slabel=torch.arange(10)
)
```

## Package layout

```
Sptio-temporal-network-PP/
├─ pyproject.toml
├─ README.md
├─ LICENSE
├─ .gitignore
├─ examples/
│  └─ config.example.yaml
├─ tests/
│  └─ test_data.py
└─ stnpp/
   ├─ __init__.py
   ├─ cli.py                  ← CLI (`stnpp --config ...`)
   ├─ data.py                 ← data_pre_formatting, generate_onehot_enc  :contentReference[oaicite:4]{index=4}
   ├─ models/
   │  ├─ __init__.py
   │  ├─ base.py              ← NeuralBaseIntensity, BasePointProcess     :contentReference[oaicite:5]{index=5}
   │  ├─ kernels.py           ← all kernel classes                        :contentReference[oaicite:6]{index=6}
   │  └─ stnpp.py             ← MultivariateHawkesNetworkDist...          :contentReference[oaicite:7]{index=7}
   └─ train/
      ├─ __init__.py
      └─ loops.py             ← train_MHP_yearly, clippers 
```
## Citation

```
@article{dong2024spatio,
  title={Spatio-temporal-network point processes for modeling crime events with landmarks},
  author={Dong, Zheng and Mateu, Jorge and Xie, Yao},
  journal={arXiv preprint arXiv:2409.10882},
  year={2024}
}
```
