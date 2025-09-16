# STNPP: Spatio‑Temporal Neural Point Process

A clean, modular Python package that implements multivariate Hawkes-style spatio‑temporal point processes on networks with neural base intensity and learnable kernels (GAT/L3Net variants).

## Install

```bash
pip install -e .
```

## Quick start

```bash
stnpp --config path/to/config.yaml
```

The CLI will:
1) load data & precompute tensors; 2) build the model (base intensity, kernel, wrapper); 3) train with logging & optional regularization; 4) save checkpoints under `results/saved_models/<modelname>/`.

See `examples/config.example.yaml` for a minimal configuration.

## Package layout

```
stnpp/
  __init__.py
  cli.py
  data.py
  models/
    __init__.py
    base.py
    kernels.py
    stnpp.py
  train/
    __init__.py
    loops.py
tests/
```

## Citing / Origins

Core ideas and several class/function names come from the original research code you provided (now reorganized). Please check source headers in `models/*` and `train/loops.py` for attribution.
