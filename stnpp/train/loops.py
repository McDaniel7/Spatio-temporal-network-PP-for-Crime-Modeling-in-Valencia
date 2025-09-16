from __future__ import annotations
import os
import numpy as np
import arrow
import torch
import torch.optim as optim


class NonNegativeClipper:
    """Clamp model parameters to valid ranges after each step."""
    def __call__(self, module):
        if hasattr(module, '_mu'):
            module._mu.data = torch.clamp(module._mu.data, min=0.)
        if hasattr(module, '_alpha'):
            module._alpha.data = torch.clamp(module._alpha.data, min=0.)
        if hasattr(module, '_beta'):
            module._beta.data = torch.clamp(module._beta.data, min=1e-5)
        if hasattr(module, '_sigma'):
            module._sigma.data = torch.clamp(module._sigma.data, min=1e-5)
        if hasattr(module, '_sigma_l'):
            module._sigma_l.data = torch.clamp(module._sigma_l.data, min=1e-5)
        if hasattr(module, '_sigma_ls'):
            module._sigma_ls.data = torch.clamp(module._sigma_ls.data, min=1e-5)
        if hasattr(module, '_gammas'):
            module._gammas.data = torch.clamp(module._gammas.data, min=1e-5)


class NonNegativeClipper2:
    """Variant that also caps alpha at 1."""
    def __call__(self, module):
        if hasattr(module, '_mu'):
            module._mu.data = torch.clamp(module._mu.data, min=0.)
        if hasattr(module, '_alpha'):
            module._alpha.data = torch.clamp(module._alpha.data, min=0., max=1.)
        if hasattr(module, '_beta'):
            module._beta.data = torch.clamp(module._beta.data, min=1e-5)
        if hasattr(module, '_sigma'):
            module._sigma.data = torch.clamp(module._sigma.data, min=1e-5)
        if hasattr(module, '_sigma_l'):
            module._sigma_l.data = torch.clamp(module._sigma_l.data, min=1e-5)
        if hasattr(module, '_sigma_ls'):
            module._sigma_ls.data = torch.clamp(module._sigma_ls.data, min=1e-5)
        if hasattr(module, '_gammas'):
            module._gammas.data = torch.clamp(module._gammas.data, min=1e-5)


def train_MHP_yearly(
    model,
    train_data,
    test_data,
    device,
    modelname,
    num_epochs=20,
    lr=1e-4,
    batch_size=5,
    stationary=False,
    l1_reg=False,
    lnu_reg=False,
    lam_l1=100,
    lam_lnu=100,
    print_iter=2,
    log_iter=100,
    tol=1e-2,
    testing=False,
    save_model=False,
    save_path=".",
    new_folder=False,
    start_epoch=0,
):
    """Train a multivariate Hawkes model for a fixed number of epochs."""

    clipper = NonNegativeClipper2() if stationary else NonNegativeClipper()

    path = os.path.join(save_path, "results", "saved_models", modelname)
    if save_model and new_folder:
        if os.path.exists(path):
            print("Duplicated folder!")
            return None
        os.makedirs(path, exist_ok=True)

    model.to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=lr)

    best_lglk = -np.inf
    prev_lglk = -np.inf
    no_incre = 0
    converge = 0
    _lr = lr
    n_batches = int(train_data.shape[0] / batch_size)

    train_llk_out, test_llk_out = [], []

    for i in range(num_epochs):
        epoch_llk_loss = 0.0
        epoch_l1_loss = 0.0
        epoch_lnu_loss = 0.0
        optimizer.zero_grad()
        for b in range(n_batches):
            idx = np.arange(batch_size * b, batch_size * (b + 1))
            data = train_data[idx]
            loglik, lams, integ = model(data.to(device))
            loss = - loglik / batch_size
            if l1_reg:
                l1_norm = torch.norm(model.kernel.get_alpha() * model.kernel._alpha_mask, p=1)
                loss = loss + lam_l1 * l1_norm
            if lnu_reg:
                lnu_norm = torch.norm(model.kernel.get_alpha() * model.kernel._alpha_mask, p='nuc')
                loss = loss + lam_lnu * lnu_norm
            loss.backward()
            optimizer.step()
            model.apply(clipper)
            epoch_llk_loss += loglik.item()
            if l1_reg: epoch_l1_loss += lam_l1 * l1_norm.item()
            if lnu_reg: epoch_lnu_loss += lam_lnu * lnu_norm.item()

        event_num = (train_data[..., 0] > 0).sum()
        event_llk = epoch_llk_loss / event_num
        train_llk_out.append(float(event_llk))

        if event_llk > best_lglk:
            best_lglk = event_llk
            no_incre = 0
        else:
            no_incre += 1
        if no_incre == 50:
            print("Learning rate decrease!")
            _lr = _lr / np.sqrt(10)
            optimizer = optim.Adadelta(model.parameters(), lr=_lr)
            no_incre = 0
            best_lglk = -np.inf
        if abs(event_llk - prev_lglk) > tol:
            converge = 0
        else:
            converge += 1
        prev_lglk = event_llk

        if (i + 1) % print_iter == 0:
            print(f"[{arrow.now()}] Epoch: {i+start_epoch}\tTemporal beta: max {model.kernel._beta.max():.5f}, min {model.kernel._beta.min():.5f}, sigma: max {model.kernel._sigma.max():.5f}, min {model.kernel._sigma.min():.5f}, mu {model._mu.mean():.5f}")
            logout = f"[{arrow.now()}] Epoch: {i+start_epoch}\tTraining Loglik: {event_llk:.5e} stag: {no_incre} converge: {converge}"
            if l1_reg: logout += f" l1_reg_loss : {epoch_l1_loss/event_num:.5f}"
            if lnu_reg: logout += f" lnu_reg_loss : {epoch_lnu_loss/event_num:.5f}"
            print(logout)

        if testing:
            with torch.no_grad():
                loglik, lams, integ = model(test_data.to(device))
                event_loglik = loglik / (test_data[..., 0] > 0).sum()
                print(f"[{arrow.now()}] Epoch: {i+start_epoch}\tTesting Loglik: {event_loglik:.5e}")
                test_llk_out.append(float(event_loglik))

        if converge == 50:
            return train_llk_out, test_llk_out

        if save_model and (i + 1) % log_iter == 0:
            model.cpu()
            torch.save(model.state_dict(), os.path.join(path, f"{modelname}-{i+start_epoch}.pth"))
            model.to(device)

    return train_llk_out, test_llk_out
