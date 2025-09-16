from __future__ import annotations
import argparse, os, yaml, numpy as np, torch, random
import pandas as pd
import geopandas as gpd
import pickle
from shapely.geometry import Polygon
from shapely.affinity import affine_transform

from .data import data_pre_formatting
from .models.base import NeuralBaseIntensity
from .models.kernels import Multivariate_Exponential_Gaussian_latent_GAT_Kernel_NWD
from .models.stnpp import MultivariateHawkesNetworkDistNeuralBaseIntensity
from .train import train_MHP_yearly


def build_model_and_data(cfg: dict, device: torch.device):
    # --- Load data (paths in cfg['data']) ---
    dcfg = cfg['data']
    outline = gpd.read_file(dcfg['outline_shp']).iloc[0, -1]
    with open(dcfg['graph_pickle'],'rb') as f:
        A = pickle.load(f)
    nd_C = np.array(A[0]); nd_dist_C = np.array(A[1]).astype(np.float32)
    eg_C = np.array(A[2]); eg_dist_C = np.array(A[3]).astype(np.float32)
    AllSPL = np.load(dcfg['allspl_npy'])

    nodes = pd.read_pickle(dcfg['nodes_pkl'])
    streets = pd.read_pickle(dcfg['streets_pkl'])
    nodes_drive = nodes['osmid'].tolist()
    edges_list = list(zip(streets['u'], streets['v'], streets['key']))
    nd_C_idx = np.array([nodes_drive.index(n) for n in nd_C])
    eg_C_idx = np.array([edges_list.index(tuple(e)) for e in eg_C])

    int_grid = np.load(dcfg['int_grid_npy'])
    int_grid_nwd = np.load(dcfg['int_grid_nwd_npy'])
    int_grid_slabel = np.load(dcfg['int_grid_slabel_npy'])
    grid_labels = np.load(dcfg['grid_labels_npy'])

    df = pd.read_pickle(dcfg['events_pkl'])
    types = np.unique(df['crime-ldmk_type'])
    type2int_dict = pd.DataFrame(np.arange(len(types)), index=types)

    # --- Time bins ---
    def make_data(years, T_span, T_used):
        bins_point = np.arange(T_span[0], T_span[1]+1, T_used[1])
        bins = np.array(list(zip(bins_point[:-1], bins_point[1:])))
        return data_pre_formatting(data=df, type2int_dict=type2int_dict,
                                   nd_C_idx=nd_C_idx, nd_dist_C=nd_dist_C,
                                   eg_C_idx=eg_C_idx, eg_dist_C=eg_dist_C,
                                   year=years, day_start=T_span[0], day_end=T_span[1], bins=bins)

    train_Y = cfg['train']['years']
    test_Y = cfg['test']['years']
    T_train = cfg['train']['T_span']
    T_test = cfg['test']['T_span']
    T_used = cfg['train']['T_used']
    train_data = make_data(train_Y, T_train, T_used)
    test_data = make_data(test_Y, T_test, T_used)

    # Prepare mu
    n_class = cfg['model']['n_class']
    mu = np.zeros(n_class)
    v, c = np.unique(train_data[train_data[:, :, 0] > 0, 1], return_counts=True)
    grid_label_v, grid_label_c = np.unique(grid_labels, return_counts=True)
    ldmk_area_prop = grid_label_c / grid_label_c.sum()
    ldmk_areas = 71 * np.tile(ldmk_area_prop, 3)
    div = cfg['train']['div']
    mu[v.astype(np.int32)] = np.clip(c / (T_used[1] - T_used[0]) / ldmk_areas / div, a_min=0.0, a_max=1e1)

    # Model pieces
    n_head = cfg['model']['n_head']; out_channel = cfg['model']['out_channel']
    alpha_mask = np.ones((n_class, n_class))
    kernel = Multivariate_Exponential_Gaussian_latent_GAT_Kernel_NWD(
        n_head=n_head, out_channel=out_channel, alpha_coef=np.ones(n_head),
        alpha=np.random.uniform(low=0., high=1., size=(n_class, n_class)),
        beta=np.array([1.0]), sigma=np.array([0.1]), sigma_l=np.random.uniform(low=.01, high=1., size=1),
        alpha_mask=alpha_mask, alpha_prior=None,
        AllSPL=torch.FloatTensor(AllSPL).to(device), device=device
    )
    neural_BI = NeuralBaseIntensity(n_class=3, embed_dim=8, mlp_layer=2, mlp_dim=32)

    model = MultivariateHawkesNetworkDistNeuralBaseIntensity(
        T=np.array(cfg['train']['T_used']), S=outline, mu=mu, data_dim=9, device=device,
        neural_BI=neural_BI, kernel=kernel, min_dist=0.01, neural_base=False,
        int_grid=torch.FloatTensor(int_grid).to(device),
        int_grid_nwd=torch.FloatTensor(int_grid_nwd).to(device),
        int_grid_slabel=torch.FloatTensor(int_grid_slabel).to(device),
    )
    model.kernel._beta.requires_grad = True
    model.kernel._sigma.requires_grad = True
    model.kernel.alpha_coef.requires_grad = True
    model._mu.requires_grad = False

    return model, torch.FloatTensor(train_data), torch.FloatTensor(test_data)


def main():
    parser = argparse.ArgumentParser(description="Train STNPP models from YAML config.")
    parser.add_argument('--config', '-c', required=True, help='Path to YAML config')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    seed = cfg['train']['seed']
    torch.random.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model, train_data, test_data = build_model_and_data(cfg, device)

    modelname = cfg['train']['modelname']
    train_llk, test_llk = train_MHP_yearly(
        model, train_data, test_data=test_data, device=device, modelname=modelname,
        num_epochs=cfg['train']['num_epochs'], lr=cfg['train']['lr'], batch_size=cfg['train']['batch_size'],
        stationary=cfg['model'].get('stationary', False),
        l1_reg=cfg['train']['l1_reg'], lam_l1=cfg['train']['lam_l1'],
        lnu_reg=cfg['train']['lnu_reg'], lam_lnu=cfg['train']['lam_lnu'],
        print_iter=cfg['train']['print_iter'], log_iter=cfg['train']['log_iter'],
        tol=cfg['train']['tol'], testing=cfg['train']['testing'],
        save_model=cfg['train']['save_model'], save_path=cfg['train']['save_path'],
        new_folder=cfg['train']['new_folder'], start_epoch=cfg['train']['start_epoch'],
    )

    print("Training complete.")


if __name__ == '__main__':
    main()
