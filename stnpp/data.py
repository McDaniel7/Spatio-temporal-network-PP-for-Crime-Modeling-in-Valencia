from __future__ import annotations

import numpy as np
import pandas as pd
import random
import datetime
from typing import Iterable, Tuple, Dict, Any


def generate_onehot_enc(n1: int, n2: int) -> np.ndarray:
    """Return concatenated one‑hot encoding basis for two label families.

    Each row has two non‑zero entries: one in the first family (size ``n1``)
    and one in the second (size ``n2``).

    Args:
        n1: Size of the first label family.
        n2: Size of the second label family.

    Returns:
        (n1*n2, n1+n2) array of concatenated one‑hot codes.
    """
    mat1 = np.eye(n1)
    mat2 = np.eye(n2)
    matt1 = np.repeat(mat1, n2, axis=0)
    matt2 = np.tile(mat2, (n1, 1))
    return np.concatenate((matt1, matt2), axis=1)


def data_pre_formatting(
    data: pd.DataFrame,
    type2int_dict: pd.DataFrame,
    nd_C_idx: np.ndarray,
    nd_dist_C: np.ndarray,
    eg_C_idx: np.ndarray,
    eg_dist_C: np.ndarray,
    year: Iterable[int] = (2015,),
    day_start: float = 0.0,
    day_end: float = 365.0,
    bins: np.ndarray | None = None,
) -> np.ndarray:
    """Convert raw event dataframe to padded tensor batches for training.

    The function filters rows to the given years, maps types to integers,
    sorts by time, splits the range [day_start, day_end] into windows defined
    by ``bins`` (shape [N, 2]), and packs each window as one sequence.
    Sequences are padded to the max length across windows.

    Expected dataframe columns (by original code)::
        'year', 'crime_type', 'crime_date_datetime', 'crime-ldmk_type',
        'crime_lon', 'crime_lat', 'grid_idx' (plus nearest‑node/edge indices).

    Args:
        data: Source dataframe of events.
        type2int_dict: Mapping frame: index=unique types, values=range(n_types).
        nd_C_idx: Nearest node indices aligned to dataframe index.
        nd_dist_C: Nearest node distances aligned to dataframe index.
        eg_C_idx: Nearest edge indices aligned to dataframe index.
        eg_dist_C: Nearest edge distances aligned to dataframe index.
        year: Iterable of years to include.
        day_start: Start day (inclusive) of global window.
        day_end: End day (inclusive) of global window.
        bins: Array of shape (N, 2) with per‑sequence [start, end] day bounds.

    Returns:
        3‑D float32 array of shape (n_seq, max_len, 9) with columns:
        (t, type, x, y, nd_idx, nd_dist, eg_idx, eg_dist, grid_idx).
    """
    assert bins is not None and bins.ndim == 2 and bins.shape[1] == 2, "Invalid bins"
    assert bins[0, 0] >= day_start and bins[-1, -1] <= day_end, "Bins out of bounds"

    random.seed(100)
    used = data[(data['year'].isin(year)) & (data['crime_type'] != 'AlarmasMujer')]
    idx = np.arange(len(used))
    sample_idx = random.sample(list(idx), int(len(idx)))
    used = used.iloc[sample_idx]

    t0 = datetime.datetime(year=min(year), month=1, day=1)
    time = (used['crime_date_datetime'] - t0).dt.total_seconds() / (24 * 3600)
    cl_type = type2int_dict.loc[used['crime-ldmk_type']].values.reshape(-1, 1)
    x = used['crime_lon'].values.reshape(-1, 1)
    y = used['crime_lat'].values.reshape(-1, 1)
    nd_idx = nd_C_idx[used.index].reshape(-1, 1)
    nd_dist = nd_dist_C[used.index].reshape(-1, 1)
    eg_idx = eg_C_idx[used.index].reshape(-1, 1)
    eg_dist = eg_dist_C[used.index].reshape(-1, 1)
    grid_idx = used['grid_idx'].values.reshape(-1, 1)

    subseq = np.hstack([time.values.reshape(-1,1), cl_type, x, y, nd_idx, nd_dist, eg_idx, eg_dist, grid_idx])
    subseq = subseq[subseq[:, 0].argsort()]
    subseq = subseq[(subseq[:, 0] >= day_start) & (subseq[:, 0] <= day_end)]

    seqs = []
    for start, end in bins:
        s = subseq[(subseq[:, 0] >= start) & (subseq[:, 0] < end)].copy()
        s[:, 0] -= start
        seqs.append(s)

    lens = [s.shape[0] for s in seqs]
    n_seq, max_len = len(seqs), (max(lens) if lens else 0)
    out = np.zeros((n_seq, max_len, 9), dtype=np.float32)
    for i, s in enumerate(seqs):
        out[i, :s.shape[0], :] = s
    return out
