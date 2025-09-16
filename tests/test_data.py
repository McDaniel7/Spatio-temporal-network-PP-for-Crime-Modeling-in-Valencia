import numpy as np
from stnpp.data import data_pre_formatting

def test_data_pre_formatting_shapes():
    import pandas as pd
    df = pd.DataFrame({
        'year':[2019,2019],
        'crime_type':['A','B'],
        'crime_date_datetime': pd.to_datetime(['2019-01-01','2019-01-01']),
        'crime-ldmk_type':['X','X'],
        'crime_lon':[0.0, 0.1],
        'crime_lat':[0.0, 0.1],
        'grid_idx':[0,1],
    })
    type2 = pd.DataFrame([0], index=['X'])
    nd_idx = np.array([0,1]); nd_dist = np.array([0.0, 0.2])
    eg_idx = np.array([0,1]); eg_dist = np.array([0.0, 0.3])
    bins = np.array([[0,1],[1,2]], dtype=float)
    out = data_pre_formatting(df, type2, nd_idx, nd_dist, eg_idx, eg_dist, year=[2019], day_start=0, day_end=10, bins=bins)
    assert out.shape[0] == 2 and out.shape[2] == 9
