import array
import os
import numpy as np
import re
import glob
import pandas as pd
import xarray as xr
from collections import defaultdict
from tqdm import tqdm
from joblib import Parallel, delayed


def treatment(fichier):
    years = list(map(int, re.findall('_([0-9]+)', fichier)))
    date_range = pd.date_range(
        '{0}-8-1'.format(years[0]),
        '{0}-7-31'.format(years[1])
    )
    df = []
    with open(fichier, 'rb') as f: 
        for _ in date_range:
            a = array.array('f')
            a.fromfile(f, 9892)
            a.byteswap()
            df.append(np.array(a))
    df = xr.DataArray(
        df,
        coords=[date_range, range(1, 9893)],
        dims=["time", "zone"]
    )
    df[1:, :] = df[1:, :] - df[:-1, :].values
    df = df.resample(time='M').sum()
    return df

def build_parameters(data_dir, prefix, models):
    parameters, couples = [], []
    for couple in models:
        data_dir_couple = data_dir.format(
            couple[0],
            couple[1],
            couple[2].upper().replace('.', '')
        )
        for fichier in glob.glob('{0}/{1}*'.format(data_dir_couple, prefix)):
            parameters.append(fichier)
            couples.append(
                (
                    couple[0],
                    couple[1],
                    couple[2] 
                )
            )
    return couples, parameters

def netcdf_monthly_time_series(data_dir, prefix, models,  n_jobs):
    couples, parameters = build_parameters(data_dir, prefix, models)
    inputs = tqdm(parameters)
    processes = Parallel(n_jobs=n_jobs)(delayed(treatment)(i) for i in inputs)
    
    df = defaultdict(lambda: [])
    for couple, process in zip(couples, processes):
        df[couple].append(process)
    
    dfhist = {}
    dfrcps = {}
    for couple in models:
        df[couple] = xr.concat(df[couple], dim="time")
        df[couple] = df[couple].sortby("time")
        model = couple[:2]
        period = couple[2]
        if period == 'historical' and model not in dfhist:
            dfhist[model] = df[couple]
        else:
            dfrcps[couple] = df[couple]

    for couple in dfrcps:
        model = couple[:2]
        df_couple = xr.concat(
                [dfhist[model], dfrcps[couple]],
                dim="time"
        )
        df_couple = xr.Dataset(
                {'DRAINC':df_couple}
            )
        os.makedirs('TREATMENT', exist_ok=True)
        df_couple.to_netcdf(
            "TREATMENT/{0}_{1}_{2}_{3}.nc".format(
                couple[0], couple[1], couple[2], prefix.replace('.', '_')
            )
        )

if __name__ == '__main__':
    data_dir = '/home/jvergnes/FRC/SURFEX/EXPLORE2/DRIAS2020/{0}/{1}/{2}/day'
    drainc_prefix = 'DRAINC_ISBA.BIN'
    runoffc_prefix = 'RUNOFFC_ISBA.BIN'




