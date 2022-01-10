import numpy as np
import xarray as xr
import os, re
from glob import iglob
import pandas as pd

import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm

import meteobrgm

units = {
    'evspsblpotAdjust':'mm/jour',
    'prsnAdjust':'mm/jour',
    'prtotAdjust':'mm/jour',
    'prliqAdjust':'mm/jour',
    'tasAdjust':'°C'
}

start = {
    'ALADIN63':1951,
    'RCA4':1970,
    'RACMO22E':1950,
    'CCLM4-8-17':1950,
    'RegCM4-6':1971,
    'WRF381P':1951,
    'REMO2009':1950,
    'REMO2015':1950,
    'HIRHAM5':1951
}

def scan_directories(directory):
    netcdfs = {}
    for path in iglob('{0}/*/*/*/*/*/*'.format(directory)):
        path_split = path.split('/')
        varname = path_split[-2].split('Adjust')[0]
        if varname == 'evspsblpot':
            varname = '_'.join([varname, os.path.splitext(path_split[-1])[0].split('_')[-1]])
        key = (
            path_split[-6], path_split[-5], path_split[-4],
            path_split[-3], path_split[-2], varname
        )
        netcdfs[key] = path

    # Add liquid precipitation to compute (substraction of prsn from prtot)
    # !!! Comment since prliq /= prtot + prsn
    # for netcdf in netcdfs.copy():
    #     if netcdf[-1] in ['prtot', 'prsn']:
    #         key = (
    #             netcdf[0], netcdf[1], netcdf[2],
    #             netcdf[3], 'prliqAdjust', 'prliq'
    #         )
    #         if key not in netcdfs.keys():
    #             netcdfs[key] = []
    #         netcdfs[key].append(netcdfs[netcdf])


def open_ncfile(nc, ncname):
    if type(ncname) is list:
        for ncn in ncname:
            mf = meteobrgm.MFSafranNetcdfDataset(ncn)
            if 'prtot' in ncn:
                datatot = mf.df['prtotAdjust']
            elif 'prsn' in ncn:
                datasn = mf.df['prsnAdjust']
        data = datatot - datasn
        varName = 'prliqAdjust'
        data.attrs['standard_name'] = 'liquid_precipitation'
        data.attrs['long_name'] = 'Bias-Adjusted Liquide Precipitation'
        mf.df['prliqAdjust'] = data
    else:
        mf = meteobrgm.MFSafranNetcdfDataset(ncname)
        varName = nc[-1].split('_')[0] + 'Adjust'
        data = mf.df[varName]
    if 'tas' in varName:
        data = data - 273.15
    else:
        data = data * 86400.
    mf.df[varName] = data
    mf.df[varName].attrs['units'] = units[varName]
    return mf, varName

def write_year(nc, ncname, mf, year, varName, directory):
    nc = list(nc)
    if type(ncname) is not list:
        if 'FAO' in ncname:
            nc[-2] = nc[-2] + '_FAO'
        if 'Hg0175' in ncname:
            nc[-2] = nc[-2] + '_Hg0175'
    txtname = '{0}_{1}_{2}_{3}_{4}'.format(
        nc[-2], nc[0], nc[1], nc[2], nc[3]
    )
    path = '/'.join([directory] + list(nc[:-1]) + [txtname])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if type(ncname) is list:
        ncname = ' et '.join([os.path.basename(nca) for nca in ncname])
    else:
        ncname = os.path.basename(ncname)
    mf.convert_to_meteo_brgm_format(path, varName, year, year + 1)

def get_ncname_hist(nc, ncname):
    ncname_hist = ncname.replace(nc[2], 'historical')
    ncname_hist = ncname_hist.replace('historical_METEO', 'Historical_METEO')
    ncname_hist = re.sub(
        '[0-9]{4}-[0-9]{4}',
        '{0}-2005'.format(start[nc[1]]),
        ncname_hist
    )
    return ncname_hist

def treatment(inp, directory):
    nc, ncname = inp[0], inp[1]
    mf, varName = open_ncfile(nc, ncname)
    years = np.unique(mf.df.time.dt.year)
    if 'historical' in nc[2]:
        years = np.arange(years[0], years[-1])
    elif 'rcp' in nc[2]:
        years = np.arange(years[0] - 1, years[-1])
    for year in years:
        if year == 2005 and 'rcp' in nc[2]:
            if type(ncname) is list:
                ncname_hist = [get_ncname_hist(nc, ncna) for ncna in ncname]
            else:
                ncname_hist = get_ncname_hist(nc, ncname)
            nchist = list(nc)
            nchist[2] = 'historical'
            mf_hist, varName_hist = open_ncfile(nchist, ncname_hist)
            data = xr.concat([mf_hist.df[varName_hist], mf.df[varName]], dim='time')
            mf.df = mf.df.reindex(time=data.time)
            mf.df[varName].data = data.data
        write_year(nc, ncname, mf, year, varName, directory)

def convert_drias2020(in_dir, out_dir, n_jobs):
    netcdfs = scan_directories(in_dir, n_jobs)
    inputs = tqdm(list(netcdfs.items()))
    processed_list = Parallel(n_jobs=n_jobs)(
            delayed(treatment)(i, out_dir) for i in inputs)

if __name__ == '__main__':
    convert_drias200(
        '/home/jvergnes/DRIAS2020',
        '/home/jvergnes/BRGM_DRIAS2020',
        10
    )
