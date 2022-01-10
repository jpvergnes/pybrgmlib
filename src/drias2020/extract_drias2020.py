import os
import sys
from itertools import product

from pandas import read_csv, date_range
from numpy import array
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np

from meteobrgm import read_meteo_brgm_format


def treatment(ninput, outrep, fname, zones):
    ystart = ninput.split('_')[-2]
    df = read_meteo_brgm_format(
        ninput,
        ystart,
        zones=zones
    )
    df = df.reset_index()
    dates = df.pop('index')
    df.insert(len(df.columns), 'Date', dates)
    os.makedirs(outrep, exist_ok=True)
    with open('{0}/{1}'.format(outrep, fname), 'w', newline='') as f:
        f.write('# ')
        df.to_csv(f, sep=' ', index=None)
    
def build_parameters(baserep, selection, variables, out_rep):
    parameters  = []
    for element in product(selection, variables):
        path = '{rep}/{gcm}/{rcm}/{scenario}/day/{variable}/'.format(
                rep=baserep,
                gcm=element[0][0],
                rcm=element[0][1],
                scenario=element[0][2],
                variable=element[1]
        )
        assert os.path.exists(path)
        for r, d, f in os.walk(path):
            for fname in f:
                ninput = os.path.join(r, fname)
                noutput_rep = '{0}/{1}/{2}/{3}'.format(
                        out_rep,
                        element[0][0],
                        element[0][1],
                        element[0][2]
                )
                parameters.append((ninput, noutput_rep, fname))
        return parameters

def extract_drias2020(baserep, selection, variables, out_rep, zones, n_jobs):
    parameters = build_parameters(baserep, selection, variables, out_rep)
    inputs = tqdm(parameters)
    processed_list = Parallel(n_jobs=n_jobs)(
            delayed(treatment)(i, j, k, zones) for i, j, k in inputs)

if __name__ == '__main__':
    baserep = 'BRGM_DRIAS2020'
    alls = [
        ('CNRM-CM5-LR', 'ALADIN63', 'historical'),
        ('CNRM-CM5-LR', 'ALADIN63', 'rcp2.6'),
        ('CNRM-CM5-LR', 'ALADIN63', 'rcp4.5'),
        ('CNRM-CM5-LR', 'ALADIN63', 'rcp8.5'),
        ('CNRM-CM5-LR', 'RACMO22E', 'historical'),
        ('CNRM-CM5-LR', 'RACMO22E', 'rcp2.6'),
        ('CNRM-CM5-LR', 'RACMO22E', 'rcp4.5'),
        ('CNRM-CM5-LR', 'RACMO22E', 'rcp8.5'),
        ('EC-EARTH', 'RACMO22E', 'historical'),
        ('EC-EARTH', 'RACMO22E', 'rcp2.6'),
        ('EC-EARTH', 'RACMO22E', 'rcp4.5'),
        ('EC-EARTH', 'RACMO22E', 'rcp8.5'),
        ('EC-EARTH', 'RCA4', 'historical'),
        ('EC-EARTH', 'RCA4', 'rcp2.6'),
        ('EC-EARTH', 'RCA4', 'rcp4.5'),
        ('EC-EARTH', 'RCA4', 'rcp8.5'),
        ('HadGEM2-ES', 'CCLM4-8-17', 'historical'),
        ('HadGEM2-ES', 'CCLM4-8-17', 'rcp4.5'),
        ('HadGEM2-ES', 'CCLM4-8-17', 'rcp8.5'),
        ('HadGEM2-ES', 'RegCM4-6', 'historical'),
        ('HadGEM2-ES', 'RegCM4-6', 'rcp2.6'),
        ('HadGEM2-ES', 'RegCM4-6', 'rcp8.5'),
        ('IPSL-CM5A-MR', 'RCA4', 'historical'),
        ('IPSL-CM5A-MR', 'RCA4', 'rcp4.5'),
        ('IPSL-CM5A-MR', 'RCA4', 'rcp8.5'),
        ('IPSL-CM5A-MR', 'WRF381P', 'historical'),
        ('IPSL-CM5A-MR', 'WRF381P', 'rcp4.5'),
        ('IPSL-CM5A-MR', 'WRF381P', 'rcp8.5'),
        ('MPI-ESM-LR', 'CCLM4-8-17', 'historical'),
        ('MPI-ESM-LR', 'CCLM4-8-17', 'rcp2.6'),
        ('MPI-ESM-LR', 'CCLM4-8-17', 'rcp4.5'),
        ('MPI-ESM-LR', 'CCLM4-8-17', 'rcp8.5'),
        ('MPI-ESM-LR', 'REMO2009', 'historical'),
        ('MPI-ESM-LR', 'REMO2009', 'rcp2.6'),
        ('MPI-ESM-LR', 'REMO2009', 'rcp4.5'),
        ('MPI-ESM-LR', 'REMO2009', 'rcp8.5'),
        ('NorESM1-M', 'HIRHAM5', 'historical'),
        ('NorESM1-M', 'HIRHAM5', 'rcp4.5'),
        ('NorESM1-M', 'HIRHAM5', 'rcp8.5'),
        ('NorESM1-M', 'REMO2015', 'historical'),
        ('NorESM1-M', 'REMO2015', 'rcp2.6'),
        ('NorESM1-M', 'REMO2015', 'rcp8.5')
    ]
    
    short_list_mf = [
        ('CNRM-CM5-LR', 'ALADIN63', 'historical'),
        ('CNRM-CM5-LR', 'ALADIN63', 'rcp2.6'),
        ('CNRM-CM5-LR', 'ALADIN63', 'rcp4.5'),
        ('CNRM-CM5-LR', 'ALADIN63', 'rcp8.5'),
        ('EC-EARTH', 'RACMO22E', 'historical'),
        ('EC-EARTH', 'RACMO22E', 'rcp2.6'),
        ('EC-EARTH', 'RACMO22E', 'rcp4.5'),
        ('EC-EARTH', 'RACMO22E', 'rcp8.5'),
        ('HadGEM2-ES', 'CCLM4-8-17', 'historical'),
        ('HadGEM2-ES', 'CCLM4-8-17', 'rcp4.5'),
        ('HadGEM2-ES', 'CCLM4-8-17', 'rcp8.5'),
        ('IPSL-CM5A-MR', 'WRF381P', 'historical'),
        ('IPSL-CM5A-MR', 'WRF381P', 'rcp4.5'),
        ('IPSL-CM5A-MR', 'WRF381P', 'rcp8.5'),
        ('MPI-ESM-LR', 'CCLM4-8-17', 'historical'),
        ('MPI-ESM-LR', 'CCLM4-8-17', 'rcp2.6'),
        ('MPI-ESM-LR', 'CCLM4-8-17', 'rcp4.5'),
        ('MPI-ESM-LR', 'CCLM4-8-17', 'rcp8.5'),
    ]
    variables = [
            'prtotAdjust',
            'evspsblpotAdjust_Hg0175',
            'prsnAdjust',
            'tasAdjust',
    ]
    out_rep = '.'
    zones = [1, 2, 3]
    n_jobs = 40
    
    selection = short_list_mf
    extract_drias2020(baserep, selection, variables, out_rep, zones, n_jobs)

