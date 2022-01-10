import array
import numpy as np
import re
import glob
import pandas as pd
import pylab as plt
from collections import defaultdict
from tqdm import tqdm
from joblib import Parallel, delayed

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

data_dir = 'E:/SURFEX/EXPLORE2/DRIAS2020/{0}/{1}/{2}/day'
drainc_prefix = 'DRAINC_ISBA.bin'
runoffc_prefix = 'RUNOFFC_ISBA.bin'

def process(fichier):
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
    df = pd.DataFrame(
        df,
        index=date_range,
        columns=range(1, 9893)
    )

def build_parameters():
    parameters = []
    for couple in alls:
        data_dir_couple = data_dir.format(
            couple[0],
            couple[1],
            couple[2].upper().replace('.', '')
        )
        for fichier in glob.glob('{0}/{1}*'.format(data_dir_couple, drainc_prefix)):
            parameters.append(fichier)
    return parameters

parameters = build_parameters()
inputs = tqdm(parameters)
processes = Parallel(n_jobs=8)(delayed(process)(i) for i in inputs)

df = {}
for proc, couple in zip(processes, alls):
    df[couple] = proc





