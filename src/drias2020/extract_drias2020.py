import os
from itertools import product

from joblib import Parallel, delayed
from tqdm import tqdm

from meteobrgm import read_meteo_brgm_format, write_meteo_brgm_format_with_date

def treatment(ninput, outrep, fname, zones):
    ystart = ninput.split('_')[-2]
    df = read_meteo_brgm_format(
        ninput,
        ystart,
        zones=zones
    )
    write_meteo_brgm_format_with_date(
        '{0}/{1}'.format(outrep, fname),
        df,
    )
    
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
    Parallel(n_jobs=n_jobs)(
        delayed(treatment)(i, j, k, zones) for i, j, k in inputs
    )

