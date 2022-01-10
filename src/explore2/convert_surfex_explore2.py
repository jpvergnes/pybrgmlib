import xarray as xr
import write_to_file
import os
from glob import iglob
from collections import defaultdict
from joblib import Parallel, delayed
from tqdm import tqdm

rempl = {
    3005:3004,
    3384:3383,
    3391:3264,
    3765:3764,
    3887:3886,
    4361:4362,
    5456:5455,
    8714:8713,
    9198:9874,
    9199:9200,
    9201:9200,
    9289:9220,
    9339:9340,
    9351:9288,
    9407:9406,
    9458:9459,
    9817:9818,
    9847:9856,
    9892:9889
}

# Scan the directories
def scan_directories(directory):
    ncs = defaultdict(list)
    for path in iglob('{0}/*/*/*/*'.format(directory)):
        p = path.split('/')
        key = (p[-4], p[-3], p[-2])
        ncs[key].append(path)
    return ncs

def treatment(nc, ncname, out_dir):
    year = int(ncname.split('_')[-3])
    champ = ncname.split('_')[-4]
    df = xr.open_dataset(ncname)
    data = df[champ]
    for key, new_key in rempl.items():
        data.sel(X=key).data = data.sel(X=new_key).data
    data = data.where(~data.isnull(), 0)
    binname = '{0}/{1}/{2}/{3}/day/{4}'.format(
        out_dir,
        nc[0], nc[1], nc[2],
        '{0}_ISBA.BIN_{1}_{2}'.format(champ, year, year + 1)
    )
    os.makedirs(os.path.dirname(binname), exist_ok=True)
    write_to_file.write_to_file(binname, data, data.shape[0], data.shape[1])

def build_parameters(ncs):
    parameters = []
    for nc, ncnames in ncs.items():
        for ncname in ncnames:
            parameters.append((nc, ncname))

def convert_surfex_explore2(in_dir, out_dir, n_jobs):
    ncs = scan_directories(in_dir)
    inputs = tqdm(build_parameters(ncs))
    Parallel(n_jobs=n_jobs)(
            delayed(treatment)(i, j, out_dir) for i, j in inputs)

if __name__ == '__main__':
    dirBRGMSurfex = '/mnt/e/EXPLORE2/MF_EXPLORE2_SURFEX'
    dirSurfex = '/mnt/e/EXPLORE2/EXPLORE2_SURFEX'
    n_jobs = 10
    convert_surfex_explore2(dirSurfex, dirBRGMSurfex, n_jobs)

