import datetime
try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

import numpy as np
import pandas as pd
import dask
import xarray
import pylab as plt

import meteobrgm

dask.config.set(**{'array.slicing.split_large_chunks': True})

X = (56000, 1200000)
Y = (1613000, 2685000)
RES = 8000.
NX, NY = 143, 134
EPSG = 27572

def build_grid_safran():
    """
    Build the mask array of the SAFRAN grid.

    Returns
    -------
    numpy.array
        Array with the zone numbers from the SAFRAN grid. No data cells are 
        equal to 9999.
    """
    coord = pkg_resources.open_text(meteobrgm, 'coord_9892')
    df = pd.read_csv(coord, header=None, delim_whitespace=True)
    Xcentre = df[4]
    Ycentre = df[5]
    XYcentre = [(x, y) for x, y in zip(Xcentre, Ycentre)]
    num_safran = df[1]

    raster = np.ones((NY, NX))*9999
    for i in range(NY):
        for j in range(NX):
            x = X[0] + RES/2. + j*RES
            y = Y[1] - RES/2. - i*RES
            if (x, y) in XYcentre:
                index = XYcentre.index((x, y))
                raster[i, j] = num_safran[index]
    return raster

def return_indices_safran(return_raster=False):
    """
    Return indices X and Y from SAFRAN

    Returns
    -------
    indices : list of tuple (int, int)
    raster (optional) : numpy.array
    """
    raster = meteobrgm.build_grid_safran()
    yj, xi  = np.indices(raster.shape)
    yj = yj + 1
    xi = xi + 1
    xi = np.ma.masked_where(raster == 9999., xi).compressed()
    yj = np.ma.masked_where(raster == 9999., yj).compressed()
    indices = [(j, i) for j, i in zip(yj, xi)]
    if return_raster:
        return indices, raster
    else:
        return indices

def read_meteo_brgm_format(fname, ystart, **kwargs):
    """
    Read data from the formatted file used in BRGM for storing
    meteorological data available on the SAFRAN grid over France
     (only 1 hydrological year starting from ystart)

    Parameters
    ----------
    fname: str
        File name to read
    ystart: int
        Starting year of the file
    
    Return
    ------
    pandas.DataFrame

    Other Parameters
    ----------------
    **kwargs: other properties, optional
        *kwargs* are used to specify the following optional properties:
        
        skiprows: rows to skip (default to 1)
        zones: SAFRAN zone numbers to extract (default to 9892)
    """
    ystart = int(ystart)
    num_zones = kwargs.get('zones', list(range(1, 9893)))
    num_zones = np.array(num_zones).astype('int')
    skiprows = kwargs.get('skiprows', 1)
    df = pd.read_csv(
        fname,
        skiprows=skiprows,
        delim_whitespace=True,
        header=None
    )
    df = df.iloc[:, num_zones - 1] # -1 car indice python
    df.columns = num_zones
    df.columns.name = 'Zone'
    df.index = pd.date_range(
        '{0}-8-1'.format(ystart),
        '{0}-7-31'.format(ystart + 1),
    )
    return df

def read_meteofrance_format(fname, zones, variables=['PS', 'PL', 'ETP', 'T']):
    """
    Read the SAFRAN data provided by Météo France and extract
    the requested zones.

    Parameters
    ----------
    fname: str
        File name to read
    zones: list of integers
        List of the zone numbers to extract
    variables: list of str, default=['PS', 'PL', 'ETP', 'T']
        List of variables as they appear in columns in the file.
    
    Return
    ------
    pandas.DataFrame
    """
    date_parser = lambda x: datetime.datetime.strptime(x, '%Y%m%d')
    df = pd.read_csv(fname, delimiter=';', header=None, parse_dates=True,
        date_parser=date_parser, index_col=0)
    champs = ['Zone'] + variables
    df.columns = pd.Index(champs)
    df = df.pivot(columns='Zone', values=champs[1:])
    df.columns = df.columns.set_names('Champ', 0)
    df.index = df.index.set_names('Date')
    selection = pd.MultiIndex.from_product([champs[1:], zones],
        names=['Champ', 'Zone'])
    return df[selection]

def write_meteo_brgm_format(fname, data, header):
    """
    Write data in brgm format

    Parameters
    ----------
    fname : filename of file handle
    data : 1D or 2D array_like     
    header : str
    """
    np.savetxt(fname,
            data,
            header=header,
            delimiter=' ',
            fmt='%.3f')


class MFSafranNetcdfDataset():
    def __init__(self, paths, parallel=False):
        indices = meteobrgm.return_indices_safran()
        df = xarray.open_mfdataset(paths, parallel)
        xdim_name, ydim_name = 'X', 'Y'
        if 'i' in df.coords.dims and 'j' in df.coords.dims:
            xdim_name, ydim_name = 'i', 'j'
        df[xdim_name] = np.arange(1, 144)
        df[ydim_name] = np.arange(134, 0, -1)
        df = df.stack(NUM_ZONE=(ydim_name, xdim_name))
        df = df.loc[{'NUM_ZONE':indices}]
        self.df = df.fillna(0)
        self.nbzones = len(indices)
        self.paths = paths

    def get_hydrological_year(self, variable, year):
        return self.df[variable].sel(
            time=slice(
                '{0}-8-1'.format(year),
                '{0}-7-31'.format(year + 1)
            ),
        )

    def convert_to_meteo_brgm_format(self, paths, variable, year_start, year_end):
        """
        Convert netcdf file in brgm format

        Parameters
        ----------
        paths : str or list of str
        variable : str
        year_start : int
        year_end : int
        """
        for year in range(year_start, year_end):
            data = self.get_hydrological_year(variable, year)
            long_name = variable
            units = 'Unknown'
            if hasattr(data, 'long_name'):
                long_name = data.long_name
            if hasattr(data, 'units'):
                units = data.units
            header = (
                    "Données de la variable {0} convertie en {1} pour l'année hydrologique "
                    "{2}/{3} des {4} zones du ou des fichiers :"
                    "{5}".format(
                        long_name, units, year, year + 1, self.nbzones, self.paths
                        )
                    )
            meteobrgm.write_meteo_brgm_format(
                '{0}_{1}_{2}'.format(
                    paths, year, year +1
                ),
                data,
                header
            )