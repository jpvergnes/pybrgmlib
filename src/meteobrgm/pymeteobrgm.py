import datetime
import os
try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

import numpy as np
import pandas as pd
import geopandas as gpd
import dask
import xarray
from tqdm import tqdm
from joblib import Parallel, delayed
from shapely.geometry import Polygon

import meteobrgm

dask.config.set(**{'array.slicing.split_large_chunks': True})

X = (56000, 1200000)
Y = (1613000, 2685000)
RES = 8000.
NX, NY = 143, 134
EPSG = 27572

def build_polygon_safran():
    """
    Build the shapefile
    """
    coord = pkg_resources.open_text(meteobrgm, 'coord_9892')
    df = pd.read_csv(coord, header=None, delim_whitespace=True)
    Xleft = df[4] - int(RES/2.)
    Xright = df[4] + int(RES/2.)
    Ytop = df[5] - int(RES/2.)
    Ybottom = df[5] + int(RES/2.)

    polygons = []
    for i in range(9892):
        polygons.append(
            Polygon(
                (
                    (Xleft[i], Ybottom[i]),
                    (Xright[i], Ybottom[i]),
                    (Xright[i], Ytop[i]),
                    (Xleft[i], Ytop[i]),
                    (Xleft[i], Ybottom[i])
                )
            )
        )
    return polygons

def build_shapefile_safran():
    """
    Build the shapefile safran

    Return
    ------
    geopandas.GeoDataFrame
    """
    safran = meteobrgm.build_polygon_safran()
    gdf_safran = gpd.GeoDataFrame(
        {'zone': np.arange(1, 9893)},
        geometry=safran,
        crs='EPSG:27572'
    )
    if not gdf_safran.has_sindex:
        gdf_safran.sindex
    return gdf_safran

def build_grid_safran():
    """
    Build the mask array of the SAFRAN grid.

    Returns
    -------
    numpy.array
        Array with the zone numbers from the SAFRAN grid. No data cells are 
        equal to 9999.
    """
    Xcentre, Ycentre, num_safran = meteobrgm.return_xy_safran()
    XYcentre = [(x, y) for x, y in zip(Xcentre, Ycentre)]

    raster = np.ones((NY, NX))*9999
    for i in range(NY):
        for j in range(NX):
            x = X[0] + RES/2. + j*RES
            y = Y[1] - RES/2. - i*RES
            if (x, y) in XYcentre:
                index = XYcentre.index((x, y))
                raster[i, j] = num_safran[index]
    return raster

def extract_zones_from_shapefile(shp_input):
    gdf = gpd.read_file(shp_input)
    if not gdf.has_sindex:
        gdf.sindex
    gdf = gdf.to_crs('EPSG:27572')
    safran = meteobrgm.build_shapefile_safran()
    return safran.overlay(gdf)

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

def return_xy_safran():
    """
    Return X and Y from SAFRAN

    Returns
    -------
    list
    """
    coord = pkg_resources.open_text(meteobrgm, 'coord_9892')
    df = pd.read_csv(coord, header=None, delim_whitespace=True)
    Xcentre = df[4]
    Ycentre = df[5]
    return Xcentre, Ycentre, df[1]

def read_meteo_brgm_format(fname, ystart, zones=9892, skiprows=1):
    """
    Read data from the formatted file used in BRGM for storing
    meteorological data available on the SAFRAN grid over France
     (only 1 hydrological year starting from ystart)

    Parameters
    ----------
    fname : str
        File name to read
    ystart : int
        Starting year of the file
    zones : list of int (optional)
        SAFRAN zone numbers to extract (default to 9892)
    skiprows : (optional)
        rows to skip (default to 1)
    
    Return
    ------
    pandas.DataFrame
    """
    ystart = int(ystart)
    zones = np.array(zones).astype('int')
    df = pd.read_csv(
        fname,
        skiprows=skiprows,
        delim_whitespace=True,
        header=None
    )
    df = df.iloc[:, zones - 1] # -1 car indice python
    df.columns = zones
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
    Write data in brgm format (no column date)

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

def write_meteo_brgm_format_with_date(fname, df, header='# '):
    """
    Write data in brgm format (with column date at the end)

    Parameters
    ----------
    fname : filename of file handle
    df : pandas.DataFrame
        Needs to have datetime index 
    """
    df.index.name = 'index'
    df = df.reset_index()
    dates = df.pop('index')
    df.insert(len(df.columns), 'Date', dates)
    with open(fname, 'w', newline='') as f:
        f.write(header)
        df.to_csv(f, sep=' ', index=None, date_format='%d/%m/%Y')

def write_excel_simultane_format(fname, df):
    """
    Write pandas.dataframe in excel simultane format.

    Parameters
    ----------
    fname : filename of file handle
    df : pandas.DataFrame
        Needs to have datetime index 
    """
    with open(fname, 'w', newline='') as f:
        f.write('# ')
        df.to_csv(f, sep=' ', date_format='%d/%m/%Y', index_label='Date')


class MFSafranNetcdfDataset():
    def __init__(self, paths, xdim_name='X', ydim_name='Y', parallel=False):
        indices = meteobrgm.return_indices_safran()
        df = xarray.open_mfdataset(paths, parallel)
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
                        long_name, units, year, year + 1, self.nbzones, os.path.basename(self.paths)
                        )
                    )
            meteobrgm.write_meteo_brgm_format(
                '{0}_{1}_{2}'.format(
                    paths, year, year +1
                ),
                data,
                header
            )

class ExtractSafran():
    variables = {
        'ETP': 'ETP_Jou_v2017_Safran_{0}_{1}',
        'Plu+Neige': 'Plu+Neige_Jou_v2017_Safran_{0}_{1}',
        'Pl_Neige': 'Pl_Neige_Jou_v2017_Safran_{0}_{1}',
        'Pluie_Liq': 'Pluie_Liq_Jou_v2017_Safran_{0}_{1}',
        'Temper': 'Temper_Jou_v2017_Safran_{0}_{1}',
    }
    host_dir = "\\\\brgm.fr\\Données\\Modélisation\\Hydrogéol_Aqui-FR\\Safran_v2017"
    def __init__(self, output_dir, name, zones=9892):
        self.input_dir = self.host_dir
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        self.output_dir = output_dir
        self.name = name
        self.zones = zones
    
    def treatment(self, year, key, value):
        fname = value.format(year, year + 1)
        fname = '{0}/{1}'.format(self.input_dir, fname)
        df = meteobrgm.read_meteo_brgm_format(fname, year, zones=self.zones)
        fname = '{0}/{1}_{2}_{3}_{4}'.format(
            self.output_dir, key, self.name, year, year + 1
        )
        meteobrgm.write_meteo_brgm_format_with_date(fname, df)

    def get_parameters(self, start_year, end_year):
        parameters = []
        for year in range(start_year, end_year):
            for key, value in self.variables.items():
                parameters.append((year, key, value))
        return parameters
    
    def extract_parallel_loop(self, start_year, end_year, n_jobs=1):
        parameters = self.get_parameters(start_year, end_year)
        inputs = tqdm(parameters)
        Parallel(n_jobs=n_jobs)(delayed(self.treatment)(*args) for args in inputs)


def extract_safran(output_dir, name, zones, start_year, end_year, n_jobs=1):
    """
    Extract zones SAFRAN from the files hosted on the BRGM server

    Parameters
    ----------
    output_dir : str
        Output directory where new files are stored
    name : str
        Name that is part of the new file name
    zones : list of int
        Zones to extract
    start_year : int
        First year of data
    end_year : int
        Last year of data
    n_jobs (default 1) : int
        Number of processes to execute
    """
    exs = ExtractSafran(
        output_dir,
        name,
        zones
    )
    exs.extract_parallel_loop(start_year, end_year, n_jobs)