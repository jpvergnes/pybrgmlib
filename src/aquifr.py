import os

import xarray as xr
import dask.array as da
from dask.diagnostics import ProgressBar
import pandas as pd
import numpy as np
import geopandas as gpd
import pylab as plt
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.stats import norm
from scipy.interpolate import interp1d
from KDEpy import FFTKDE

import cloneades
import meteobrgm

models = [
    'mart_som',
    'mart_npc',
    'mart_als',
    'mart_poc',
    'mart_teg',
    'mart_bno',
    'odic_seine-oise',
    'odic_seine-eure',
    'odic_marne-loing',
    'odic_marne-oise',
    'odic_seine',
    'odic_loire'
]

stations_filename = {
    'mart_som':'D:/Documents/vergnes/OneDrive - BRGM/Projets/EXPLORE2/AquiFR/StationsDebits/Stations_Explore2/stations_som',
    'mart_npc':'D:/Documents/vergnes/OneDrive - BRGM/Projets/EXPLORE2/AquiFR/StationsDebits/Stations_Explore2/stations_npc',
    'mart_als':'D:/Documents/vergnes/OneDrive - BRGM/Projets/EXPLORE2/AquiFR/StationsDebits/Stations_Explore2/stations_als',
    'mart_poc':'D:/Documents/vergnes/OneDrive - BRGM/Projets/EXPLORE2/AquiFR/StationsDebits/Stations_Explore2/stations_poc',
    'mart_teg':'D:/Documents/vergnes/OneDrive - BRGM/Projets/EXPLORE2/AquiFR/StationsDebits/Stations_Explore2/stations_teg',
    'mart_bno':'D:/Documents/vergnes/OneDrive - BRGM/Projets/EXPLORE2/AquiFR/StationsDebits/Stations_Explore2/stations_bno',
    'odic_seine-oise':'D:/Documents/vergnes/OneDrive - BRGM/Projets/EXPLORE2/AquiFR/StationsDebits/Stations_AquiFR_V1.3/stations_seine-oise',
    'odic_seine-eure':'D:/Documents/vergnes/OneDrive - BRGM/Projets/EXPLORE2/AquiFR/StationsDebits/Stations_AquiFR_V1.3/stations_seine-eure',
    'odic_marne-loing':'D:/Documents/vergnes/OneDrive - BRGM/Projets/EXPLORE2/AquiFR/StationsDebits/Stations_AquiFR_V1.3/stations_marne-loing',
    'odic_marne-oise':'D:/Documents/vergnes/OneDrive - BRGM/Projets/EXPLORE2/AquiFR/StationsDebits/Stations_AquiFR_V1.3/stations_marne-oise',
    'odic_seine':'D:/Documents/vergnes/OneDrive - BRGM/Projets/EXPLORE2/AquiFR/StationsDebits/Stations_AquiFR_V1.3/stations_seine',
    'odic_loire':'D:/Documents/vergnes/OneDrive - BRGM/Projets/EXPLORE2/AquiFR/StationsDebits/Stations_AquiFR_V1.3/stations_loire',
}


models_crs = {
    'L2E': 'EPSG:27572',
    'L93': 'EPSG:2154',
    'UTM32N': 'EPSG:32632'
}

models_dict = {
    'odic_loire':{
        'name': 'Loire',
        'nb_layers' : 3,
        'nb_cells': 37620,
        'nb_rivcells': 16141,
        'code': 'loire',
        'model': 'odic',
        'spatial_ref': 'L2E',
        'aquifers': {
            'nb_layers' : 3,
            'names': ['Beauce', 'Craie', 'Cenomanien'],
            'nb_cells': [6738, 15448, 15434],
            'type': ['aquifer', 'aquifer', 'aquifer'],
        }
    },
    'odic_basse-normandie':{
        'name': 'Basse-Normandie',
        'nb_layers' : 4,
        'nb_cells': 37667,
        'nb_rivcells': 12371,
        'code': 'basse-normandie',
        'model': 'odic',
        'spatial_ref': 'L2E',
        'aquifers': {
            'nb_layers' : 4,
            'names': ['Craie', 'Bathonien', 'Bajocien', 'Dogger Sud'],
            'nb_cells': [2939, 14446, 17392, 2890],
            'type': ['aquifer', 'aquifer', 'aquifer', 'aquifer'],
        }
    },
    'odic_seine':{
        'name': 'Seine',
        'nb_layers' : 6,
        'nb_cells': 41609,
        'nb_rivcells': 6481,
        'code': 'seine',
        'model': 'odic',
        'spatial_ref': 'L2E',
        'aquifers': {
            'nb_layers' : 6,
            'names': [
                'Beauce', 'Brie', 'Champigny', 'Lutetien', 'Thanetien', 'Craie'
            ],
            'nb_cells': [1707, 2989, 3050, 6300, 3022, 24541],
            'type': [
                'aquifer', 'aquifer', 'aquifer', 'aquifer', 'aquifer', 'aquifer'
            ],
        }
    },
    'odic_seine-eure':{
        'name': 'Seine-Eure',
        'nb_layers' : 1,
        'nb_cells': 57306,
        'nb_rivcells': 6980,
        'code': 'seine-eure',
        'model': 'odic',
        'spatial_ref': 'L2E',
        'aquifers': {
            'nb_layers' : 1,
            'names': ['Craie'],
            'nb_cells': [57306],
            'type': ['aquifer'],
        }
    },
    'odic_seine-oise':{
        'name': 'Seine-Oise',
        'nb_layers' : 4,
        'nb_cells': 87178,
        'nb_rivcells': 13021,
        'code': 'seine-oise',
        'model': 'odic',
        'spatial_ref': 'L2E',
        'aquifers': {
            'nb_layers' : 4,
            'names': ['Lutetien', 'Cuisien', 'Thanetien', 'Craie'],
            'nb_cells': [5480, 6829, 11508, 63361],
            'type': ['aquifer', 'aquifer', 'aquifer', 'aquifer'],
        }
    },
    'odic_marne-oise':{
        'name': 'Marne-Oise',
        'nb_layers' : 2,
        'nb_cells': 45904,
        'nb_rivcells': 8762,
        'code': 'marne-oise',
        'model': 'odic',
        'spatial_ref': 'L2E',
        'aquifers': {
            'nb_layers' : 2,
            'names': ['Tertiaire', 'Craie'],
            'nb_cells': [1953, 43951],
            'type': ['aquifer', 'aquifer'],
        }
    },
    'odic_marne-loing':{
        'name': 'Marne-Loing',
        'nb_layers' : 4,
        'nb_cells': 66235,
        'nb_rivcells': 9821,
        'code': 'marne-loing',
        'model': 'ODIC',
        'spatial_ref': 'L2E',
        'aquifers': {
            'nb_layers' : 4,
            'names': ['Tertiaire', 'Craie', 'Turonien', 'Turonien-Sud'],
            'nb_cells': [10086, 1886, 929, 53334],
            'type': ['aquifer', 'aquifer', 'aquifer', 'aquifer'],
        }
    },
    'odic_somme':{
        'name': 'Somme',
        'nb_layers' : 1,
        'nb_cells': 63226,
        'nb_rivcells': 10871,
        'code': 'somme',
        'model': 'odic',
        'spatial_ref': 'L2E',
        'aquifers': {
            'nb_layers' : 1,
            'names': ['Craie'],
            'nb_cells': [63226],
            'type': ['aquifer'],
        }
    },
    'mart_als':{
        'name': 'Alsace',
        'nb_layers' : 3,
        'nb_cells': 40947,
        'dx_min' : 125,
        'nb_rivcells': 1018,
        'code': 'als',
        'model': 'mart',
        'spatial_ref': 'UTM32N',
        'aquifers': {
            'nb_layers' : 3,
            'names': ['Quaternaire', 'Pliocene median', 'Pliocene profond'],
            'nb_cells': [14627, 14393, 11927],
            'type': ['aquifer', 'aquifer', 'aquifer'],
        }
    },
    'mart_bno': {
        'name': 'Basse-Normandie',
        'nb_layers': 10,
        'nb_cells': 93800,
        'nb_rivcells': 1523,
        'code': 'bno',
        'model': 'mart',
        'spatial_ref': 'L93',
        'aquifers': {
            'nb_layers' : 10,
            'names': [
                'Alluvions-Cenozoique', 'Craie', 'Infra-Cenomanien',
                'Oxfordien', 'Callovo-Oxfordien',
                'Bathonien moyen-superieur', 'Bathonien moyen-inferieur',
                'Bajocien-Aalenien-Toarcien', 'Infra-Toarcien', 'Socle'
            ],
            'nb_cells': [
                2513, 1435, 1649, 1580, 6431,
                15567, 14217, 16307, 14511, 19590
            ],
            'type': [
                'aquifer', 'aquifer', 'aquifer',
                'aquifer', 'aquitard', 'aquifer',
                'aquitard', 'aquifer', 'aquifer',
                'aquifer'
            ]
        }
    },
    'mart_npc': {
        'name': 'Nord-Pas-de-Calais',
        'nb_layers' : 10,
        'nb_cells': 226077,
        'nb_rivcells': 3317,
        'code': 'npc',
        'model': 'mart',
        'spatial_ref': 'L2E',
        'aquifers': {
            'nb_layers' : 10,
            'names': [
                'Alluvions-sables-littoraux', 'Limons',
                'Argiles des Flandres-Ypresien', 'Sables Ostricourt-Landenien',
                'Argiles de Louvil', 'Craie du Seno-Turonien productive',
                'Craie du Seno-Turonien non productive', 'Dieves',
                'Cenomanien', 'Carbonifere'
            ],
            'nb_cells': [
                4974, 26210, 9780, 15954, 17095,
                49868, 21265, 43049, 35265, 2617
            ],
            'type': [
                'aquifer', 'semi-perméable', 'aquitard',
                'aquifer', 'aquitard', 'aquifer',
                'aquitard', 'aquifer', 'aquifer',
                'aquifer'
            ]
        }
    },
    'mart_poc':{
        'name': 'Poitou-Charentes',
        'nb_layers' : 8,
        'nb_cells': 90084,
        'dx_min' : 1000,
        'nb_rivcells': 2481,
        'code': 'poc',
        'model': 'mart',
        'spatial_ref': 'L2E',
        'aquifers': {
            'nb_layers' : 8,
            'names': [
                'Bri du marais', 'Cretaces et alterites',
                'Jurassique superieur altere',
                'Jurassique superieur non altere',
                'Dogger', 'Toarcien', 'Infra-Toarcien', 'Socle'
            ],
            'nb_cells': [1325, 4667, 8541, 8353, 15139, 16070, 16666, 19323],
            'type': [
                'aquitard', 'aquifer', 'aquifer',
                'aquitard', 'aquifer', 'aquitard',
                'aquifer', 'aquifer'
            ],
        }
    },
    'mart_som':{
        'name': 'Somme',
        'nb_layers' : 1,
        'nb_cells': 66924,
        'nb_rivcells': 4638,
        'code': 'som',
        'model': 'mart',
        'spatial_ref': 'L2E',
        'aquifers': {
            'nb_layers' : 1,
            'names': ['Craie'],
            'nb_cells': [66924],
            'type': ['aquifer'],
        }
    },
    'mart_teg':{
        'name': 'Tarn-et-Garonne',
        'nb_layers' : 2,
        'nb_cells': 36442,
        'nb_rivcells': 2539,
        'code': 'teg',
        'model': 'mart',
        'spatial_ref': 'L93',
        'aquifers': {
            'nb_layers' : 2,
            'names': ['Alluvial 1', 'Alluvial 2'],
            'nb_cells': [18221, 18221],
            'type': ['aquifer', 'aquifer'],
        }
    }
}

def piezometer_from_netcdf(nc_filename):
    """
    Open a netcdf file and return simulated piezometers

    Parameter
    ---------
    nc_filename: str

    Return
    ------
    pandas.DataFrame
    """
    df = xr.open_mfdataset(nc_filename)
    df = df.swap_dims({'pzo':'codes_piezos'})
    df = df[['h_piezos', 'X_pzo', 'Y_pzo']]
    # df = df['h_piezos'].to_series().unstack('codes_piezos')
    return df

def gauging_station_from_netcdf(nc_filename, txt_filename):
    """
    Open a daily netcdf file and return simulated riverflows 
    based on an Aqui-FR text filename

    Parameter
    ---------
    nc_filename: str
        Daily netcdf fil
    txt_filename: str

    Return
    ------
    pandas.DataFrame
    """
    df = xr.open_dataset(nc_filename)
    stations = pd.read_csv(
        txt_filename,
        sep=',',
        index_col=0
    )
    df = df['q'].sel(riv=stations['INDICE'])
    df = df.to_series().unstack('riv')
    df.columns = stations.index
    df.columns.name = 'codes_hydros'
    return df

def return_aquifer_outcrop(model, shp_dir):
    """
    Return aquifer outcrop for a Aqui-FR model

    Parameters
    ----------
    model: str
        Example : 'mart_som'
    shp_dir: str
        Directory where all the layer shapefiles lies (ex : mart_som_c1.shp, etc.)

    Return
    ------
    geopandas.GeoDataFrame
    """
    nb_layer = models_dict[model]['aquifers']['nb_layers']
    types = models_dict[model]['aquifers']['type']
    code = models_dict[model]['code']
    shapefile = '{0}/{1}_c1.shp'.format(shp_dir, code)
    gdf = gpd.read_file(shapefile)
    gdf = gdf[['id_maille', 'geometry']]
    gdf['Layer'] = 1
    gdf['Aquifer'] = 1
    if types[0] != 'aquifer':
        gdf['Aquifer'] = 0
    if nb_layer > 1:
        for layer in range(2, nb_layer + 1):
            shapefile = '{0}/{1}_c{2}.shp'.format(shp_dir, code, layer)
            gdf_under = gpd.read_file(shapefile)
            gdf_under = gdf_under[['id_maille', 'geometry']]
            res_difference = gdf_under.overlay(gdf, how='difference')
            res_difference['Layer'] = layer
            res_difference['Aquifer'] = 1
            if types[layer-1] != 'aquifer':
                res_difference['Aquifer'] = 0
            gdf = pd.concat([gdf, res_difference])
    return gdf

def return_first_aquifer_layer(model, shp_dir):
    """
    Return first aquifer layer for a Aqui-FR model

    Parameters
    ----------
    model: str
        Example : 'mart_som'
    shp_dir: str
        Directory where all the layer shapefiles lies (ex : mart_som_c1.shp, etc.)

    Return
    ------
    geopandas.GeoDataFrame
    """
    nb_layer = models_dict[model]['aquifers']['nb_layers']
    types = models_dict[model]['aquifers']['type']
    code = models_dict[model]['code']
    for layer_1 in range(1, nb_layer + 1):
        if types[layer_1 - 1] == 'aquifer':
            break
    shapefile = '{0}/{1}_c{2}.shp'.format(shp_dir, code, layer_1)
    gdf = gpd.read_file(shapefile)
    gdf = gdf[['id_maille', 'geometry']]
    gdf['Layer'] = layer_1
    if nb_layer > 1:
        for layer in range(layer_1 + 1, nb_layer + 1):
            if types[layer - 1] == 'aquifer':
                shapefile = '{0}/{1}_c{2}.shp'.format(shp_dir, code, layer)
                gdf_under = gpd.read_file(shapefile)
                gdf_under = gdf_under[['id_maille', 'geometry']]
                res_difference = gdf_under.overlay(gdf, how='difference')
                res_difference['Layer'] = layer
                gdf = pd.concat([gdf, res_difference])
    return gdf

def return_indices_first_aquifr_layer(shp_dir, models, save_shp=''):
    """
    """
    gdf = gpd.read_file('{0}/{1}_aquifer_layer.shp'.format(shp_dir, models[0]))
    gdf['model'] = models[0]
    for model in models[1:]:
        gdf_under = gpd.read_file('{0}/{1}_aquifer_layer.shp'.format(shp_dir, model))
        gdf_under = gdf_under.to_crs('EPSG:27572')
        res_difference = gdf_under.overlay(gdf, how='difference')
        res_difference['model'] = model
        gdf = pd.concat([gdf, res_difference])
    if save_shp:
        gdf.sindex
        gdf.to_file(save_shp)
    return gdf

def save_all_model_outcrops(shp_dir, out_dir):
    """
    Compute and save all the model outcrops in shapefile formats
    
    Parameter
    ---------
    shp_dir: str
        Directory where all the layer shapefiles lie
    out_dir: str
        Directory where the resulting layer shapefiles goes
    """
    for model in models:
        gdf = return_aquifer_outcrop(model, shp_dir)
        gdf.to_file('{0}/{1}_outcrop.shp'.format(out_dir, model), mode='w')

def save_all_model_first_aquifer_layers(shp_dir, out_dir):
    """
    Compute and save all the model first aquifer layers in shapefile formats
    
    Parameter
    ---------
    shp_dir: str
        Directory where all the layer shapefiles lie
    out_dir: str
        Directory where the resulting layer shapefiles goes
    """
    for model in models:
        gdf = return_first_aquifer_layer(model, shp_dir)
        gdf.to_file(
            '{0}/{1}_aquifer_layer.shp'.format(out_dir, model), mode='w'
        )


class ConvertSingleSimAquiFR():
    def __init__(self, name, results_dir, new_results_dir):
        self.name = name
        self.sim_dir = '{0}/{1}'.format(results_dir, name)
        self.new_sim_dir = '{0}/{1}'.format(new_results_dir, name)
        if not os.path.isdir(self.new_sim_dir):
            os.makedirs(self.new_sim_dir)


    def convert_h_monthly_to_netcdf(self, shp):
        """
        Convert simulated water level of an AquiFR simulation
        into hydrological yearly netdf files

        Parameters
        ----------
        shp: str
        """
        gdf = gpd.read_file(shp)
        dfs = []
        nbs = 0
        for model in models:
            df = xr.open_mfdataset(
                '{0}/{1}_mon.nc'.format(self.sim_dir, model),
                parallel=True,
                engine='h5netcdf'
            )
            df = df.sel(sou=gdf.loc[gdf['model'] == model, 'id_maille'] - 1)
            nbe = nbs + len(df['sou'])
            df['sou'] = np.arange(nbs, nbe)
            nbs = nbe
            dfs.append(df)
        dfs = xr.concat(dfs, dim='sou')
        datasets, paths = [], []
        for year in np.unique(dfs['time'].dt.year)[:-1]:
            year = int(year)
            datasets.append(
                dfs.sel(
                    time=slice(
                        '{0}-8-1'.format(year), '{0}-7-31'.format(year + 1)
                    )
                )
            )
            paths.append(
                '{0}/{1}_H_MONTHLY_{2}_{3}.nc'.format(
                    self.new_sim_dir,
                    self.name,
                    year,
                    year+1
                )
            )
        xr.save_mfdataset(datasets, paths)

    def convert_hpiezo_daily_to_netcdf(self):
        """
        Convert daily simulated water level of AquiFR
        into a single netcdf file

        Parameters
        ----------
        year1, year2: int
            Start and end year
        """
        df2 = []
        for model in models:
            filename = '{0}/{1}_day.nc'.format(self.sim_dir, model)
            df2.append(piezometer_from_netcdf(filename))
        df2 = xr.concat(df2, dim='codes_piezos')
        df2 = df2.drop_duplicates(dim='codes_piezos', keep='first')
        year1 = int(df2['time'][0].dt.year)
        year2 = int(df2['time'][-1].dt.year)
        df2 = df2.set_coords(['X_pzo', 'Y_pzo'])
        df2.to_netcdf(
            '{0}/{1}_HPIEZO_DAILY_{2}_{3}.nc'.format(
                self.new_sim_dir,
                self.name,
                year1,
                year2
            ),
            mode='w'
        )

    def convert_qstation_daily_to_netcdf(self, year1, year2):
        """
        Convert daily simulated river flow of AquiFR
        into a single netcdf file

        Parameters
        ----------
        year1, year2: int
            Start and end year
        """
        df2 = []
        for model in models:
            filename = '{0}/{1}_day.nc'.format(self.sim_dir, model)
            df2.append(gauging_station_from_netcdf(filename, stations_filename[model]))
        df2 = pd.concat(df2, axis=1)
        df2 = df2.iloc[:, ~df2.columns.duplicated(keep='first')]
        df2 = df2.stack().to_xarray()
        df2 = df2.sel(
            time=slice('{0}-8-1'.format(year1), '{0}-7-31'.format(year2))
        )
        df2.name = 'qstation'
        df2.to_netcdf(
            '{0}/{1}_QSTATION_DAILY_{2}_{3}.nc'.format(
                self.new_sim_dir,
                self.name,
                year1,
                year2
            ),
            mode='w'
        )
    
    def convert_h_monthly_to_1km(self, shp_dir):
        """
        Convert to 1 km.
        """
        gdf_safran = gpd.read_file('{0}/grille_1km.shp'.format(shp_dir))
        gdf = gpd.read_file(
            '{0}/AquiFR_first_aquifer_models_weight_zone.shp'.format(shp_dir)
        )
        gdf = gdf.set_index('sou')
        gdf = gdf.sort_index()

        dfs = xr.open_mfdataset(
            "{0}/{1}_H_MONTHLY*.nc".format(
                self.new_sim_dir, self.name
            ),
            parallel=True,
            engine='h5netcdf'
        )
        dfs = dfs.sel(sou=gdf.index)
        dfs = dfs.assign(weight=gdf['weight'])

        h_weighted = dfs.h * dfs.weight
        h_weighted = h_weighted.assign_coords({'zone': gdf['zone']})
        h_weighted = h_weighted.load()
        h_weighted = h_weighted.groupby('zone').sum()
        weight = dfs['weight'].assign_coords({'zone':gdf['zone']})
        weight = weight.groupby('zone').sum()
        h_weighted = h_weighted / weight
        
        gdf_h_weighted = gdf_safran.set_index('zone').loc[h_weighted['zone']]
        h_weighted = h_weighted.assign_coords(
            dict(
                x=gdf_h_weighted.centroid.x,
                y=gdf_h_weighted.centroid.y
            )
        )
        h_weighted = h_weighted.to_dataset(name='h')
        datasets, paths = [], []
        for year in np.unique(h_weighted['time'].dt.year)[:-1]:
            year = int(year)
            datasets.append(
                h_weighted.sel(
                    time=slice(
                        '{0}-8-1'.format(year), '{0}-7-31'.format(year + 1)
                    )
                )
            )
            paths.append(
                '{0}/{1}_H_MONTHLY_1KM_{2}_{3}.nc'.format(
                    self.new_sim_dir,
                    self.name,
                    year,
                    year+1
                )
            )
        xr.save_mfdataset(datasets, paths)

def write_1km_grid_safran(shp_dir):
    """
    Create a 1-km resolution grid based on SAFRAN
    """
    gdf_safran = meteobrgm.build_shapefile_safran(res=1000.)
    gdf_safran = gdf_safran.to_crs('EPSG:27572')
    gdf_safran.sindex
    gdf_safran['area_saf'] = gdf_safran.area
    gdf_safran.to_file('{0}/grille_1km.shp'.format(shp_dir))

def write_1km_grid_shapefiles(shp_dir):
    """
    Write intermediate shapefiles for converting in 1-km grid
    """
    gdf_safran = gpd.read_file('{0}/grille_1km.shp'.format(shp_dir))
    gdf_safran.sindex
    gdf = gpd.read_file('{0}/AquiFR_first_aquifer_models.shp'.format(shp_dir), crs='EPSG:27572')
    gdf.index.name = 'sou'
    gdf.sindex
    gdf = gdf.reset_index()
    temp = gdf.overlay(gdf_safran, how='intersection')
    temp['weight'] = temp.area / temp['area_saf']
    temp = temp.set_index('sou')
    temp = temp.sort_index()
    temp.sindex
    temp.to_file('{0}/AquiFR_first_aquifer_models_weight_zone.shp'.format(shp_dir))

def compute_ips_from_xarray(df, dim_name, ref_period=None, njobs=1, **kwargs):
    """
    Compute IPS from a monthly xarray dataset

    Parameters
    ----------
    df : xarray.Dataset
    ref_period : list (optional)
        list of two years as int

    Return
    ------
    pandas.DataFrame
        Computed IPS over the whole serie
    """
    df = df.dropna(dim='time', how='all')
    dfref = df
    if type(ref_period) is list:
        dfref = df.sel(
            time=slice(str(ref_period[0]), str(ref_period[1]))
        )
    def single_ips(dfref, month, sou):
        item_ref = dfref.sel(
            {
                'time':(dfref.time.dt.month == month),
                dim_name:sou
            }
        )
        data = df.sel(
            {
                'time':(df.time.dt.month == month),
                dim_name:sou
            }
        )
        data_ref = item_ref.dropna(dim='time')
        min1 = kwargs.get('min', data_ref.min())
        max1 = kwargs.get('max', data_ref.max())
        if max1 - min1 >= 1:
            min1 = np.round(min1, 1) - 0.5
            max1 = np.round(max1, 1) + 0.5
        else:
            min1 -= 1
            max1 += 1
        data_ref = data_ref.values
        data_ref = np.maximum(data_ref, min1.values)
        data_ref = np.minimum(data_ref, max1.values)
        kde = FFTKDE(kernel='epa', bw='silverman')
        x, y = kde.fit(np.unique(data_ref.round(2))).evaluate()
        y = np.cumsum(y) / np.sum(y)

        f = interp1d(x, y, bounds_error=False, fill_value='extrapolate')
        data_f = np.maximum(f(data), 0.001)
        data_f = np.minimum(data_f, 0.999)
        spli = norm.ppf(data_f)
        df.loc[
            {
                'time':(df.time.dt.month == month),
                dim_name:sou
            }
        ] = spli
    index = pd.MultiIndex.from_product(
        (
            df.time.dt.month.data, df[dim_name].data
        ),
        names=['month', dim_name]
    )
    inputs = tqdm(index)
    Parallel(n_jobs=njobs, backend='threading')(
        delayed(single_ips)(
            dfref,
            month, sou
        ) for month, sou in inputs
    )
    return df

def set_netcdf_obs(obs_dir, codes, new_file):
    """
    Set the observation netcdf

    Parameters
    ----------
    obs_dir: str
        Directory where lies the csv
    codes: list
        BSS codes
    new_file: str
        Absolute path name of the new netcdf file
    """
    df_obs = {}
    for code_piezo in codes:
        print(code_piezo)
        df_obs[code_piezo] = cloneades.read_csv_chronique(
            '{0}/{1}.csv'.format(
                obs_dir,
                code_piezo.replace('/', '_')
            )
        )['val_calc_ngf'].resample('D').mean()
    df_obs = pd.concat(df_obs, axis=1)
    df_obs.columns.name = 'codes_piezos'
    df_obs.index.name = 'time'
    data = xr.DataArray(df_obs, name='hpiezo')
    data.to_netcdf(new_file)

def plot_hpiezo_daily(codes, df_obs, df_sim, figure_dir):
    """
    Plot observed and simulated piezometric head evolutions

    Parameters
    ---------
    codes: list
    df_obs: pd.DataFrame
    df_sim: pd.DataFrame
    figure_dir: str
    """
    for code in codes:
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        for sim in df_sim['simulation'].to_numpy():
            df_sim['hpiezo'].sel(codes_piezos=code, simulation=sim).plot.line(
                ax=ax,
                ls='--',
                label=sim
            )
        df_obs['hpiezo'].sel(codes_piezos=code).dropna(dim="time").plot.line(
            ax=ax,
            color='b',
            label='Obs'
        )
        ax.set_title('{0}'.format(code))
        ax.set_ylabel('m')
        ax.legend()
        ax.grid()
        fig.tight_layout()
        os.makedirs(figure_dir, exist_ok=True)
        fig.savefig('{0}/{1}.png'.format(
            figure_dir, code.replace('/', '_')), dpi=300
        )
        plt.close('all')

def prepare_observations(dir_piezo, dir_res_proc):
    df_to_keep = pd.read_csv(
        '{0}/piezometres_to_keep.csv'.format(dir_piezo),
        index_col='code_bss'
    )
    obs_dir = '{0}/pz_chronique_corriges'.format(dir_piezo)
    set_netcdf_obs(
        obs_dir,
        df_to_keep.index,
        '{0}/OBS_HPIEZO_DAILY.nc'.format(dir_res_proc)
    )

def convert_hpiezo_daily(simulations, dir_res, dir_res_proc):
    for simulation in simulations:
        sim = ConvertSingleSimAquiFR(simulation, dir_res, dir_res_proc) 
        sim.convert_hpiezo_daily_to_netcdf()

def convert_h_monthly(
    simulations, shp, dir_res, dir_res_proc
):
    for simulation in simulations:
        sim = ConvertSingleSimAquiFR(simulation, dir_res, dir_res_proc) 
        sim.convert_h_monthly_to_netcdf(shp)

def convert_h_monthly_1_km(
    simulations, shp_dir, dir_res, dir_res_proc
):
    for simulation in simulations:
        sim = ConvertSingleSimAquiFR(simulation, dir_res, dir_res_proc) 
        sim.convert_h_monthly_to_1km(shp_dir)

def plot_hpiezo(simulations, dir_piezo, dir_res_proc):
    df_sim = []
    for sim in simulations:
        df_sim.append(xr.open_dataset(
            '{0}\{1}\{1}_HPIEZO_DAILY_1958_2020.nc'.format(dir_res_proc, sim)
        ))
    df_sim = xr.concat(df_sim, pd.Index(simulations, name='simulation'))
    df_obs = xr.open_dataset('{0}/OBS_HPIEZO_DAILY.nc'.format(dir_res_proc))
    codes = pd.read_csv(
        '{0}/piezometres_references.csv'.format(dir_piezo)
    )['code_bss']
    plot_hpiezo_daily(
        codes,
        df_obs,
        df_sim,
        'figures'
    )

def plot_map(da, gdf_safran):
    dfp = xr.DataArray(
        [np.nan]*len(gdf_safran),
        coords={'zone':gdf_safran['zone']}
    )
    dfp.loc[da['zone']] = da
    dfp = dfp.assign_coords(
        dict(
            x=gdf_safran.centroid.x,
            y=gdf_safran.centroid.y
        )
    )
    dfp = dfp.set_index(zone=['y', 'x']).unstack('zone')
    dfp = dfp.sortby('y').sortby('x')
    dfp.plot.pcolormesh()


