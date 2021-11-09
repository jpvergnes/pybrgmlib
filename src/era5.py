import os
import csv
import secrets
import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import rasterio
import rasterio.warp as warp
import cdsapi
import eccodes as ecs

"""
Packages requis :
    - numpy
    - pandas
    - rasterio (dépend de gdal - librairie C à installer via gestionnaire de package) :
    utile pour la reprojection vers la grille SAFRAN
    - cdsapi (les instructions se trouvent ici :https://cds.climate.copernicus.eu/api-how-to)
    - eccodes (dispo sur pypi - librairie C à installer via gestionnaire de package)

Lors de l'exécution des programmes, un dossier "cache" est créé dans le répertoire 
avec un fichier .cacherc et des fichiers grib téléchargés. Ce fichier.cacherc permet
la mise en correspondance d'une requête avec ledit fichier grib. Si on demande la même requête,
le fichier ne sera pas retéléchargé car déjà présent dans le cache. Compte tenu du nombre 
de requêtes possibles, j'ai trouvé ce système de requête - nom de fichier.grib (en hexadecimal)
utile pour ne pas avoir des noms à ralonge.
"""

class AbstractReproj(object):
    """
    Outil pour reprojeter d'une projection (src) à une autre (dst)
    """
    src_transform = ''
    src_crs = ''
    src_grid = {}
    def __init__(self, product):
        """
        Argument
            - product : nom du produit du climate data store
        """
        self.product = product

    def set_dst_grid(self, Xeast, Ysouth, Xwest, Ynorth, NX, NY, crs):
        """
        Configure la grille de destination
        Arguments :
            - Xeast : limite est
            - Ysouth : limite sud
            - Xwest : limite ouest
            - Ynorth : limite nord
            - NX : nombre de pixels en X
            - NY : nombre de pixels en Y
            - crs : système de projection de coordonnées, en format
            dictionnaire de rasterio (exemple : {'init':'EPSG:4326'})
        """
        self.dst_transform = rasterio.transform.from_bounds(
            Xeast, Ysouth, Xwest, Ynorth, NX, NY
            )
        self.dst_grid = {}
        self.dst_grid['Xeast'] = Xeast
        self.dst_grid['Ysouth'] = Ysouth
        self.dst_grid['Xwest'] = Xwest
        self.dst_grid['Ynorth'] = Ynorth
        self.dst_grid['NX'] = NX
        self.dst_grid['NY'] = NY
        self.dst_crs = crs

    def reproject_to_dst(self, src, dst, src_nodata=9999., dst_nodata=9999.):
        """
        Reprojection de src à dst
        Arguments:
            - src : 2D or 3D np.array ou rasterio band
            - dst : 2D or 3D np.array ou rasterio band
            - src_nodata : valeurs nodata dans src
            - dst_nodata : valeurs nodata dans dst
        """
        new_dst = dst.copy()
        warp.reproject(
            src,
            new_dst,
            src_transform=self.src_transform,
            src_crs=self.src_crs,
            dst_transform=self.dst_transform,
            dst_crs=self.dst_crs,
            resampling=warp.Resampling.bilinear,
            src_nodata=src_nodata,
            dst_nodata=dst_nodata,
        )
        new_dst = np.where(dst == dst_nodata, dst_nodata, new_dst)
        return new_dst

    def drop_nodata(self, df, zones):
        """
        Supprime les zones à 9999.
        Arguments :
            - df : pd.DataFrame avec l'ensemble des zones en colonnes
            - zones : zones à garder/supprimer
        """
        indices, = np.where(zones.flatten() != 9999.)
        indices = indices + 1
        df = df.loc[:, (slice(None), indices)]
        df = df.sort_index(axis=1)
        variables = df.columns.get_level_values('variable').unique()
        columns = pd.MultiIndex.from_product(
            [variables, list(range(1, len(indices) + 1))]
            )
        df.columns = columns
        return df

    def write_year_in_csv(self, df, year, variable, hydro_year=False, **kwargs):
        """
        Ecriture en format csv
        Arguments
            - df: dataframe à écrire
            - year: année à écrire
            - variable : variable à écrire
            - hydro_year : si année hydrologique ou pas
            - **kwargs : dictionnaires d'arguments fournis à df.to_csv
        """
        df = df.round(2)
        if hydro_year:
            start = '{0}-8-1'.format(year)
            end = '{0}-7-31'.format(year + 1)
            filename = 'csv_files/{0}_{1}_{2}_{3}.csv'.format(
                self.product, variable, year, year+1
                )
        else:
            start = '{0}-1-1'.format(year)
            end = '{0}-12-31'.format(year)
            filename = 'csv_files/{0}_{1}_{2}.csv'.format(
                self.product, variable, year
                )
        if not os.path.isdir('csv_files'):
            os.makedirs('csv_files')
        df = df.loc[start:end, variable]
        df.index.name = len(df.columns) + 1
        df = df.reset_index().sort_index(axis=1)
        df.to_csv(
            filename, index=None, sep=' ', float_format='%.2f', **kwargs
            )

    def read_year_in_csv(self, year, variable):
        """
        Lecture d'un fichier csv
        Arguments :
            - year: année à lire
            - variable : variable à lire
        """
        filename = 'csv_files/{0}_{1}_{2}.csv'.format(
            self.product, variable, year
            )
        df = pd.read_csv(
            filename, delim_whitespace=True, index_col=-1, parse_dates=True,
            quoting=csv.QUOTE_NONE
        )
        df.columns = list(map(int, df.columns))
        return df

    def write_monthly_mean_raster(self, df, variable, grid, src_crs, src_transform, suffix):
        """
        Ecriture d'une moyenne mensuelle en raster
        Arguments :
            - df : pd.DataFrame des variables
            - variable : variable à écrire
            - grid : grille correspondante
            - src_crs : système de projection de la grille
            - src_transform : transformation de la grille
            - suffix : suffixe au nom de variable
        """
        if not os.path.isdir('rasters'):
            os.makedirs('rasters')
        df = df.loc[:, variable]
        mask = df == 9999.
        cols = mask.all()[mask.all()].index
        dfmonth = df.resample('M').mean()
        dfmonth.loc[:, cols] = 9999.
        assert dfmonth.iloc[0, :].shape[0] == grid['NY']*grid['NX']
        for _, row in dfmonth.iterrows():
            src = row.values.reshape(grid['NY'], grid['NX'])
            save_raster_as_tiff(
                'rasters/{0}_{1}_{2}_{3}_{4}.tiff'.format(
                    self.product, suffix, variable,
                    row.name.year, row.name.month
                    ),
                src, src_crs, src_transform
                )
    
    def write_dst_monthly_mean_raster(self, df, variable):
        """
        Ecriture d'une moyenne mensuelle de dst
        Arguments :
            - df : pd.DataFrame des variables
            - variable : variable à écrire
        """
        self.write_monthly_mean_raster(
            df, variable, self.dst_grid, self.dst_crs, self.dst_transform, 'dst'
            )

    def write_src_monthly_mean_raster(self, df, variable):
        """
        Ecriture d'une moyenne mensuelle de src
        Arguments :
            - df : pd.DataFrame des variables
            - variable : variable à écrire
        """
        self.write_monthly_mean_raster(
            df, variable, self.src_grid, self.src_crs, self.src_transform, 'src'
            )

class EOBS(AbstractReproj):
    def set_src_grid_from_netcdf(self, filename, nc_lon='longitude', nc_lat='latitude'):
        rootgrp = Dataset(filename)
        lats = rootgrp[nc_lat]
        lons = rootgrp[nc_lon]
        self.src_grid = {}
        self.src_grid['Xwest'] = np.round(lons[0], 2)
        self.src_grid['Xeast'] = np.round(lons[-1], 2)
        self.src_grid['Ysouth'] = np.round(lats[0], 2)
        self.src_grid['Ynorth'] = np.round(lats[-1], 2)
        self.src_grid['NX'] = len(lons)
        self.src_grid['NY'] = len(lats)
        self.src_grid['Xres'] = np.round((lons[-1]-lons[0])/len(lons), 2)
        self.src_grid['Yres'] = np.round((lats[-1]-lats[0])/len(lats), 2)
        self.src_transform = rasterio.transform.from_bounds(
            self.src_grid['Xwest'] - self.src_grid['Xres'] / 2.,
            self.src_grid['Ysouth'] - self.src_grid['Yres'] / 2.,
            self.src_grid['Xeast'] + self.src_grid['Xres'] / 2.,
            self.src_grid['Ynorth'] + self.src_grid['Yres'] / 2.,
            self.src_grid['NX'], self.src_grid['NY']
        )
        self.src_crs = {'init':'EPSG:4326'}


class ERA5(AbstractReproj):
    def __init__(self, *, product, area, cum_period=1):
        """
        Arguments
            - product : nom du produit à demander à cdsapi
            - area : domaine demandé
            - cum_period : période sur laquelle les variables sont cumulés (en heure)
        """
        AbstractReproj.__init__(self, product)
        self.c = cdsapi.Client()
        self.area = area
        self.cum_period = cum_period
        if not os.path.isdir('cache'):
            os.makedirs('cache')
            with open('cache/.cacherc', 'w') as _:
                pass

    def _build_name(self, **kwargs):
        components = ['{0}'.format(self.product)]
        for key in sorted(kwargs):
            if type(kwargs[key]) is list:
                if 'area' not in key:
                    kwargs[key].sort()
                kwargs[key] = '_'.join(list(map(str, kwargs[key])))
            components.append(
                '{0}_{1}'.format(key, kwargs[key])
            )
        return '_'.join(components)

    def _build_cache(self):
        dcache = {}
        with open('cache/.cacherc', 'r') as f:
            for line in f:
                key, val = line.split()
                dcache[key] = val
        return dcache

    def retrieve(self, filename, **kwargs):
        self.c.retrieve(
            self.product,
            {
                'format':'grib',
                'product_type':'reanalysis',
                **kwargs
            },
            filename)

    def download_grib(self, name, force=False, **kwargs):
        dcache = self._build_cache()
        if name not in dcache:
            filename = 'cache/{0}.grib'.format(secrets.token_hex(16))
            self.retrieve(filename, **kwargs)
            with open('cache/.cacherc', 'a+') as f:
                f.write('{0} {1}\n'.format(name, filename))
        elif name in dcache and force:
            filename = dcache['name']
            self.retrieve(filename, **kwargs)

    def grid_from_grib(self, name):
        grid = {}
        dcache = self._build_cache()
        with open(dcache[name], 'rb') as f:
            gid = ecs.codes_grib_new_from_file(f)
            grid['NX'] = ecs.codes_get(gid, 'numberOfColumns')
            grid['NY'] = ecs.codes_get(gid, 'numberOfRows')
            grid['Xres'] = ecs.codes_get(gid, 'iDirectionIncrementInDegrees')
            grid['Yres'] = ecs.codes_get(gid, 'jDirectionIncrementInDegrees')
            grid['Ynorth'] = ecs.codes_get(gid, 'latitudeOfFirstGridPointInDegrees')
            grid['Xwest'] = ecs.codes_get(gid, 'longitudeOfFirstGridPointInDegrees')
            grid['Ysouth'] = ecs.codes_get(gid, 'latitudeOfLastGridPointInDegrees')
            grid['Xeast'] = ecs.codes_get(gid, 'longitudeOfLastGridPointInDegrees')
            values = ecs.codes_get_values(gid)
        return grid, values

    def set_src_grid_from_grib(self, variable):
        """
        Définit les informations géographique à partir d'un fichier grib
        Argument :
            - variable : variable unique pour récupération du fichier grib
        """
        kwargs = dict(year=2000, month=1, day=1, time='01:00',
            variable=variable, area=self.area)
        name = self._build_name(**kwargs)
        self.download_grib(name, **kwargs)
        self.src_grid, _ = self.grid_from_grib(name)
        self.src_transform = rasterio.transform.from_bounds(
            self.src_grid['Xwest'] - self.src_grid['Xres'] / 2.,
            self.src_grid['Ysouth'] - self.src_grid['Yres'] / 2.,
            self.src_grid['Xeast'] + self.src_grid['Xres'] / 2.,
            self.src_grid['Ynorth'] + self.src_grid['Yres'] / 2.,
            self.src_grid['NX'], self.src_grid['NY']
        )
        self.src_crs = '+proj=latlong'
    
    def get_values_from_grib(self, name):
        def build_dataframe(values, dates):
            variable_names = values.keys()
            df = []
            for key in variable_names:
                df.append(
                    pd.DataFrame(values[key], index=dates[key])
                )
            df = pd.concat(df, axis=1)
            columns = pd.MultiIndex.from_product(
                [
                    variable_names,
                    np.arange(
                        self.src_grid['NX']*self.src_grid['NY']
                    ).astype('int')
                ]
            )
            columns = columns.set_names(['variable', 'zone'])
            df.columns = columns
            return df
        dcache = self._build_cache()
        with open(dcache[name], 'rb') as f:
            values = defaultdict()
            dates = defaultdict()
            nb = ecs.codes_count_in_file(f)
            for _ in range(nb):
                gid = ecs.codes_grib_new_from_file(f)
                variable_name = ecs.codes_get(gid, 'shortName')
                if variable_name not in values:
                    values[variable_name] = []
                values[variable_name].append(
                    ecs.codes_get_values(gid)
                    )
                if variable_name not in dates:
                    dates[variable_name] = []
                date = ecs.codes_get_string(gid, 'validityDate')
                tim = '{0:04}'.format(ecs.codes_get(gid, 'validityTime'))
                dates[variable_name].append(datetime.datetime(
                    int(date[:4]), int(date[4:6]), int(date[6:]),
                    int(tim[:2]), int(tim[2:])
                    ))
        assert np.unique([len(dates[key]) for key in dates]).shape[0] == 1
        assert np.unique([dates[key][-1] for key in dates]).shape[0] == 1
        return build_dataframe(values, dates)

    def request_values_from_api(self, **kwargs):
        name = self._build_name(**kwargs)
        self.download_grib(name, **kwargs)
        return self.get_values_from_grib(name)

    def request_period(self, variable, **kwargs):
        """
        Requête à cds pour une période spécifiée dans kwargs
        (voir le site cdsapi pour la syntaxe des arguments)
        Arguments:
            - variable : variable à récupérer
            - kwargs : dictionnaire d'arguments à passer à cdsapi
        """
        return self.request_values_from_api(
                    variable=variable, area=self.area, **kwargs
                    )

    def request_extended_period(self, variable, **kwargs):
        """
        Requête à cds pour une période spécifiée dans kwargs
        (voir le site cdsapi pour la syntaxe des arguments)
        Ajoute 1 jour en plus (valeur à 00:00 du jour suivant)
        Arguments:
            - variable : variable à récupérer
            - kwargs : dictionnaire d'arguments à passer à cdsapi
        """
        df = self.request_values_from_api(
                    variable=variable, area=self.area, **kwargs
                    )
        next_day = df.index[-1] + datetime.timedelta(hours=1)
        if next_day.hour == 0:
            df2 = self.request_values_from_api(
                variable=variable, area=self.area,
                year=next_day.year, month=next_day.month,
                day=next_day.day, time='00:00',
            )
            df = pd.concat([df, df2], axis=0)
        return df

    def decum_variable(self, df):
        decum = df.where(df == 9999., df - df.shift(1))
        ic = int(self.cum_period)
        decum.iloc[1::ic, :] = df.iloc[1::ic, :]
        decum.iloc[0, :] = 0
        return decum

    def get_hourly_accumulated_variable(self, df):
        """
        Décumule des variables cumulées
        """
        if self.cum_period > 1:
            assert df.index[0].hour == 0 # Période commmence en début de journée
            df = self.decum_variable(df)
        return df

    def get_daily_variable(self, df2, func, is_cum=False, **kwargs):
        """
        Retourn des variables journalières
        Arguments :
            - df2 : pd.DataFrame de variables
            - func : 'sum' ou 'mean'
            - is_cum : si la variable est cumulé sur plusieurs heures ou non.
            Fait appel à decum_variable dans ce cas
        TO DO : pour des données cumulées sur 24 h ,ne sert à rien de
        décumuler puis de recumuler à nouveau
        """
        if 'variable' in kwargs:
            assert type(kwargs['variable']) is list 
        variable = kwargs.get('variable', slice(None))
        df = df2.loc[:, variable]
        assert df.index[0].hour == 0 # Période commmence en début de journée
        mask = df == 9999.
        cols = mask.all()[mask.all()].index
        if is_cum and self.cum_period > 1:
            df = self.decum_variable(df)
        df.index = df.index - datetime.timedelta(hours=1) # Décalage d'1 heure
        if func == 'sum':
            df = df.resample('D').sum() # Somme après décalage
        elif func == 'mean':
            df = df.resample('D').mean() # Moyenne après décalage
        df.loc[:, cols] = 9999.
        return df.iloc[1:, :] # 1er jour dû au décalage d'1 heure

    def reproject(self, df, dst):
        dfout = []
        variables = df.columns.get_level_values('variable').unique()
        for variable in variables:
            df2 = df.xs(key=variable, axis=1)
            dfout2 = []
            for _, row in df2.iterrows():
                row = row.values.reshape(
                    (self.src_grid['NY'], self.src_grid['NX'])
                    )
                dst = self.reproject_to_dst(row, dst)
                dfout2.append(dst.flatten())
            dfout.append(pd.DataFrame(dfout2, index=df.index))
        dfout = pd.concat(dfout, axis=1)
        columns = pd.MultiIndex.from_product(
            [
                variables,
                np.arange(
                    self.dst_grid['NX']*self.dst_grid['NY']
                ).astype('int')
            ]
        )
        columns = columns.set_names(['variable', 'zone'])
        dfout.columns = columns
        return dfout

    def add_nodata(self, df, zones):
        """
        Les zones masquées de SAFRAN sont ajoutées à df
        """
        variables = df.columns.get_level_values('variable').unique()
        columns = pd.MultiIndex.from_product(
            [variables, list(range(len(zones.flatten())))]
            )
        u, w = np.where(zones != 9999.)
        dfout = []
        z = zones.copy()
        for variable in variables:
            df2 = []
            for _, row in df.iterrows():
                z[u, w] = row[variable].squeeze().values
                df2.append(z.flatten())
            df2 = pd.DataFrame(df2)
            dfout.append(df2)
        dfout = pd.concat(dfout, axis=1)
        dfout.columns = columns
        dfout.index = df.index
        return dfout

    def multiply_by_factor(self, df, factor, **kwargs):
        """
        Multiplie les variables par un facteur
        """
        if 'variable' in kwargs:
            assert type(kwargs['variable']) is list 
        variable = kwargs.get('variable', slice(None))
        df = df.sort_index(axis=1)
        dfv = df.loc[:, variable]
        dfv = dfv.where(dfv == 9999., dfv * factor)
        df.loc[:, variable] = dfv.values
        return df

    def add_factor(self, df, factor, **kwargs):
        """
        Ajoute un facteur à des variables
        """
        variable = kwargs.get('variable', slice(None))
        if 'variable' in kwargs:
            assert type(kwargs['variable']) is list 
        dfv = df.loc[:, variable]
        dfv = dfv.where(dfv == 9999., dfv + factor)
        df.loc[:, variable] = dfv.values
        return df


def save_raster_as_tiff(filename, dst, crs, transform):
    with rasterio.open(
        filename, 'w', driver='GTiff',
        width=dst.shape[1], height=dst.shape[0],
        count=1, dtype=dst.dtype,
        crs=crs, transform=transform,
        nodata=9999.
    ) as new_dataset:
        new_dataset.write(dst, 1)