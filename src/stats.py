import numpy as np
import pandas as pd
import datetime
from KDEpy import FFTKDE
from scipy.stats import norm
from scipy.interpolate import interp1d

def nash(obs, sim, prec=2):
    return np.round(1 - (np.sum((obs - sim)**2)) / 
        (np.sum((obs - np.mean(obs))**2)) , prec)

def rmse(obs, sim, prec=2):
    return np.round(np.sqrt(np.mean((sim - obs)**2)) , prec)

def ratio(obs, sim, prec=2):
    return np.round(np.mean(sim) / np.mean(obs), prec)

def bias(obs, sim, prec=2):
    return np.round(np.mean(sim) - np.mean(obs), prec)

class Statistiques(object):
    def __init__(self, obs, sim, prec=2):
        self.prec = prec
        self.moy_obs = self.set_moy(obs)
        self.moy_sim = self.set_moy(sim)
        self.std_obs = np.ma.std(obs)
        self.std_sim = np.ma.std(sim)
        self.eff = self.set_eff(obs, sim)
        self.rms = self.set_rms(obs, sim)
        self.cor = self.set_cor(obs, sim)

    def set_moy(self, data):
        data = np.ma.average(data, axis=0)
        return data

    def set_eff(self, obs, sim):
        data = 1 - (np.ma.sum((sim-obs)**2,
            axis=0))/(np.ma.sum((obs-self.moy_obs)**2, axis=0))
        return np.round(data, self.prec)

    def set_rms(self, obs, sim):
        data = np.ma.sqrt((np.ma.sum((sim-obs)**2, axis=0))/
            obs.shape[0])
        return np.round(data, self.prec)

    def set_cor(self, obs, sim):
        data = np.ma.sum((obs-self.moy_obs)*(sim-self.moy_sim), axis=0)/(
                np.ma.sqrt(np.ma.sum((obs-self.moy_obs)**2, axis=0))*
                np.ma.sqrt(np.ma.sum((sim-self.moy_sim)**2, axis=0)))
        return np.round(data, self.prec)

    def set_ratio(self):
        data = self.moy_sim/self.moy_obs
        return np.round(data, self.prec)
    
    def set_biais(self):
        data = self.moy_sim - self.moy_obs
        return np.round(data, self.prec)

def compute_ips(dfobs, ref_period=None, **kwargs):
    """
    Compute IPS from a time series

    Parameters
    ----------
    dfobs : pandas.DataFrame
    ref_period : list (optional)
        list of two years as int

    Return
    ------
    pandas.DataFrame
        Computed IPS over the whole serie
    """
    if type(dfobs) is pd.core.series.Series:
        dfobs = pd.DataFrame(dfobs)
    dfobs = dfobs.resample('M').mean()
    save_columns = dfobs.columns
    dfobs.columns = list(range(dfobs.shape[1]))
    nbs_obs = dfobs.columns
    dfobs.loc[:, 'year'] = dfobs.index.year
    dfobs.loc[:, 'month'] = dfobs.index.month
    dfspli = []
    for nb_obs in nbs_obs: 
        df = dfobs.pivot(index='year', columns='month', values=nb_obs)
        dfref = df
        if type(ref_period) is list:
            dfref = dfobs.loc[ref_period[0]:ref_period[1],:].pivot(
                index='year', columns='month', values=nb_obs
            )
        for month in dfref.columns:
            data_ref = dfref.loc[:, month].dropna()
            min1 = kwargs.get('min', data_ref.min())
            max1 = kwargs.get('max', data_ref.max())
            if max1 - min1 >= 1:
                min1 = round(min1, 1) - 0.5
                max1 = round(max1, 1) + 0.5
            else:
                min1 -= 1
                max1 += 1
            data_ref = data_ref.values
            data_ref = np.maximum(data_ref, min1)
            data_ref = np.minimum(data_ref, max1)
            kde = FFTKDE(kernel='epa', bw='silverman')
            x, y = kde.fit(np.unique(data_ref.round(2))).evaluate()
            y = np.cumsum(y) / np.sum(y)

            f = interp1d(x, y, bounds_error=False, fill_value='extrapolate')
            data = df.loc[:, month].dropna()
            data_f = np.maximum(f(data), 0.001)
            data_f = np.minimum(f(data), 0.999)
            spli = norm.ppf(data_f)
            df.loc[:, month][~df.loc[:, month].isna()] = spli

        df = df.stack(dropna=False)
        index = np.array(df.index.values.tolist())
        index = [datetime.datetime(d[0], d[1], 1) for d in index]
        df.index = index
        dfspli.append(df.dropna())
    dfspli = pd.concat(dfspli, axis=1)
    dfspli.columns = save_columns
    return dfspli