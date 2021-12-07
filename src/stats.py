import numpy as np
from numpy.lib.npyio import save
import pandas as pd
import datetime
from KDEpy import FFTKDE
from scipy.stats import norm
from scipy.interpolate import interp1d
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing

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

def compute_ips(dfobs, ref_period=None, njobs=1, **kwargs):
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
    dfobs = dfobs.dropna(axis=0, how='all')
    save_columns = dfobs.columns
    dfobs.columns = list(range(dfobs.shape[1]))
    dfobs.loc[:, 'year'] = dfobs.index.year
    dfobs.loc[:, 'month'] = dfobs.index.month
    dfref = dfobs
    if type(ref_period) is list:
        dfref = dfobs.loc[str(ref_period[0]):str(ref_period[1]),:].pivot(
            index='year', columns='month'
        )
    df = dfobs.pivot(index='year', columns='month')
    def single_ips(name, item_ref):
        data_ref = item_ref.dropna()
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
        data = df.loc[:, name]
        data_f = np.maximum(f(data), 0.001)
        data_f = np.minimum(data_f, 0.999)
        spli = norm.ppf(data_f)
        if np.isnan(spli).any():
            import ipdb; ipdb.set_trace()
        return spli
    inputs = tqdm(list(dfref.iteritems()))
    splis = Parallel(n_jobs=njobs)(
        delayed(single_ips)(name, item) for name, item in inputs
    )
    df.iloc[:, :] = np.array(splis).T
    df = df.stack(dropna=False)
    df.index = [
        datetime.datetime(d[0], d[1], 1) for d in df.index.values.tolist()
    ]
    df.columns = save_columns
    return df