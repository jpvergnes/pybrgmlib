import numpy as np
import pandas as pd

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