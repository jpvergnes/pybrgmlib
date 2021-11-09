import era5
import calendar
import pandas as pd
import meteobrgm

# Hourly data
# Demande des données sur le domaine longitude 50° à 52° et latitude -1° à 1°
cds = era5.ERA5(
    product='reanalysis-era5-land', area=[54, -8, 39, 15], cum_period=24
    )
# Define grid
cds.set_src_grid_from_grib('total_precipitation')

# Get twelve month for the 2000 year for two variables (
# (température à 2 m, précipitations totales)
# Variables horaires cumulées sur 24 h (pour tp, par pour 2t)
df = []
for month in range(1, 13):
    df1 = cds.request_extended_period(
        [
            '2m_temperature', 'total_precipitation',
        ],
        year=2000,
        month=month,
        day=list(range(1, calendar.monthlen(2000, month) + 1)),
        time=['{0:02}:00'.format(h) for h in range(24)]
    )
    # Get daily variable
    # Données précipitations cumulées sur 24 h donc à décumuler (is_cum)
    # 'sum' car somme des données horaires (décumulées) sur 24 h 
    df2 = cds.get_daily_variable(df1, 'sum', is_cum=True, variable=['tp'])
    # 'mean' car moyenne de la température sur une journée
    df3 = cds.get_daily_variable(df1, 'mean', variable=['2t'])
    df1 = pd.concat([df2, df3], axis=1)
    df.append(df1)
df = pd.concat(df, axis=0)

# --- Reprojection --- #
zones_safran = meteobrgm.build_raster_safran()
cds.set_dst_grid(
    meteobrgm.X[0], meteobrgm.Y[0], meteobrgm.X[1], meteobrgm.Y[1],
    meteobrgm.NX, meteobrgm.NY,
    {'init':'EPSG:{0}'.format(meteobrgm.EPSG)}
)
df = cds.multiply_by_factor(df, 1000, variable=['tp']) # conversion m/day -> mm/day
df = cds.add_factor(df, -273.15, variable=['2t']) # conversion kelvin -> °C
dfproj = cds.reproject(df, zones_safran)
dfproj = cds.drop_nodata(dfproj, zones_safran)
for variable in ['tp', '2t']:
    cds.write_year_in_csv(dfproj, 2000, variable)

# --- Save rasters --- #
dfproj2 = cds.add_nodata(dfproj, zones_safran)
for variable in ['tp', '2t']:
    cds.write_src_monthly_mean_raster(df, variable)
    cds.write_dst_monthly_mean_raster(dfproj2, variable)