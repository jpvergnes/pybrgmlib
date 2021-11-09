import era5
import calendar
import pandas as pd
import safran

# Hourly data
# Demande des données sur le domaine longitude 39°N à 54°N et latitude -8°E à 15°O
cds = era5.ERA5(
    product='reanalysis-era5-single-levels' , area=[54, -8, 39, 15],
    )
# Define grid
cds.set_src_grid_from_grib('total_precipitation')

# Get twelve month for the 2000 year for two variables (
# (température à 2 m, précipitations totales)
# Variables horaires 
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
    # 'sum' car cumul des donées horaires sur 1 journée
    df2 = cds.get_daily_variable(df1, 'sum', variable=['tp'])
    # 'mean' car moyenne de la température sur une journée
    df3 = cds.get_daily_variable(df1, 'mean', variable=['2t'])
    df1 = pd.concat([df2, df3], axis=1)
    df.append(df1)
df = pd.concat(df, axis=0) # df: grille era5

# --- Reprojection --- #
zones_safran = safran.build_raster_safran()
cds.set_dst_grid(
    safran.X[0], safran.Y[0], safran.X[1], safran.Y[1],
    safran.NX, safran.NY,
    {'init':'EPSG:{0}'.format(safran.EPSG)}
)
df = cds.multiply_by_factor(df, 1000, variable=['tp']) # conversion m/day -> mm/day
df = cds.add_factor(df, -273.15, variable=['2t']) # conversion kelvin -> °C
dfproj = cds.reproject(df, zones_safran) # dfproj : df reprojeté sur la grille safran
dfproj = cds.drop_nodata(dfproj, zones_safran) # on supprime les nodatas hors zones safran
for variable in ['tp', '2t']:
    cds.write_year_in_csv(dfproj, 2000, variable) # écriture en format csv

# --- Save rasters --- #
dfproj2 = cds.add_nodata(dfproj, zones_safran)
for variable in ['tp', '2t']:
    cds.write_src_monthly_mean_raster(df, variable) # moyenne mensuelles df
    cds.write_dst_monthly_mean_raster(dfproj2, variable) # moyennes mensuelles dfproj2