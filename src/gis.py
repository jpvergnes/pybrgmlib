"""
Fonctions GIS
"""
import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString
from fiona.crs import from_epsg
import gdal
import osr, ogr
import numpy as np

def get_coord_transform(inputEPSG, outputEPSG):
    """
    Return an osr.CoordinateTransformation object from an EPSG to another
    EPSG

    Parameters
    ----------
    inputEPSG : str
    outputEPSG : str

    Return
    ------
    osr.CoordinateTransformation object instance
    """
    inSpatialRef = osr.SpatialReference()
    inSpatialRef.ImportFromEPSG(inputEPSG)

    outSpatialRef = osr.SpatialReference()
    outSpatialRef.ImportFromEPSG(outputEPSG)

    return osr.CoordinateTransformation(inSpatialRef, outSpatialRef)

def reproj_point(x, y, inputEPSG, outputEPSG):
    """
    Reproject a point with (x, y) coordinates from inputEPSG 
    to outputEPSG

    Parameters
    ----------
    x : int
    y : int
    inputEPSG : str
    outputEPSG : str

    Return
    ------
    ogr.Geometry(ogr.wkbPoint)
    """
    point = ogr.Geometry(ogr.wkbPoint)
    point.AddPoint(x, y)
    coordTransform = get_coord_transform(inputEPSG, outputEPSG)
    point.Transform(coordTransform)
    return point.GetX(), point.GetY()

def reproj_polygon(x, y, inputEPSG, outputEPSG):
    """
    Reproject a polygon with (x, y) coordinates from inputEPSG 
    to outputEPSG

    Parameters
    ----------
    x : list of int
    y : list of int
    inputEPSG : str
    outputEPSG : str

    Return
    ------
    ogr.Geometry(ogr.wkbLinearRing)
    """
    ring = ogr.Geometry(ogr.wkbLinearRing)
    coordTransform = get_coord_transform(inputEPSG, outputEPSG)
    for xp, yp in zip(x, y):
        xp, yp = reproj_point(xp, yp, coordTransform)
        ring.AddPoint(xp, yp)
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    return poly

def bln2gdf(bln, epsg, unit=1):
    """
    Read a bln file with epsg projection

    Parameters
    ----------
    bln : str
        bln file to read
    epsg : str
        code epsg de la projection
    unit : int or float
        correction factor for converting the units of the bln file in meters
        example : 1000 if bln in km

    Return
    ------
    geopandas.DataFrame
    """
    gdf = gpd.GeoDataFrame()
    gdf['geometry'] = None
    with open(bln, encoding='ISO-8859-1') as fbln:
        line = fbln.readline()
        id_ent = 0
        while line.strip():
            fields = line.strip().split()
            nb_pts = int(fields[0])
            entitie = []
            for _ in range(nb_pts):
                line = fbln.readline()
                entitie.append(tuple([float(value)*unit
                    for value in line.strip().split()]))
            if len(entitie) == 1:
                entitie = Point(entitie)
            elif len(entitie) > 1 and entitie[0] != entitie[-1]:
                entitie = LineString(entitie)
            elif len(entitie) > 1 and entitie[0] == entitie[-1]:
                entitie = Polygon(entitie)
            gdf.loc[id_ent, 'geometry'] = entitie
            if len(fields) > 2:
                gdf.loc[id_ent, 'Etiquette'] = fields[2]
            line = fbln.readline()
            id_ent += 1
    gdf.crs = from_epsg(epsg)
    return gdf

def bln2shp(bln, shp, epsg, unit=1):
    """
    Read a bln file with epsg projection

    Parameters
    ----------
    bln : str
        bln file to read
    shp : str
        shp file to write
    epsg : str
        code epsg de la projection
    unit : int or float
        correction factor for converting the units of the bln file in meters
        example : 1000 if bln in km
    """
    gdf = bln2gdf(bln, epsg, unit=unit)
    gdf.to_file(shp)