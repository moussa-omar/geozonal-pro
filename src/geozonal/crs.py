from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import geopandas as gpd
from pyproj import CRS, Geod
from shapely.geometry.base import BaseGeometry


@dataclass(frozen=True)
class CRSInfo:
    crs: CRS
    is_geographic: bool


def ensure_same_crs(gdf: gpd.GeoDataFrame, target_crs: CRS) -> Tuple[gpd.GeoDataFrame, bool]:
    """
    Reprojects GeoDataFrame to target CRS if needed.
    Returns (gdf_out, did_reproject).
    """
    if gdf.crs is None:
        raise ValueError("Input GeoDataFrame has no CRS. Please set gdf.crs before calling.")

    src = CRS.from_user_input(gdf.crs)
    tgt = CRS.from_user_input(target_crs)

    if src == tgt:
        return gdf, False

    return gdf.to_crs(tgt), True


def geometry_area(geom: BaseGeometry, crs: CRS) -> float:
    """
    Returns geometry area.
    - If CRS is projected: uses planar area (geom.area).
    - If CRS is geographic: uses geodesic area via pyproj.Geod.
    """
    crs_obj = CRS.from_user_input(crs)

    if not crs_obj.is_geographic:
        return float(geom.area)

    # Geographic: compute geodesic area (meters^2) using WGS84 ellipsoid
    geod = Geod(ellps="WGS84")

    # pyproj.Geod.geometry_area_perimeter returns signed area
    area, _perim = geod.geometry_area_perimeter(geom)
    return abs(float(area))
