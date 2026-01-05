from __future__ import annotations

from typing import Literal

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import CRS

from .crs import ensure_same_crs


def sample_raster(
    points: gpd.GeoDataFrame,
    raster_path: str,
    band: int = 1,
    method: Literal["nearest", "bilinear"] = "nearest",
    out_col: str = "value",
) -> gpd.GeoDataFrame:
    """
    Samples raster values at point locations and returns a copy of points with a new column.

    method:
    - 'nearest': nearest pixel using rasterio.sample (default)
    - 'bilinear': bilinear interpolation using 2x2 neighborhood around the point,
                  defined w.r.t. pixel centers (so center-of-4-pixels works as expected).
                  If any neighbor is nodata or point is out of bounds -> NaN.
    """
    if points.empty:
        return points.copy()

    if not points.geometry.geom_type.isin(["Point"]).all():
        raise ValueError("sample_raster expects all geometries to be Point.")

    if method not in ("nearest", "bilinear"):
        raise ValueError("method must be 'nearest' or 'bilinear'.")

    pts = points.copy()

    with rasterio.open(raster_path) as ds:
        pts, _ = ensure_same_crs(pts, CRS.from_user_input(ds.crs))
        nodata = ds.nodata

        def _is_nodata(v: float) -> bool:
            if nodata is None:
                return False
            return bool(np.isclose(v, float(nodata)))

        coords = [(geom.x, geom.y) for geom in pts.geometry]

        # -----------------------------
        # nearest (unchanged behavior)
        # -----------------------------
        if method == "nearest":
            samples = list(ds.sample(coords, indexes=band))
            vals = np.array([s[0] if len(s) else np.nan for s in samples], dtype=float)

            if nodata is not None and np.isfinite(float(nodata)):
                vals = np.where(np.isclose(vals, float(nodata)), np.nan, vals)

            pts[out_col] = vals
            return pts

        # -----------------------------
        # bilinear w.r.t. pixel centers
        # -----------------------------
        inv = ~ds.transform
        out_vals: list[float] = []

        for (x, y) in coords:
            # world -> pixel coords (col,row) in float
            col, row = inv * (float(x), float(y))

            # convert to coordinates relative to pixel centers
            # so that integer pixel indices correspond to pixel centers
            col_c = col - 0.5
            row_c = row - 0.5

            c0 = int(np.floor(col_c))
            r0 = int(np.floor(row_c))
            c1 = c0 + 1
            r1 = r0 + 1

            # bounds check: need a 2x2 neighborhood
            if r0 < 0 or c0 < 0 or r1 >= ds.height or c1 >= ds.width:
                out_vals.append(np.nan)
                continue

            # read 2x2 window
            w = ds.read(band, window=((r0, r1 + 1), (c0, c1 + 1))).astype(float)
            q11 = float(w[0, 0])
            q21 = float(w[0, 1])
            q12 = float(w[1, 0])
            q22 = float(w[1, 1])

            # nodata => NaN
            if any(_is_nodata(q) for q in (q11, q21, q12, q22)):
                out_vals.append(np.nan)
                continue

            dx = float(col_c - c0)
            dy = float(row_c - r0)

            v = (
                q11 * (1 - dx) * (1 - dy)
                + q21 * dx * (1 - dy)
                + q12 * (1 - dx) * dy
                + q22 * dx * dy
            )
            out_vals.append(float(v))

        pts[out_col] = np.array(out_vals, dtype=float)
        return pts
