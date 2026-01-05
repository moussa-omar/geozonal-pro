from __future__ import annotations

from typing import Sequence, Dict, Any, Literal, Tuple

import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
from pyproj import CRS
from rasterio.features import geometry_window, geometry_mask
from rasterio.mask import raster_geometry_mask

from .crs import ensure_same_crs, geometry_area
from .raster import valid_mask


StatName = Literal[
    "count",
    "min",
    "max",
    "mean",
    "median",
    "std",
    "p10",
    "p90",
    "nodata_ratio",
    "coverage_ratio",
    # NEW robust stats:
    "p25",
    "p75",
    "iqr",
    "robust_mean",
    # NEW edge quality metric:
    "edge_coverage_ratio",
]


def _pixel_size_from_transform(transform) -> float:
    # rasterio Affine: a = pixel width, e = -pixel height
    px = float(abs(transform.a))
    py = float(abs(transform.e))
    return float((px + py) / 2.0)


def _edge_ring_geom(geom, pixel_size: float):
    """
    Define edge region as a 1-pixel-thick ring around polygon boundary.
    ring = buffer(+px) - buffer(-px)

    If inner buffer becomes empty (very small polygon), we fallback to outer buffer.
    """
    if geom is None or geom.is_empty or pixel_size <= 0:
        return None

    outer = geom.buffer(pixel_size)
    inner = geom.buffer(-pixel_size)

    if outer.is_empty:
        return None
    if inner.is_empty:
        return outer

    ring = outer.difference(inner)
    return ring if (ring is not None and not ring.is_empty) else None


def _edge_counts(
    ds: rasterio.io.DatasetReader,
    geom,
    band: int,
    all_touched: bool,
) -> Tuple[int, int]:
    """
    Returns (edge_total_pixels, edge_valid_pixels) computed on the edge ring geometry.
    Pixels are counted by geometry selection; validity is defined by nodata masking.
    """
    pixel_size = _pixel_size_from_transform(ds.transform)
    ring = _edge_ring_geom(geom, pixel_size)
    if ring is None:
        return 0, 0

    try:
        shape_mask, _out_transform, window = raster_geometry_mask(
            ds, [ring], crop=True, all_touched=all_touched, invert=False
        )
    except ValueError:
        # no overlap with raster
        return 0, 0

    arr = ds.read(band, window=window).astype(float)

    inside = ~shape_mask
    total = int(np.count_nonzero(inside))
    if total == 0:
        return 0, 0

    inside_vals = arr[inside]
    m = valid_mask(inside_vals, ds.nodata)
    valid = int(np.count_nonzero(m))
    return total, valid


def _compute_stats(
    values: np.ndarray,
    total_count: int,
    valid_count: int,
    geom_area: float,
    pixel_area: float,
    stats: Sequence[StatName],
    edge_total: int | None = None,
    edge_valid: int | None = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    if "count" in stats:
        out["count"] = int(valid_count)

    # Handle empty / no valid pixels
    if valid_count == 0:
        for s in stats:
            if s in ("count", "nodata_ratio", "coverage_ratio", "edge_coverage_ratio"):
                continue
            out[s] = np.nan
    else:
        if "min" in stats:
            out["min"] = float(np.min(values))
        if "max" in stats:
            out["max"] = float(np.max(values))
        if "mean" in stats:
            out["mean"] = float(np.mean(values))
        if "median" in stats:
            out["median"] = float(np.median(values))
        if "std" in stats:
            out["std"] = float(np.std(values, ddof=0))

        # Percentiles (compute only those requested)
        need_ps = []
        for p in (10, 25, 75, 90):
            if f"p{p}" in stats:
                need_ps.append(p)

        perc_map: Dict[int, float] = {}
        if need_ps:
            percs = np.percentile(values, need_ps)
            for p, v in zip(need_ps, percs):
                perc_map[p] = float(v)
                out[f"p{p}"] = float(v)

        # IQR = p75 - p25
        if "iqr" in stats:
            p25 = perc_map.get(25, float(np.percentile(values, 25)))
            p75 = perc_map.get(75, float(np.percentile(values, 75)))
            out["iqr"] = float(p75 - p25)

        # robust_mean: trimmed mean between p10 and p90
        if "robust_mean" in stats:
            lo = perc_map.get(10, float(np.percentile(values, 10)))
            hi = perc_map.get(90, float(np.percentile(values, 90)))
            trimmed = values[(values >= lo) & (values <= hi)]
            out["robust_mean"] = float(np.mean(trimmed)) if trimmed.size else float(np.mean(values))

    # nodata_ratio uses total_count (pixels inside geometry) vs valid_count
    if "nodata_ratio" in stats:
        out["nodata_ratio"] = float(1.0 - (valid_count / total_count)) if total_count > 0 else np.nan

    # coverage_ratio uses geometry area vs valid pixels area (clipped to [0,1])
    if "coverage_ratio" in stats:
        if geom_area > 0 and valid_count > 0:
            valid_area = valid_count * pixel_area
            out["coverage_ratio"] = float(min(1.0, max(0.0, valid_area / geom_area)))
        else:
            out["coverage_ratio"] = np.nan

    # edge_coverage_ratio uses edge_total vs edge_valid
    if "edge_coverage_ratio" in stats:
        if edge_total is None or edge_valid is None or edge_total == 0:
            out["edge_coverage_ratio"] = np.nan
        else:
            out["edge_coverage_ratio"] = float(edge_valid / edge_total)

    return out


def zonal_stats(
    polygons: gpd.GeoDataFrame,
    raster_path: str,
    stats: Sequence[StatName] = (
        "mean",
        "min",
        "max",
        "std",
        "p10",
        "p90",
        "nodata_ratio",
        "coverage_ratio",
    ),
    band: int = 1,
    all_touched: bool = False,
    engine: Literal["mask", "window"] = "mask",
    keep_geometry: bool = True,
) -> gpd.GeoDataFrame:
    """
    Compute nodata-aware zonal statistics of a raster over polygon geometries.

    engine:
      - "mask": uses raster_geometry_mask (geometry-only mask) + read window (robust, correct nodata_ratio)
      - "window": reads minimal window and masks (more efficient)
    """
    if polygons.empty:
        return polygons.copy()

    if not polygons.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).all():
        raise ValueError("zonal_stats expects Polygon/MultiPolygon geometries only.")

    gdf = polygons.copy()

    with rasterio.open(raster_path) as ds:
        gdf, _ = ensure_same_crs(gdf, CRS.from_user_input(ds.crs))

        transform = ds.transform
        pixel_area = abs(transform.a * transform.e)

        results = []
        for geom in gdf.geometry:
            if geom is None or geom.is_empty:
                results.append({s: np.nan for s in stats})
                continue

            geom_area = geometry_area(geom, CRS.from_user_input(ds.crs))

            edge_total = None
            edge_valid = None
            if "edge_coverage_ratio" in stats:
                edge_total, edge_valid = _edge_counts(ds, geom, band=band, all_touched=all_touched)

            if engine == "mask":
                try:
                    # shape_mask is a geometry-only mask (does NOT apply nodata)
                    # For numpy masks: inside geometry => False, outside => True
                    shape_mask, _out_transform, window = raster_geometry_mask(
                        ds,
                        [geom],
                        crop=True,
                        all_touched=all_touched,
                        invert=False,
                    )
                except ValueError:
                    # shapes do not overlap raster
                    vals = np.array([], dtype=float)
                    total_count = 0
                    valid_count = 0
                else:
                    arr = ds.read(band, window=window).astype(float)

                    inside = ~shape_mask  # True inside polygon
                    total_count = int(np.count_nonzero(inside))

                    if total_count == 0:
                        vals = np.array([], dtype=float)
                        valid_count = 0
                    else:
                        inside_vals = arr[inside]
                        m = valid_mask(inside_vals, ds.nodata)
                        vals = inside_vals[m]
                        valid_count = int(vals.size)

            elif engine == "window":
                win = geometry_window(ds, [geom], pad_x=0, pad_y=0, north_up=True, rotated=False)
                arr = ds.read(band, window=win).astype(float)
                win_transform = ds.window_transform(win)

                geom_m = geometry_mask(
                    [geom],
                    out_shape=arr.shape,
                    transform=win_transform,
                    invert=True,
                    all_touched=all_touched,
                )

                total_count = int(np.count_nonzero(geom_m))
                if total_count == 0:
                    vals = np.array([], dtype=float)
                    valid_count = 0
                else:
                    arr_inside = arr[geom_m]
                    m = valid_mask(arr_inside, ds.nodata)
                    vals = arr_inside[m]
                    valid_count = int(vals.size)
            else:
                raise ValueError("engine must be 'mask' or 'window'.")

            stat_row = _compute_stats(
                values=vals,
                total_count=total_count,
                valid_count=valid_count,
                geom_area=geom_area,
                pixel_area=pixel_area,
                stats=stats,
                edge_total=edge_total,
                edge_valid=edge_valid,
            )
            results.append(stat_row)

    stats_df = pd.DataFrame(results)
    if keep_geometry:
        return gdf.join(stats_df)

    return gpd.GeoDataFrame(stats_df)
