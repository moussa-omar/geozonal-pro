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


# Supported stats names returned as columns in the output GeoDataFrame
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
    # Robust stats additions
    "p25",
    "p75",
    "iqr",
    "robust_mean",
    # Quality metric on polygon boundary
    "edge_coverage_ratio",
]


def _pixel_size_from_transform(transform) -> float:
    """Approximate pixel size from an affine transform (average of |dx| and |dy|)."""
    px = float(abs(transform.a))
    py = float(abs(transform.e))
    return float((px + py) / 2.0)


def _edge_ring_geom(geom, pixel_size: float):
    """
    Build a ~1-pixel-thick ring around the polygon boundary:
        ring = buffer(+px) - buffer(-px)

    This is useful to measure if the polygon border intersects nodata (e.g., clouds/masks).
    """
    if geom is None or geom.is_empty or pixel_size <= 0:
        return None

    outer = geom.buffer(pixel_size)
    inner = geom.buffer(-pixel_size)

    if outer.is_empty:
        return None
    if inner.is_empty:
        # Very small polygons: inner buffer collapses -> fallback to outer
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
    Return (edge_total_pixels, edge_valid_pixels) for the boundary ring region.
    Valid pixels are those not equal to raster nodata.
    """
    pixel_size = _pixel_size_from_transform(ds.transform)
    ring = _edge_ring_geom(geom, pixel_size)
    if ring is None:
        return 0, 0

    try:
        # raster_geometry_mask returns a geometry-only mask (no nodata masking)
        shape_mask, _out_transform, window = raster_geometry_mask(
            ds, [ring], crop=True, all_touched=all_touched, invert=False
        )
    except ValueError:
        # No overlap with raster
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
    """
    Compute per-zone statistics from:
      - values: valid (non-nodata) raster values inside the zone
      - total_count: total pixels inside zone geometry (including nodata)
      - valid_count: number of valid pixels inside zone
    """
    out: Dict[str, Any] = {}

    # Basic count
    if "count" in stats:
        out["count"] = int(valid_count)

    # If no valid pixels, stats become NaN (except ratios/count)
    if valid_count == 0:
        for s in stats:
            if s in ("count", "nodata_ratio", "coverage_ratio", "edge_coverage_ratio"):
                continue
            out[s] = np.nan
    else:
        # Simple stats
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

        # Percentiles only if requested (saves work)
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

        # IQR = p75 - p25 (robust spread)
        if "iqr" in stats:
            p25 = perc_map.get(25, float(np.percentile(values, 25)))
            p75 = perc_map.get(75, float(np.percentile(values, 75)))
            out["iqr"] = float(p75 - p25)

        # robust_mean: trimmed mean between p10 and p90 (reduces outlier influence)
        if "robust_mean" in stats:
            lo = perc_map.get(10, float(np.percentile(values, 10)))
            hi = perc_map.get(90, float(np.percentile(values, 90)))
            trimmed = values[(values >= lo) & (values <= hi)]
            out["robust_mean"] = float(np.mean(trimmed)) if trimmed.size else float(np.mean(values))

    # nodata_ratio: fraction of pixels inside zone that are nodata
    if "nodata_ratio" in stats:
        out["nodata_ratio"] = float(1.0 - (valid_count / total_count)) if total_count > 0 else np.nan

    # coverage_ratio: how much of the polygon area is covered by valid pixels (approx by pixel area)
    if "coverage_ratio" in stats:
        if geom_area > 0 and valid_count > 0:
            valid_area = valid_count * pixel_area
            out["coverage_ratio"] = float(min(1.0, max(0.0, valid_area / geom_area)))
        else:
            out["coverage_ratio"] = np.nan

    # edge_coverage_ratio: valid pixels on polygon boundary ring / total boundary ring pixels
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
    Nodata-aware zonal statistics for a raster over polygon geometries.

    engine:
      - "mask": uses raster_geometry_mask -> correct total_count and robust nodata_ratio
      - "window": uses geometry_window + geometry_mask -> faster for large rasters
    """
    if polygons.empty:
        return polygons.copy()

    if not polygons.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).all():
        raise ValueError("zonal_stats expects Polygon/MultiPolygon geometries only.")

    gdf = polygons.copy()

    with rasterio.open(raster_path) as ds:
        # Ensure the vector layer is in the raster CRS for correct overlay
        gdf, _ = ensure_same_crs(gdf, CRS.from_user_input(ds.crs))

        pixel_area = abs(ds.transform.a * ds.transform.e)

        results = []
        for geom in gdf.geometry:
            if geom is None or geom.is_empty:
                results.append({s: np.nan for s in stats})
                continue

            geom_area = geometry_area(geom, CRS.from_user_input(ds.crs))

            # Optional edge metric (computed only if requested)
            edge_total = None
            edge_valid = None
            if "edge_coverage_ratio" in stats:
                edge_total, edge_valid = _edge_counts(ds, geom, band=band, all_touched=all_touched)

            # --- Engine selection ---
            if engine == "mask":
                try:
                    shape_mask, _out_transform, window = raster_geometry_mask(
                        ds,
                        [geom],
                        crop=True,
                        all_touched=all_touched,
                        invert=False,
                    )
                except ValueError:
                    # No overlap between geometry and raster extent
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
    return gdf.join(stats_df) if keep_geometry else gpd.GeoDataFrame(stats_df)
