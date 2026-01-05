from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import rasterio
from rasterio.io import DatasetReader


@dataclass(frozen=True)
class RasterMeta:
    crs: object
    nodata: Optional[float]
    transform: object
    width: int
    height: int


def open_raster(path: str) -> DatasetReader:
    return rasterio.open(path)


def get_raster_meta(ds: DatasetReader) -> RasterMeta:
    return RasterMeta(
        crs=ds.crs,
        nodata=ds.nodata,
        transform=ds.transform,
        width=ds.width,
        height=ds.height,
    )


def valid_mask(arr: np.ndarray, nodata: Optional[float]) -> np.ndarray:
    """
    Returns boolean mask for valid pixels (True = valid).
    Handles nodata and NaNs.
    """
    m = np.isfinite(arr)
    if nodata is not None and np.isfinite(nodata):
        m = m & (arr != nodata)
    return m
