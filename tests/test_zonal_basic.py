import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from geozonal import zonal_stats


def _write_test_raster(path):
    # 10x10 values 1..100 row-major
    arr = np.arange(1, 101, dtype=np.float32).reshape((10, 10))
    transform = from_origin(0, 10, 1, 1)  # pixel size 1x1, top-left at (0,10)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=10,
        width=10,
        count=1,
        dtype="float32",
        crs="EPSG:3857",
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(arr, 1)
    return arr


def test_zonal_mean_min_max(tmp_path):
    raster_path = tmp_path / "r.tif"
    arr = _write_test_raster(raster_path)

    # Polygon covering top-left 5x5 pixels: x in [0,5], y in [5,10]
    poly = box(0, 5, 5, 10)
    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[poly], crs="EPSG:3857")

    out = zonal_stats(gdf, str(raster_path), stats=("mean", "min", "max"), engine="mask")
    # top-left 5x5 corresponds to rows 0..4, cols 0..4
    sub = arr[0:5, 0:5]
    assert np.isclose(out.loc[0, "mean"], float(sub.mean()))
    assert np.isclose(out.loc[0, "min"], float(sub.min()))
    assert np.isclose(out.loc[0, "max"], float(sub.max()))
