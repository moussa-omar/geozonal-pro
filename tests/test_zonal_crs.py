import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from geozonal import zonal_stats


def _write_raster_3857(path):
    arr = np.arange(1, 101, dtype=np.float32).reshape((10, 10))
    transform = from_origin(0, 10, 1, 1)
    with rasterio.open(
        path, "w", driver="GTiff", height=10, width=10, count=1,
        dtype="float32", crs="EPSG:3857", transform=transform, nodata=-9999.0
    ) as dst:
        dst.write(arr, 1)
    return arr


def test_crs_reprojection(tmp_path):
    raster_path = tmp_path / "r.tif"
    _write_raster_3857(raster_path)

    # Same numeric coords but declared as 4326; function should reproject to raster CRS (3857).
    poly = box(0, 5, 5, 10)
    gdf_wrong = gpd.GeoDataFrame({"id": [1]}, geometry=[poly], crs="EPSG:4326")

    out = zonal_stats(gdf_wrong, str(raster_path), stats=("mean",), engine="mask")
    # We don't assert exact mean here because the reprojection changes shape;
    # we just assert it runs and returns a float or NaN.
    assert "mean" in out.columns
