import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point

from geozonal import sample_raster


def _write_test_raster(path):
    arr = np.arange(1, 101, dtype=np.float32).reshape((10, 10))
    transform = from_origin(0, 10, 1, 1)
    with rasterio.open(
        path, "w", driver="GTiff", height=10, width=10, count=1,
        dtype="float32", crs="EPSG:3857", transform=transform, nodata=-9999.0
    ) as dst:
        dst.write(arr, 1)
    return arr


def test_sampling(tmp_path):
    raster_path = tmp_path / "r.tif"
    arr = _write_test_raster(raster_path)

    # Point at center of pixel (col=0,row=0) in this transform is (0.5, 9.5)
    pts = gpd.GeoDataFrame({"id": [1]}, geometry=[Point(0.5, 9.5)], crs="EPSG:3857")
    out = sample_raster(pts, str(raster_path), out_col="v")

    assert np.isclose(out.loc[0, "v"], float(arr[0, 0]))
