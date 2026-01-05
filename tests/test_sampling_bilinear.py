import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point

from geozonal import sample_raster


def test_bilinear_sampling_center(tmp_path):
    path = tmp_path / "r.tif"
    arr = np.array([[0, 10], [20, 30]], dtype=np.float32)
    transform = from_origin(0, 2, 1, 1)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=1,
        dtype="float32",
        crs="EPSG:3857",
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(arr, 1)

    # Center of the 2x2 block -> average = 15
    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[Point(1, 1)], crs="EPSG:3857")
    out = sample_raster(gdf, str(path), out_col="v", method="bilinear")

    assert np.isclose(out.loc[0, "v"], 15.0)
