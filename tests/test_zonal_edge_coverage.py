import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from geozonal import zonal_stats


def test_edge_coverage_ratio(tmp_path):
    path = tmp_path / "r.tif"

    # 10x10 raster, pixel size 1, extent x:[0,10], y:[0,10]
    arr = np.ones((10, 10), dtype=np.float32)
    nodata = -9999.0

    # Set nodata on top row and left column (on the edge ring)
    arr[0, :] = nodata
    arr[:, 0] = nodata

    transform = from_origin(0, 10, 1, 1)
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
        nodata=nodata,
    ) as dst:
        dst.write(arr, 1)

    # Polygon aligned with raster extent
    poly = box(0, 0, 10, 10)
    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[poly], crs="EPSG:3857")

    out = zonal_stats(
        gdf,
        str(path),
        stats=("edge_coverage_ratio",),
        engine="mask",
        all_touched=False,
    )

    # Edge ring (1 pixel thick): perimeter pixels = 10*2 + 8*2 = 36
    total_edge = 36
    nodata_edge = 10 + 10 - 1  # top row + left col - overlap corner
    valid_edge = total_edge - nodata_edge
    expected = valid_edge / total_edge

    assert np.isclose(out.loc[0, "edge_coverage_ratio"], expected)
