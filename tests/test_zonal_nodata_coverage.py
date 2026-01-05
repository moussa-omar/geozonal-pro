import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from geozonal import zonal_stats


def _write_test_raster_with_nodata(path):
    arr = np.arange(1, 101, dtype=np.float32).reshape((10, 10))
    nodata = -9999.0
    # make a 2x2 nodata block inside top-left 5x5 (rows 1..2, cols 1..2)
    arr[1:3, 1:3] = nodata

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
    return arr, nodata


def test_nodata_ratio_and_coverage(tmp_path):
    raster_path = tmp_path / "r.tif"
    arr, nodata = _write_test_raster_with_nodata(raster_path)

    poly = box(0, 5, 5, 10)  # 5x5 pixels => 25 total
    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[poly], crs="EPSG:3857")

    out = zonal_stats(
        gdf,
        str(raster_path),
        stats=("mean", "nodata_ratio", "coverage_ratio", "count"),
        engine="mask",
    )

    # total 25 pixels, 4 nodata => valid 21
    expected_valid = 25 - 4
    assert out.loc[0, "count"] == expected_valid
    assert np.isclose(out.loc[0, "nodata_ratio"], 4 / 25)

    # coverage_ratio uses valid_area / polygon_area; here pixel area=1, polygon area=25 => 21/25
    assert np.isclose(out.loc[0, "coverage_ratio"], expected_valid / 25)
