import numpy as np
import geopandas as gpd
from shapely.geometry import box

from geozonal import zonal_stats
from tests.test_zonal_basic import _write_test_raster


def test_robust_mean_and_iqr(tmp_path):
    raster_path = tmp_path / "r.tif"
    arr = _write_test_raster(raster_path)

    # Polygon covering top-left 5x5 pixels
    poly = box(0, 5, 5, 10)
    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[poly], crs="EPSG:3857")

    out = zonal_stats(
        gdf,
        str(raster_path),
        stats=("p25", "p75", "iqr", "robust_mean"),
        engine="mask",
    )

    sub = arr[0:5, 0:5].ravel()
    p25 = np.percentile(sub, 25)
    p75 = np.percentile(sub, 75)
    iqr = p75 - p25

    lo = np.percentile(sub, 10)
    hi = np.percentile(sub, 90)
    trimmed = sub[(sub >= lo) & (sub <= hi)]
    rmean = trimmed.mean()

    assert np.isclose(out.loc[0, "p25"], p25)
    assert np.isclose(out.loc[0, "p75"], p75)
    assert np.isclose(out.loc[0, "iqr"], iqr)
    assert np.isclose(out.loc[0, "robust_mean"], rmean)
