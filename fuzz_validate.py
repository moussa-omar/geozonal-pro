import math
import random
import tempfile
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box, Polygon

from geozonal import zonal_stats


def _rand_box_within(xmin, ymin, xmax, ymax, min_size=0.5, max_size=8.0) -> Polygon:
    """Random axis-aligned rectangle within extent."""
    w = random.uniform(min_size, max_size)
    h = random.uniform(min_size, max_size)
    x0 = random.uniform(xmin, xmax - w)
    y0 = random.uniform(ymin, ymax - h)
    return box(x0, y0, x0 + w, y0 + h)


def _write_random_raster(path: Path, w: int, h: int, nodata: float, nodata_frac: float) -> np.ndarray:
    """Random float raster, with some nodata injected."""
    arr = np.random.uniform(0, 100, size=(h, w)).astype(np.float32)

    n = int(w * h * nodata_frac)
    if n > 0:
        idx = np.random.choice(w * h, size=n, replace=False)
        arr.flat[idx] = nodata

    transform = from_origin(0, h, 1, 1)  # extent: x in [0,w], y in [0,h]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype="float32",
        crs="EPSG:3857",
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(arr, 1)

    return arr


def main() -> int:
    random.seed(0)
    np.random.seed(0)

    trials = 80
    zones_per_trial = 25
    nodata = -9999.0

    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)

        for t in range(trials):
            w = random.randint(20, 80)
            h = random.randint(20, 80)
            nodata_frac = random.uniform(0.0, 0.25)

            raster_path = tmp / f"r_{t}.tif"
            _write_random_raster(raster_path, w, h, nodata=nodata, nodata_frac=nodata_frac)

            xmin, ymin, xmax, ymax = 0.0, 0.0, float(w), float(h)
            geoms = [_rand_box_within(xmin, ymin, xmax, ymax) for _ in range(zones_per_trial)]
            gdf = gpd.GeoDataFrame({"id": list(range(zones_per_trial))}, geometry=geoms, crs="EPSG:3857")

            stats = ("mean", "min", "max", "count", "nodata_ratio", "coverage_ratio")
            out_mask = zonal_stats(gdf, str(raster_path), stats=stats, engine="mask", all_touched=False)
            out_win = zonal_stats(gdf, str(raster_path), stats=stats, engine="window", all_touched=False)

            for i in range(zones_per_trial):
                c1 = int(out_mask.loc[i, "count"])
                c2 = int(out_win.loc[i, "count"])
                if c1 != c2:
                    raise AssertionError(f"[trial {t}] engine mismatch count zone {i}: {c1} vs {c2}")

                for k in ["mean", "min", "max", "nodata_ratio", "coverage_ratio"]:
                    a = float(out_mask.loc[i, k])
                    b = float(out_win.loc[i, k])

                    if k in ("nodata_ratio", "coverage_ratio"):
                        if not math.isnan(a) and not (0.0 <= a <= 1.0):
                            raise AssertionError(f"[trial {t}] mask {k} out of [0,1] zone {i}: {a}")
                        if not math.isnan(b) and not (0.0 <= b <= 1.0):
                            raise AssertionError(f"[trial {t}] window {k} out of [0,1] zone {i}: {b}")

                    if math.isnan(a) or math.isnan(b):
                        if not (math.isnan(a) and math.isnan(b)):
                            raise AssertionError(f"[trial {t}] engine mismatch {k} zone {i}: {a} vs {b}")
                        continue

                    if not np.isclose(a, b, atol=1e-6, rtol=0.0):
                        raise AssertionError(f"[trial {t}] engine mismatch {k} zone {i}: {a} vs {b}")

                if c1 > 0:
                    for out, label in [(out_mask, "mask"), (out_win, "window")]:
                        mn = float(out.loc[i, "min"])
                        mx = float(out.loc[i, "max"])
                        me = float(out.loc[i, "mean"])
                        if not (mn <= me <= mx):
                            raise AssertionError(f"[trial {t}] {label} min<=mean<=max violated zone {i}")

                if c1 == 0:
                    for out, label in [(out_mask, "mask"), (out_win, "window")]:
                        me = float(out.loc[i, "mean"])
                        mn = float(out.loc[i, "min"])
                        mx = float(out.loc[i, "max"])
                        if not (math.isnan(me) and math.isnan(mn) and math.isnan(mx)):
                            raise AssertionError(f"[trial {t}] {label} count=0 but stats not NaN zone {i}")

        print(f"FUZZ VALIDATION PASSED: {trials} trials x {zones_per_trial} zones (all_touched=False).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
