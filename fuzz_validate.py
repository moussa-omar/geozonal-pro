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
    """
    Generate a random axis-aligned rectangle within the given extent.
    Sizes are random but bounded to avoid degenerate geometries.
    """
    w = random.uniform(min_size, max_size)
    h = random.uniform(min_size, max_size)
    x0 = random.uniform(xmin, xmax - w)
    y0 = random.uniform(ymin, ymax - h)
    return box(x0, y0, x0 + w, y0 + h)


def _write_random_raster(path: Path, w: int, h: int, nodata: float, nodata_frac: float) -> np.ndarray:
    """
    Create a random float raster (values in [0,100]) and inject a fraction of nodata pixels.
    This simulates realistic imperfect remote-sensing rasters (cloud masks, missing data, etc.).
    """
    arr = np.random.uniform(0, 100, size=(h, w)).astype(np.float32)

    # Inject nodata pixels by randomly selecting indices
    n = int(w * h * nodata_frac)
    if n > 0:
        idx = np.random.choice(w * h, size=n, replace=False)
        arr.flat[idx] = nodata

    # Pixel size 1x1, top-left at (0,h) => extent x:[0,w], y:[0,h]
    transform = from_origin(0, h, 1, 1)

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
    """
    Fuzz validation (property-based testing style):
    - Generate many random rasters and random rectangles
    - Compute zonal stats with both engines ("mask" and "window")
    - Assert the two engines agree on core outputs and invariants hold
    """
    random.seed(0)
    np.random.seed(0)

    trials = 80
    zones_per_trial = 25
    nodata = -9999.0

    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)

        for t in range(trials):
            # Random raster size and nodata fraction per trial
            w = random.randint(20, 80)
            h = random.randint(20, 80)
            nodata_frac = random.uniform(0.0, 0.25)

            raster_path = tmp / f"r_{t}.tif"
            _write_random_raster(raster_path, w, h, nodata=nodata, nodata_frac=nodata_frac)

            # Random rectangles inside raster extent
            xmin, ymin, xmax, ymax = 0.0, 0.0, float(w), float(h)
            geoms = [_rand_box_within(xmin, ymin, xmax, ymax) for _ in range(zones_per_trial)]
            gdf = gpd.GeoDataFrame({"id": list(range(zones_per_trial))}, geometry=geoms, crs="EPSG:3857")

            # Compute stats using both implementations to cross-validate
            stats = ("mean", "min", "max", "count", "nodata_ratio", "coverage_ratio")
            out_mask = zonal_stats(gdf, str(raster_path), stats=stats, engine="mask", all_touched=False)
            out_win = zonal_stats(gdf, str(raster_path), stats=stats, engine="window", all_touched=False)

            for i in range(zones_per_trial):
                # Invariant: valid pixel count should match between engines
                c1 = int(out_mask.loc[i, "count"])
                c2 = int(out_win.loc[i, "count"])
                if c1 != c2:
                    raise AssertionError(f"[trial {t}] engine mismatch count zone {i}: {c1} vs {c2}")

                # Compare numeric outputs; handle NaN consistently
                for k in ["mean", "min", "max", "nodata_ratio", "coverage_ratio"]:
                    a = float(out_mask.loc[i, k])
                    b = float(out_win.loc[i, k])

                    # Ratio sanity: within [0,1] when defined
                    if k in ("nodata_ratio", "coverage_ratio"):
                        if not math.isnan(a) and not (0.0 <= a <= 1.0):
                            raise AssertionError(f"[trial {t}] mask {k} out of [0,1] zone {i}: {a}")
                        if not math.isnan(b) and not (0.0 <= b <= 1.0):
                            raise AssertionError(f"[trial {t}] window {k} out of [0,1] zone {i}: {b}")

                    # If one is NaN, the other must be NaN too (engine consistency)
                    if math.isnan(a) or math.isnan(b):
                        if not (math.isnan(a) and math.isnan(b)):
                            raise AssertionError(f"[trial {t}] engine mismatch {k} zone {i}: {a} vs {b}")
                        continue

                    # When both are numbers, they should match within tolerance
                    if not np.isclose(a, b, atol=1e-6, rtol=0.0):
                        raise AssertionError(f"[trial {t}] engine mismatch {k} zone {i}: {a} vs {b}")

                # Invariant: when count > 0, mean must lie within [min, max]
                if c1 > 0:
                    for out, label in [(out_mask, "mask"), (out_win, "window")]:
                        mn = float(out.loc[i, "min"])
                        mx = float(out.loc[i, "max"])
                        me = float(out.loc[i, "mean"])
                        if not (mn <= me <= mx):
                            raise AssertionError(f"[trial {t}] {label} min<=mean<=max violated zone {i}")

                # Invariant: when count == 0, value stats should be NaN
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
