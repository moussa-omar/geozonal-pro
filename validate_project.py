from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import tempfile

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box, Point

from geozonal import zonal_stats, sample_raster


@dataclass
class CheckResult:
    name: str
    ok: bool
    details: str = ""


def _isclose(a: float, b: float, tol: float = 1e-6) -> bool:
    if a is None or b is None:
        return False
    if isinstance(a, float) and math.isnan(a) and isinstance(b, float) and math.isnan(b):
        return True
    return bool(np.isclose(a, b, atol=tol, rtol=0.0))


def _assert(cond: bool, msg: str) -> tuple[bool, str]:
    return (True, msg) if cond else (False, msg)


def _write_demo_raster(path: Path, with_nodata: bool) -> tuple[np.ndarray, float]:
    """
    Writes a 10x10 raster with values 1..100, EPSG:3857, pixel=1.
    Optionally injects a 2x2 nodata block inside the top-left 5x5 zone.
    """
    arr = np.arange(1, 101, dtype=np.float32).reshape((10, 10))
    nodata = -9999.0
    if with_nodata:
        arr[1:3, 1:3] = nodata  # 4 nodata pixels inside top-left 5x5

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


def _build_zones() -> gpd.GeoDataFrame:
    poly1 = box(0, 5, 5, 10)   # top-left 5x5 pixels
    poly2 = box(5, 0, 10, 5)   # bottom-right 5x5 pixels
    return gpd.GeoDataFrame({"zone_id": [1, 2]}, geometry=[poly1, poly2], crs="EPSG:3857")


def check_golden_no_nodata(tmp: Path) -> CheckResult:
    raster_path = tmp / "r_nonodata.tif"
    arr, _nodata = _write_demo_raster(raster_path, with_nodata=False)
    zones = _build_zones()

    out = zonal_stats(
        zones,
        str(raster_path),
        stats=("mean", "min", "max", "count", "nodata_ratio", "coverage_ratio"),
        engine="mask",
    )

    # Expected for zone1 (top-left 5x5): rows 0..4, cols 0..4
    sub1 = arr[0:5, 0:5]
    exp1 = {
        "count": 25,
        "mean": float(sub1.mean()),
        "min": float(sub1.min()),
        "max": float(sub1.max()),
        "nodata_ratio": 0.0,
        "coverage_ratio": 1.0,
    }

    # Expected for zone2 (bottom-right 5x5): rows 5..9, cols 5..9
    sub2 = arr[5:10, 5:10]
    exp2 = {
        "count": 25,
        "mean": float(sub2.mean()),
        "min": float(sub2.min()),
        "max": float(sub2.max()),
        "nodata_ratio": 0.0,
        "coverage_ratio": 1.0,
    }

    for i, exp in [(0, exp1), (1, exp2)]:
        for k, v in exp.items():
            got = out.loc[i, k]
            if isinstance(v, float):
                if not _isclose(float(got), float(v), tol=1e-6):
                    return CheckResult(
                        "golden_no_nodata",
                        False,
                        f"Zone idx {i} field '{k}' expected {v} got {got}",
                    )
            else:
                if int(got) != int(v):
                    return CheckResult(
                        "golden_no_nodata",
                        False,
                        f"Zone idx {i} field '{k}' expected {v} got {got}",
                    )

    return CheckResult("golden_no_nodata", True, "All expected stats match.")


def check_golden_with_nodata(tmp: Path) -> CheckResult:
    raster_path = tmp / "r_withnodata.tif"
    arr, nodata = _write_demo_raster(raster_path, with_nodata=True)
    zones = _build_zones()

    out = zonal_stats(
        zones,
        str(raster_path),
        stats=("mean", "count", "nodata_ratio", "coverage_ratio"),
        engine="mask",
    )

    # Zone1 has 25 pixels, 4 nodata => 21 valid
    sub1 = arr[0:5, 0:5]
    valid1 = sub1[sub1 != nodata]
    exp1 = {
        "count": 21,
        "mean": float(valid1.mean()),
        "nodata_ratio": 4 / 25,
        "coverage_ratio": 21 / 25,
    }

    # Zone2 unaffected
    sub2 = arr[5:10, 5:10]
    exp2 = {
        "count": 25,
        "mean": float(sub2.mean()),
        "nodata_ratio": 0.0,
        "coverage_ratio": 1.0,
    }

    for i, exp in [(0, exp1), (1, exp2)]:
        for k, v in exp.items():
            got = float(out.loc[i, k])
            if not _isclose(got, float(v), tol=1e-6):
                return CheckResult(
                    "golden_with_nodata",
                    False,
                    f"Zone idx {i} field '{k}' expected {v} got {got}",
                )

    return CheckResult("golden_with_nodata", True, "Nodata ratio & coverage behave as expected.")


def check_sampling(tmp: Path) -> CheckResult:
    raster_path = tmp / "r_sample.tif"
    arr, _ = _write_demo_raster(raster_path, with_nodata=False)

    pts = gpd.GeoDataFrame(
        {"pt_id": [1, 2]},
        geometry=[Point(0.5, 9.5), Point(9.5, 0.5)],  # centers of (0,0) and (9,9)
        crs="EPSG:3857",
    )
    out = sample_raster(pts, str(raster_path), out_col="v")

    v1 = float(out.loc[0, "v"])
    v2 = float(out.loc[1, "v"])

    if not _isclose(v1, float(arr[0, 0])):
        return CheckResult("sampling", False, f"Expected first sample {arr[0,0]} got {v1}")
    if not _isclose(v2, float(arr[9, 9])):
        return CheckResult("sampling", False, f"Expected second sample {arr[9,9]} got {v2}")

    return CheckResult("sampling", True, "Point sampling matches expected pixels.")


def check_engine_consistency(tmp: Path) -> CheckResult:
    raster_path = tmp / "r_consistency.tif"
    _arr, _ = _write_demo_raster(raster_path, with_nodata=True)
    zones = _build_zones()

    stats = ("mean", "count", "nodata_ratio", "coverage_ratio")
    a = zonal_stats(zones, str(raster_path), stats=stats, engine="mask")
    b = zonal_stats(zones, str(raster_path), stats=stats, engine="window")

    for i in range(len(zones)):
        for k in stats:
            ga = float(a.loc[i, k]) if k != "count" else int(a.loc[i, k])
            gb = float(b.loc[i, k]) if k != "count" else int(b.loc[i, k])

            if k == "count":
                if ga != gb:
                    return CheckResult("engine_consistency", False, f"Zone {i} count differs: {ga} vs {gb}")
            else:
                if not _isclose(ga, gb, tol=1e-6):
                    return CheckResult("engine_consistency", False, f"Zone {i} '{k}' differs: {ga} vs {gb}")

    return CheckResult("engine_consistency", True, "mask and window engines match on key metrics.")


def check_no_overlap_returns_nan(tmp: Path) -> CheckResult:
    raster_path = tmp / "r_nooverlap.tif"
    _arr, _ = _write_demo_raster(raster_path, with_nodata=False)

    # Polygon far away from raster extent -> should not crash, should return NaN stats
    poly = box(1000, 1000, 1010, 1010)
    zones = gpd.GeoDataFrame({"id": [1]}, geometry=[poly], crs="EPSG:3857")

    out = zonal_stats(zones, str(raster_path), stats=("mean", "count", "nodata_ratio"), engine="mask")

    mean = out.loc[0, "mean"]
    count = out.loc[0, "count"]
    nodr = out.loc[0, "nodata_ratio"]

    # We expect: no overlap -> mean NaN, count 0, nodata_ratio NaN
    ok_mean = isinstance(mean, float) and math.isnan(mean)
    ok_count = int(count) == 0
    ok_nodr = isinstance(nodr, float) and math.isnan(nodr)

    ok, msg = _assert(ok_mean and ok_count and ok_nodr, f"mean={mean}, count={count}, nodata_ratio={nodr}")
    return CheckResult("no_overlap", ok, msg)


def main() -> int:
    results: list[CheckResult] = []

    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)

        results.append(check_golden_no_nodata(tmp))
        results.append(check_golden_with_nodata(tmp))
        results.append(check_sampling(tmp))
        results.append(check_engine_consistency(tmp))
        results.append(check_no_overlap_returns_nan(tmp))

    # Print report
    width = max(len(r.name) for r in results) + 2
    failed = [r for r in results if not r.ok]

    print("\nVALIDATION REPORT")
    print("-" * 80)
    for r in results:
        status = "PASS" if r.ok else "FAIL"
        print(f"{r.name:<{width}} {status}  {r.details}")

    print("-" * 80)
    if failed:
        print(f"FAILED: {len(failed)}/{len(results)} checks.")
        return 1

    print(f"ALL CHECKS PASSED: {len(results)}/{len(results)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
