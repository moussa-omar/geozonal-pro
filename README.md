# GeoZonal Pro

GeoZonal Pro is a Python project for **raster zonal statistics** (statistics of raster values inside polygon zones) and **raster sampling** (extracting raster values at point locations).  
It is designed to be **CRS-safe**, **nodata-aware**, and **easy to validate** both in Python and visually in **QGIS**.

The repository includes:
- a small installable package (`src/geozonal/`)
- a demo script that generates reproducible outputs
- unit tests (pytest)
- a deterministic validation script (“golden” checks)
- a fuzz validation script (randomized stress testing)
- QGIS-ready outputs (GeoJSON + GeoTIFF from the demo)

---

## What the project does

### 1) Zonal statistics (Raster → Polygons)
Given:
- a raster (GeoTIFF)
- a GeoDataFrame with Polygon / MultiPolygon geometries

`zonal_stats(...)` computes nodata-aware statistics **per polygon**.

**Supported statistics**
- `count`: number of **valid** pixels inside the polygon
- `min`, `max`
- `mean`, `median`, `std`
- `p10`, `p90`
- `nodata_ratio`: fraction of pixels inside polygon that are nodata
- `coverage_ratio`: fraction of polygon area covered by valid pixels  
  computed as:  
  `coverage_ratio = (valid_count * pixel_area) / polygon_area`, clamped to `[0, 1]`

**CRS handling**
- If polygon CRS differs from raster CRS, polygons are **automatically reprojected** to the raster CRS before computation.

**Engines**
Two internal computation strategies are supported:
- `engine="mask"`: uses geometry masking with `rasterio` masking utilities (robust reference implementation).
- `engine="window"`: reads a minimal window plus a geometry mask (more efficient).
The repo includes validation to ensure both engines match on key metrics.

**Edge cases**
- Polygon does not overlap raster → returns `count=0` and value stats as `NaN` (ratios may be `NaN` if undefined).
- Empty geometry → returns `NaN` stats.

---

### 2) Raster sampling (Raster → Points)
Given:
- a raster (GeoTIFF)
- a GeoDataFrame of Point geometries

`sample_raster(...)` extracts raster values at each point.

**Methods**
- `nearest`: nearest pixel sampling (default)
- `bilinear`: bilinear interpolation for continuous rasters

**Nodata handling**
- Raster nodata values are converted to `NaN`.

**CRS handling**
- If point CRS differs from raster CRS, points are automatically reprojected to the raster CRS.

---

## Repository structure

geozonal-pro/
├─ src/geozonal/
│ ├─ init.py
│ ├─ zonal.py # zonal_stats implementation
│ ├─ sampling.py # sample_raster implementation (nearest + bilinear)
│ ├─ crs.py # CRS helpers (ensure_same_crs, geometry_area, etc.)
│ ├─ raster.py # nodata/valid-mask helpers
│ └─ cli.py # optional CLI
├─ tests/
│ ├─ test_zonal_basic.py
│ ├─ test_zonal_crs.py
│ ├─ test_zonal_nodata_coverage.py
│ ├─ test_zonal_robust_stats.py
│ ├─ test_zonal_edge_coverage.py
│ ├─ test_sampling.py
│ └─ test_sampling_bilinear.py
├─ demo_quickstart.py # reproducible demo + exports for QGIS
├─ validate_project.py # deterministic “golden” validation report
├─ fuzz_validate.py # randomized stress test + engine consistency checks
├─ pyproject.toml
└─ README.md
---

## Installation and environment

### 1) Create a virtual environment


python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
2) Install dependencies
Editable install (recommended if the package is configured in pyproject.toml):

pip install -e ".[dev]"
If you want manual install of main tools:

pip install numpy pandas geopandas rasterio shapely pyproj pytest ruff
Quickstart demo (reproducible example)
Run:

python demo_quickstart.py
What it does:

Creates a synthetic 10×10 raster with values 1..100.

Injects a 2×2 nodata block inside the top-left zone (4 pixels) to demonstrate nodata behavior.

Creates two polygon zones:

Zone 1: top-left 5×5 pixels (x:[0,5], y:[5,10])

Zone 2: bottom-right 5×5 pixels (x:[5,10], y:[0,5])

Runs zonal_stats() and prints a table.

Creates two points and runs sample_raster().

Saves outputs that can be opened in QGIS.

Files saved by the demo:

demo_raster.tif

demo_zones.geojson

demo_zonal_out.geojson

demo_sampled_points.geojson

Understanding nodata_ratio and coverage_ratio in the demo
Because the demo injects 4 nodata pixels inside Zone 1:

Zone 1 total pixels = 25

valid pixels = 21
So:

count = 21

nodata_ratio = 4/25 = 0.16

coverage_ratio = 21/25 = 0.84

Zone 2 does not contain nodata in the demo, so it typically has:

count = 25

nodata_ratio = 0.0

coverage_ratio = 1.0

Usage
A) Zonal statistics example
python
Copia codice
import geopandas as gpd
from geozonal import zonal_stats

zones = gpd.read_file("zones.geojson")

out = zonal_stats(
    zones,
    "raster.tif",
    stats=("mean", "min", "max", "std", "p10", "p90", "count", "nodata_ratio", "coverage_ratio"),
    engine="mask",          # "mask" or "window"
    all_touched=False,
)

out.to_file("zonal_out.geojson", driver="GeoJSON")
Parameters (summary)

stats: which metrics to compute (tuple of names)

band: raster band index (default: 1)

all_touched: whether to include pixels touched by polygon boundary

engine: "mask" (reference) or "window" (efficient)

keep_geometry: keep original geometry in output

B) Raster sampling example
python
Copia codice
import geopandas as gpd
from geozonal import sample_raster

pts = gpd.read_file("points.geojson")

out = sample_raster(
    pts,
    "raster.tif",
    out_col="value",
    method="nearest",       # or "bilinear"
)

out.to_file("points_with_values.geojson", driver="GeoJSON")
Testing and validation
This repository includes three complementary validation layers.

1) Unit tests (pytest)
Run all tests:

python -m pytest -q
The tests cover:

basic zonal stats on known synthetic rasters

CRS reprojection behavior

nodata handling (count, nodata_ratio, coverage_ratio)

point sampling (nearest)

bilinear sampling

2) Deterministic validation suite (human-readable report)
Run:

python validate_project.py
This script:

generates small deterministic rasters/zones

checks expected outputs (“golden values”)

checks sampling correctness

checks engine consistency

checks no-overlap behavior

It prints a report like:

VALIDATION REPORT
--------------------------------------------------------------------------------
golden_no_nodata     PASS  ...
golden_with_nodata   PASS  ...
sampling             PASS  ...
engine_consistency   PASS  ...
no_overlap           PASS  ...
--------------------------------------------------------------------------------
ALL CHECKS PASSED: 5/5.
3) Fuzz validation (randomized stress testing)
Run:

python fuzz_validate.py
This script:

generates random rasters (random size + random nodata fraction)

generates random rectangular zones

runs zonal_stats() with both engines (mask and window)

asserts that outputs match within tolerance

asserts invariants like:

ratios stay in [0,1] when defined

if count>0 then min <= mean <= max

if count==0 then value stats are NaN

Expected final message:


FUZZ VALIDATION PASSED: 80 trials x 25 zones (all_touched=False).
QGIS workflow (visual verification)
Open QGIS.

Drag-and-drop the demo outputs:

demo_raster.tif

demo_zones.geojson

demo_zonal_out.geojson

demo_sampled_points.geojson

Inspect zonal output:

Right click demo_zonal_out.geojson → Open Attribute Table

Review columns: mean, min, max, count, nodata_ratio, coverage_ratio, etc.

Optional numeric cross-check in QGIS:

Processing Toolbox → Zonal statistics

Raster layer: demo_raster.tif

Polygon layer: demo_zones.geojson

Match all_touched-style behavior (if available)

Compare mean/min/max/count with GeoZonal Pro outputs

This provides a practical external check using a widely used GIS tool.

Code quality (linting)
Run Ruff:


ruff check .
Auto-fix safe issues:


ruff check . --fix