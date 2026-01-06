import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box, Point

from geozonal import zonal_stats, sample_raster


def main() -> None:
    # ---------------------------------------------------------------------
    # 1) Create a tiny synthetic raster (10x10) with values 1..100.
    #    This makes expected outputs easy to reason about and verify.
    # ---------------------------------------------------------------------
    arr = np.arange(1, 101, dtype=np.float32).reshape((10, 10))

    # Optional: inject nodata inside the top-left zone (a 2x2 block = 4 pixels).
    # This demonstrates that:
    #   - count counts only valid pixels
    #   - nodata_ratio increases accordingly
    #   - coverage_ratio decreases accordingly
    nodata = -9999.0
    arr[1:3, 1:3] = nodata

    # Affine transform: top-left corner at (0, 10), pixel size 1x1.
    # So raster spans x:[0,10], y:[0,10] in EPSG:3857 coordinates.
    transform = from_origin(0, 10, 1, 1)

    raster_path = "demo_raster.tif"
    with rasterio.open(
        raster_path,
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

    # ---------------------------------------------------------------------
    # 2) Create polygon zones (vector layer).
    #    Zone 1: top-left 5x5 pixels (x:[0,5], y:[5,10])
    #    Zone 2: bottom-right 5x5 pixels (x:[5,10], y:[0,5])
    # ---------------------------------------------------------------------
    poly1 = box(0, 5, 5, 10)
    poly2 = box(5, 0, 10, 5)
    zones = gpd.GeoDataFrame({"zone_id": [1, 2]}, geometry=[poly1, poly2], crs="EPSG:3857")

    # ---------------------------------------------------------------------
    # 3) Run zonal statistics (nodata-aware).
    #    engine="mask" ensures robust geometry masking and correct nodata_ratio.
    # ---------------------------------------------------------------------
    out = zonal_stats(
        zones,
        raster_path,
        stats=("mean", "min", "max", "std", "p10", "p90", "nodata_ratio", "coverage_ratio", "count"),
        engine="mask",
    )

    print("\nZONAL STATS RESULT:")
    print(out.drop(columns="geometry").to_string(index=False))

    # ---------------------------------------------------------------------
    # 4) Sample raster values at points.
    #    Points are placed at pixel centers:
    #      - (0.5, 9.5) -> top-left pixel (row=0, col=0) ~ value 1
    #      - (9.5, 0.5) -> bottom-right pixel (row=9, col=9) ~ value 100
    # ---------------------------------------------------------------------
    pts = gpd.GeoDataFrame(
        {"pt_id": [1, 2]},
        geometry=[Point(0.5, 9.5), Point(9.5, 0.5)],
        crs="EPSG:3857",
    )
    sampled = sample_raster(pts, raster_path, out_col="rval")

    print("\nSAMPLED POINTS RESULT:")
    print(sampled.drop(columns="geometry").to_string(index=False))

    # ---------------------------------------------------------------------
    # 5) Save demo outputs to disk (for QGIS inspection / reproducibility).
    # ---------------------------------------------------------------------
    zones.to_file("demo_zones.geojson", driver="GeoJSON")
    out.to_file("demo_zonal_out.geojson", driver="GeoJSON")
    sampled.to_file("demo_sampled_points.geojson", driver="GeoJSON")

    print("\nSaved files:")
    print(" - demo_raster.tif")
    print(" - demo_zones.geojson")
    print(" - demo_zonal_out.geojson")
    print(" - demo_sampled_points.geojson")


if __name__ == "__main__":
    main()
