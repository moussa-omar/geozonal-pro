import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box, Point

from geozonal import zonal_stats, sample_raster


def main() -> None:
    # 1) Create a tiny raster (10x10) with values 1..100
    arr = np.arange(1, 101, dtype=np.float32).reshape((10, 10))

    # OPTIONAL: inject nodata inside the top-left 5x5 zone (2x2 block = 4 pixels)
    # This will make zone 1: count=21, nodata_ratio=4/25=0.16, coverage_ratio=21/25=0.84
    nodata = -9999.0
    arr[1:3, 1:3] = nodata

    transform = from_origin(0, 10, 1, 1)  # pixel size 1x1, top-left at (0,10)

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

    # 2) Create polygons (zones)
    # Zone 1 covers top-left 5x5 pixels: x in [0,5], y in [5,10]
    # Zone 2 covers bottom-right 5x5 pixels: x in [5,10], y in [0,5]
    poly1 = box(0, 5, 5, 10)
    poly2 = box(5, 0, 10, 5)
    zones = gpd.GeoDataFrame({"zone_id": [1, 2]}, geometry=[poly1, poly2], crs="EPSG:3857")

    # 3) Run zonal stats
    out = zonal_stats(
        zones,
        raster_path,
        stats=("mean", "min", "max", "std", "p10", "p90", "nodata_ratio", "coverage_ratio", "count"),
        engine="mask",
    )

    print("\nZONAL STATS RESULT:")
    print(out.drop(columns="geometry").to_string(index=False))

    # 4) Create points and sample raster
    # Point at center of pixel (row=0, col=0) is (0.5, 9.5) -> value should be 1 (unless nodata injected there)
    # Point at center of pixel (row=9, col=9) is (9.5, 0.5) -> value should be 100
    pts = gpd.GeoDataFrame(
        {"pt_id": [1, 2]},
        geometry=[Point(0.5, 9.5), Point(9.5, 0.5)],
        crs="EPSG:3857",
    )
    sampled = sample_raster(pts, raster_path, out_col="rval")

    print("\nSAMPLED POINTS RESULT:")
    print(sampled.drop(columns="geometry").to_string(index=False))

    # 5) Save outputs
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
