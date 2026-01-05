from __future__ import annotations

import argparse

import geopandas as gpd

from .zonal import zonal_stats
from .sampling import sample_raster


def _cmd_zonal(args: argparse.Namespace) -> int:
    gdf = gpd.read_file(args.polygons)
    out = zonal_stats(
        polygons=gdf,
        raster_path=args.raster,
        stats=tuple(args.stats),
        band=args.band,
        all_touched=args.all_touched,
        engine=args.engine,
    )
    if args.out:
        out.to_file(args.out)
    else:
        print(out.drop(columns="geometry").head(10).to_string(index=False))
    return 0


def _cmd_sample(args: argparse.Namespace) -> int:
    gdf = gpd.read_file(args.points)
    out = sample_raster(gdf, args.raster, band=args.band, out_col=args.out_col)
    if args.out:
        out.to_file(args.out)
    else:
        print(out.drop(columns="geometry").head(10).to_string(index=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="geozonal", description="GeoZonal Pro CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    pz = sub.add_parser("zonal", help="Compute zonal stats for polygons over a raster")
    pz.add_argument("--raster", required=True)
    pz.add_argument("--polygons", required=True)
    pz.add_argument("--out", default=None, help="Output vector file (GeoPackage/GeoJSON/etc.)")
    pz.add_argument("--band", type=int, default=1)
    pz.add_argument("--engine", choices=["mask", "window"], default="mask")
    pz.add_argument("--all-touched", action="store_true")
    pz.add_argument(
        "--stats",
        nargs="+",
        default=["mean", "min", "max", "std", "p10", "p90", "nodata_ratio", "coverage_ratio"],
    )
    pz.set_defaults(func=_cmd_zonal)

    ps = sub.add_parser("sample", help="Sample raster values at point locations")
    ps.add_argument("--raster", required=True)
    ps.add_argument("--points", required=True)
    ps.add_argument("--out", default=None)
    ps.add_argument("--band", type=int, default=1)
    ps.add_argument("--out-col", default="value")
    ps.set_defaults(func=_cmd_sample)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
