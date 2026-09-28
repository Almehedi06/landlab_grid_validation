"""Landslide probability field vs the landslide validation points, as Landlab node fields.

Both become named fields on one RasterModelGrid and are compared by name, the
same call a Landlab model script would make. Every cell without a point counts
as a negative here (no pseudo-absence mask).

    python examples/landslide_field_vs_points.py [--data /mnt/c/Users/amehedi/Downloads]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

from landlab_grid_validation import (  # noqa: E402
    FieldSpec,
    compare_fields_on_grid,
    grid_from_raster,
    plot_comparison,
    raster_to_node_field,
    vector_to_node_field,
)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("/mnt/c/Users/amehedi/Downloads"))
    args = ap.parse_args(argv)
    pio = args.data / "pioneer"
    tif = pio / "probability_fields/P_Landslide.tif"

    grid, crs = grid_from_raster(tif)
    points = gpd.read_file(pio / "Initiation_Points_final/Initiation_Points_final.shp").to_crs(crs)
    landslides = points[points.FailureTyp == "LS"]

    grid.add_field("landslide__probability_of_failure",
                   raster_to_node_field(grid, crs, tif), at="node")
    grid.add_field("landslide__validation_points",
                   vector_to_node_field(grid, crs, landslides, tolerance_cells=1), at="node")

    result = compare_fields_on_grid(
        grid, "landslide__probability_of_failure", "landslide__validation_points",
        FieldSpec(name="Landslide probability", kind="continuous", units="probability",
                  threshold=0.5),
        FieldSpec(name="landslide points (3 x 3 cells)", kind="binary"),
    )
    out = pio / "grid_validation_example/landslide_field_vs_points.png"
    plot_comparison(grid, result, points=landslides, marker_size=8).savefig(out, dpi=200)
    print(f"AUC {result.metrics['roc_auc']:.3f}  ->  {out}")


if __name__ == "__main__":
    main()
