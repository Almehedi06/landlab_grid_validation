"""Pioneer / Stehekin: the package checked against results we already have.

1. The three Landlab probability layers against all 89 mapped initiation points
   (3 x 3-cell blocks), with the 90 pseudo-absence blocks as negatives. The AUCs
   must equal the earlier scoring in pioneer/cellwise_validation (class All,
   balanced negatives); the script stops if they do not.
2. Landlab MWR elevation change against the dDEM averaged onto the same 10 m
   grid, both classed as erosion / stable / deposition at +/-0.5 m.

    python examples/pioneer.py [--data /mnt/c/Users/amehedi/Downloads] [--out DIR]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from landlab_grid_validation import (  # noqa: E402
    FieldSpec,
    compare_fields_on_grid,
    grid_from_raster,
    plot_comparison,
    raster_to_node_field,
    vector_to_node_field,
)

LAYERS = [("P_Landslide", "Landslide"), ("P_Channel_ini", "Channel initiation"),
          ("P_Combined_H", "Combined hillslope")]
CHANGE = {0: "stable", 1: "erosion", 2: "deposition"}
CHANGE_COLOURS = {0: "#eeeeee", 1: "#b2182b", 2: "#2166ac"}


def classify(dz, lod=0.5):
    """Erosion (1) below -lod, deposition (2) above +lod, stable (0) between; NaN stays NaN."""
    classes = np.select([dz < -lod, dz > lod], [1.0, 2.0], 0.0)
    return np.where(np.isfinite(dz), classes, np.nan)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("/mnt/c/Users/amehedi/Downloads"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    pio = args.data / "pioneer"
    out = args.out or pio / "grid_validation_example"
    out.mkdir(parents=True, exist_ok=True)
    summary = {}

    # 1 -- probability layers vs mapped initiations -------------------------
    grid, crs = grid_from_raster(pio / "probability_fields/P_Landslide.tif")
    points = gpd.read_file(pio / "Initiation_Points_final/Initiation_Points_final.shp").to_crs(crs)
    observed = vector_to_node_field(grid, crs, points, tolerance_cells=1)
    absence = vector_to_node_field(grid, crs, pio / "negative_points/random90_unbiased.shp",
                                   tolerance_cells=1)
    mask = (observed == 1) | (absence == 1)          # evidence plus pseudo-absence blocks
    print(f"evidence cells {int(observed.sum())}, negative cells "
          f"{int((mask & (observed == 0)).sum())}")

    with open(pio / "cellwise_validation/cellwise_metrics.csv") as f:
        earlier = {r["layer"]: float(r["roc_auc"]) for r in csv.DictReader(f)
                   if r["mechanism"] == "All" and r["negatives"] == "balanced"}

    for key, label in LAYERS:
        probability = raster_to_node_field(grid, crs, pio / f"probability_fields/{key}.tif")
        result = compare_fields_on_grid(
            grid, probability, observed,
            FieldSpec(name=label, kind="continuous", units="probability", threshold=0.5),
            FieldSpec(name="mapped initiations", kind="binary"),
            mask=mask,
        )
        auc = result.metrics["roc_auc"]
        match = abs(auc - earlier[key]) < 1e-9
        print(f"  {label:<20} AUC {auc:.6f}   earlier {earlier[key]:.6f}   "
              f"{'match' if match else 'MISMATCH'}")
        if not match:
            raise SystemExit(f"{label}: AUC differs from the earlier scoring")
        summary[label] = result.to_dict()
        fig = plot_comparison(grid, result, points=points, marker_size=5)
        fig.savefig(out / f"{key}_vs_initiations.png", dpi=200)
        plt.close(fig)

    # 2 -- MWR elevation change vs dDEM, as erosion / stable / deposition ----
    mwr = raster_to_node_field(grid, crs, args.data / "MWR_final_results.asc/MWR.tif")
    ddem = raster_to_node_field(
        grid, crs,
        pio / "main_pipeline_run/corrections/pioneer/final_dem_products/dod_corrected.tif",
        resampling="average",
    )
    change = dict(kind="categorical", classes=CHANGE)
    result = compare_fields_on_grid(
        grid, classify(mwr), classify(ddem),
        FieldSpec(name="Landlab MWR", **change),
        FieldSpec(name="dDEM, 10 m mean", **change),
    )
    m = result.metrics
    print(f"  MWR vs dDEM classes: overall agreement {m['overall_accuracy']:.3f} "
          f"over {m['n']:,} cells")
    for name, s in m["per_class"].items():
        print(f"    {CHANGE[int(name)]:<10} recall {s['recall']:.2f}  precision {s['precision']:.2f}"
              f"  (observed cells {s['support']:,})")
    summary["MWR vs dDEM"] = result.to_dict()
    fig = plot_comparison(grid, result, class_colours=CHANGE_COLOURS)
    fig.savefig(out / "mwr_vs_ddem_classes.png", dpi=200)
    plt.close(fig)

    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"outputs: {out}")


if __name__ == "__main__":
    main()
