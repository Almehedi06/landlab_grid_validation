"""Test the package on Stehekin: mapped points, the dDEM and Landlab MWR (10 m grid).

1. dDEM vs points (known answer): dDEM erosion depth, reprojected bilinearly to
   10 m, against each mechanism's 3 x 3 evidence blocks with the pseudo-absence
   blocks as negatives. The AUCs must equal the earlier dod_erosion rows in
   pioneer/cellwise_validation; the script stops if they do not.
2. MWR vs dDEM: MWR erosion depth against dDEM erosion (10 m mean below -0.5 m),
   exact cell and within one cell.
3. MWR vs points: MWR erosion depth against the evidence blocks. A real test
   only if MWR was not started from these points.

    python examples/mwr_ddem_points.py [--data /mnt/c/Users/amehedi/Downloads] [--out DIR]
"""

from __future__ import annotations

import argparse
import csv
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

MECHANISMS = {"LS": "Landslide", "Runoff": "Runoff", "Firehose": "Firehose", "All": "All types"}


def within_one_cell(grid, yes):
    """Cells with a yes in their 3 x 3 neighbourhood."""
    m = np.nan_to_num(yes).astype(bool).reshape(grid.shape)
    p = np.pad(m, 1)
    rows, cols = m.shape
    out = np.zeros_like(m)
    for dr in range(3):
        for dc in range(3):
            out |= p[dr:dr + rows, dc:dc + cols]
    return out.reshape(-1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("/mnt/c/Users/amehedi/Downloads"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    pio = args.data / "pioneer"
    out = args.out or pio / "grid_validation_mwr_ddem"
    out.mkdir(parents=True, exist_ok=True)
    dod_path = pio / "main_pipeline_run/corrections/pioneer/final_dem_products/dod_corrected.tif"

    grid, crs = grid_from_raster(pio / "probability_fields/P_Landslide.tif")
    points = gpd.read_file(pio / "Initiation_Points_final/Initiation_Points_final.shp").to_crs(crs)
    evidence = {c: vector_to_node_field(grid, crs, points[points.FailureTyp == c], tolerance_cells=1)
                for c in ("LS", "Runoff", "Firehose")}
    evidence["All"] = vector_to_node_field(grid, crs, points, tolerance_cells=1)
    absence = vector_to_node_field(grid, crs, pio / "negative_points/random90_unbiased.shp",
                                   tolerance_cells=1)
    negatives = (absence == 1) & (evidence["All"] == 0)

    ddem_depth = -raster_to_node_field(grid, crs, dod_path, resampling="bilinear")
    mwr_depth = np.clip(-raster_to_node_field(grid, crs, args.data / "MWR_final_results.asc/MWR.tif"),
                        0, None)
    with open(pio / "cellwise_validation/cellwise_metrics.csv") as f:
        earlier = {r["mechanism"]: float(r["roc_auc"]) for r in csv.DictReader(f)
                   if r["layer"] == "dod_erosion" and r["negatives"] == "balanced"}

    rows = []

    def score(test, name, values, code, check=None):
        result = compare_fields_on_grid(
            grid, values, evidence[code],
            FieldSpec(name=name, kind="continuous", units="m", threshold=0.5,
                      sign_convention="positive_is_loss"),
            FieldSpec(name=f"{MECHANISMS[code]} initiations", kind="binary"),
            mask=(evidence[code] == 1) | negatives,
        )
        m = result.metrics
        if check is not None and abs(m["roc_auc"] - check) > 1e-9:
            raise SystemExit(f"{test} / {code}: AUC {m['roc_auc']:.6f} differs from earlier {check:.6f}")
        rows.append(dict(test=test, evidence=MECHANISMS[code], auc=round(m["roc_auc"], 3),
                         recall=round(m["recall"], 2), precision=round(m["precision"], 2)))
        return result

    # 1 -- dDEM vs points (known answer) -------------------------------------
    for code in MECHANISMS:
        result = score("dDEM vs points", "dDEM erosion depth", ddem_depth, code, earlier[code])
    fig = plot_comparison(grid, result, points=points, marker_size=5)
    fig.savefig(out / "1_ddem_vs_points.png", dpi=200)
    plt.close(fig)

    # 2 -- MWR vs dDEM erosion -------------------------------------------------
    ddem_mean = raster_to_node_field(grid, crs, dod_path, resampling="average")
    observed = np.where(np.isfinite(ddem_mean), (ddem_mean < -0.5).astype(float), np.nan)
    result = compare_fields_on_grid(
        grid, mwr_depth, observed,
        FieldSpec(name="MWR erosion depth", kind="continuous", units="m", threshold=0.5),
        FieldSpec(name="dDEM erosion (10 m mean < -0.5 m)", kind="binary"),
    )
    m = result.metrics
    rows.append(dict(test="MWR vs dDEM, exact cell", evidence="dDEM erosion",
                     auc=round(m["roc_auc"], 3), recall=round(m["recall"], 2),
                     precision=round(m["precision"], 2)))
    valid = result.valid_mask
    pred_yes, obs_yes = (result.predicted_yes == 1) & valid, (result.observed_yes == 1) & valid
    rows.append(dict(test="MWR vs dDEM, within 1 cell", evidence="dDEM erosion", auc="",
                     recall=round((obs_yes & within_one_cell(grid, pred_yes)).sum() / obs_yes.sum(), 2),
                     precision=round((pred_yes & within_one_cell(grid, obs_yes)).sum() / pred_yes.sum(), 2)))
    fig = plot_comparison(grid, result)
    fig.savefig(out / "2_mwr_vs_ddem.png", dpi=200)
    plt.close(fig)

    # 3 -- MWR vs points -----------------------------------------------------------
    for code in MECHANISMS:
        result = score("MWR vs points", "MWR erosion depth", mwr_depth, code)
    fig = plot_comparison(grid, result, points=points, marker_size=5)
    fig.savefig(out / "3_mwr_vs_points.png", dpi=200)
    plt.close(fig)

    with open(out / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{'test':<28}{'evidence':<14}{'AUC':>6}{'recall':>8}{'precision':>11}")
    for r in rows:
        print(f"{r['test']:<28}{r['evidence']:<14}{str(r['auc']):>6}{r['recall']:>8}{r['precision']:>11}")
    print("dDEM vs points matches the earlier scoring for all four evidence sets")
    print(f"outputs: {out}")


if __name__ == "__main__":
    main()
