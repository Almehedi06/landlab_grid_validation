"""Pioneer / Stehekin validation with landlab_grid_validation.

1. Landlab probability layers (Landslide, Channel initiation, Combined
   hillslope) against the mapped initiation points: each mechanism and all 89
   pooled, as 3 x 3-cell blocks, with the 90 pseudo-absence blocks as
   negatives. Every AUC must equal the earlier scoring in
   pioneer/cellwise_validation; the script stops if one does not.
2. The USGS post-fire likelihood (basin scale, 24 mm/h) against the same
   evidence, and all four models side by side on the cells USGS assessed.
3. Landlab MWR against the dDEM averaged to 10 m: erosion yes/no (AUC of the
   modelled erosion depth) and erosion / stable / deposition classes at +/-0.5 m.

Writes maps, metrics.csv, summary.json and auc_by_mechanism.png to --out.

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

MECHANISMS = {"LS": "Landslide", "Runoff": "Runoff", "Firehose": "Firehose", "All": "All types"}
LAYERS = [("P_Landslide", "Landslide"), ("P_Channel_ini", "Channel initiation"),
          ("P_Combined_H", "Combined hillslope")]
USGS = "USGS likelihood (24 mm/h)"
COLOURS = {"Landslide": "#8c8c8c", "Channel initiation": "#3d8ce8",
           "Combined hillslope": "#e8a33d", USGS: "#6a3d9a"}
CHANGE = {0: "stable", 1: "erosion", 2: "deposition"}
CHANGE_COLOURS = {0: "#eeeeee", 1: "#b2182b", 2: "#2166ac"}


def classify(dz, lod=0.5):
    """Erosion (1) below -lod, deposition (2) above +lod, stable (0) between; NaN stays NaN."""
    classes = np.select([dz < -lod, dz > lod], [1.0, 2.0], 0.0)
    return np.where(np.isfinite(dz), classes, np.nan)


def save(fig, path):
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("/mnt/c/Users/amehedi/Downloads"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    pio = args.data / "pioneer"
    out = args.out or pio / "grid_validation_example"
    out.mkdir(parents=True, exist_ok=True)

    # evidence on the 10 m model grid ----------------------------------------
    grid, crs = grid_from_raster(pio / "probability_fields/P_Landslide.tif")
    points = gpd.read_file(pio / "Initiation_Points_final/Initiation_Points_final.shp").to_crs(crs)
    evidence = {code: vector_to_node_field(grid, crs, points[points.FailureTyp == code],
                                           tolerance_cells=1)
                for code in ("LS", "Runoff", "Firehose")}
    evidence["All"] = vector_to_node_field(grid, crs, points, tolerance_cells=1)
    absence = vector_to_node_field(grid, crs, pio / "negative_points/random90_unbiased.shp",
                                   tolerance_cells=1)
    negatives = (absence == 1) & (evidence["All"] == 0)   # never score evidence as a negative
    print("evidence cells:", {c: int(v.sum()) for c, v in evidence.items()},
          "| negative cells:", int(negatives.sum()))

    # 1 + 2 -- probability models vs initiation evidence ----------------------
    models = {label: raster_to_node_field(grid, crs, pio / f"probability_fields/{key}.tif")
              for key, label in LAYERS}
    basins = gpd.read_file(pio / "usgs_hazard_pio2024/pio2024-basins.shp")
    models[USGS] = vector_to_node_field(grid, crs, basins, value="P_24mmh",
                                        background=np.nan, all_touched=False)
    in_basins = np.isfinite(models[USGS])

    with open(pio / "cellwise_validation/cellwise_metrics.csv") as f:
        earlier = {(r["layer"], r["mechanism"]): float(r["roc_auc"]) for r in csv.DictReader(f)
                   if r["negatives"] == "balanced"}
    stem = {label: key for key, label in LAYERS}

    rows, summary, common_auc = [], {}, {}
    for label, values in models.items():
        for code, name in MECHANISMS.items():
            for cells in ("all", "USGS basins"):
                if label == USGS and cells == "all":
                    continue                               # USGS exists only in basins
                mask = (evidence[code] == 1) | negatives
                if cells == "USGS basins":
                    mask &= in_basins
                result = compare_fields_on_grid(
                    grid, values, evidence[code],
                    FieldSpec(name=label, kind="continuous", units="probability", threshold=0.5),
                    FieldSpec(name=f"{name} initiations", kind="binary"),
                    mask=mask,
                )
                m = result.metrics
                if cells == "all":
                    before = earlier[(stem[label], code)]
                    if abs(m["roc_auc"] - before) > 1e-9:
                        raise SystemExit(f"{label} / {code}: AUC {m['roc_auc']:.6f} differs "
                                         f"from the earlier {before:.6f}")
                else:
                    common_auc[(label, code)] = m["roc_auc"]
                rows.append(dict(model=label, evidence=name, cells=cells,
                                 n_positive=m["true_positive"] + m["false_negative"],
                                 n_negative=m["true_negative"] + m["false_positive"],
                                 roc_auc=round(m["roc_auc"], 4),
                                 average_precision=round(m["average_precision"], 4),
                                 recall_at_0_5=round(m["recall"], 4),
                                 precision_at_0_5=round(m["precision"], 4)))
                summary[f"{label} | {name} | {cells}"] = result.to_dict()
                if code == "All" and (cells == "all" or label == USGS):
                    tag = stem.get(label, "usgs_likelihood")
                    save(plot_comparison(grid, result, points=points, marker_size=5),
                         out / f"{tag}_vs_initiations.png")
    print("  all 12 Landlab AUCs match the earlier scoring")

    # AUC chart: all four models on the cells USGS assessed -------------------
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    x, w = np.arange(len(MECHANISMS)), 0.2
    for i, label in enumerate(models):
        ax.bar(x + (i - 1.5) * w, [common_auc[(label, c)] for c in MECHANISMS], w,
               color=COLOURS[label], edgecolor="black", linewidth=0.6, label=label)
    ax.axhline(0.5, ls="--", color="0.35", lw=1.2)
    ax.text(len(MECHANISMS) - 0.55, 0.505, "chance", ha="right", va="bottom", color="0.3")
    ax.set_xticks(x)
    ax.set_xticklabels(list(MECHANISMS.values()))
    ax.set_ylim(0.3, 0.9)
    ax.set_ylabel("AUC")
    ax.set_title("Model skill against mapped initiations, on the cells USGS assessed")
    ax.legend(loc="upper left", ncol=2, framealpha=0.95)
    ax.grid(axis="y", lw=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    save(fig, out / "auc_by_mechanism.png")

    # 3 -- MWR vs dDEM ----------------------------------------------------------
    mwr = raster_to_node_field(grid, crs, args.data / "MWR_final_results.asc/MWR.tif")
    ddem = raster_to_node_field(
        grid, crs,
        pio / "main_pipeline_run/corrections/pioneer/final_dem_products/dod_corrected.tif",
        resampling="average",
    )
    observed_erosion = np.where(np.isfinite(ddem), (ddem < -0.5).astype(float), np.nan)
    result = compare_fields_on_grid(
        grid, np.clip(-mwr, 0, None), observed_erosion,
        FieldSpec(name="MWR erosion depth", kind="continuous", units="m", threshold=0.5),
        FieldSpec(name="dDEM erosion (10 m mean < -0.5 m)", kind="binary"),
    )
    m = result.metrics
    rows.append(dict(model="Landlab MWR (erosion depth)", evidence="dDEM erosion", cells="survey",
                     n_positive=m["true_positive"] + m["false_negative"],
                     n_negative=m["true_negative"] + m["false_positive"],
                     roc_auc=round(m["roc_auc"], 4), average_precision=round(m["average_precision"], 4),
                     recall_at_0_5=round(m["recall"], 4), precision_at_0_5=round(m["precision"], 4)))
    summary["MWR erosion vs dDEM erosion"] = result.to_dict()
    save(plot_comparison(grid, result), out / "mwr_erosion.png")
    print(f"  MWR erosion vs dDEM erosion: AUC {m['roc_auc']:.3f}, recall {m['recall']:.2f}, "
          f"precision {m['precision']:.2f}")

    change = dict(kind="categorical", classes=CHANGE)
    result = compare_fields_on_grid(grid, classify(mwr), classify(ddem),
                                    FieldSpec(name="Landlab MWR", **change),
                                    FieldSpec(name="dDEM, 10 m mean", **change))
    summary["MWR vs dDEM classes"] = result.to_dict()
    save(plot_comparison(grid, result, class_colours=CHANGE_COLOURS), out / "mwr_vs_ddem_classes.png")
    print(f"  MWR vs dDEM classes: overall agreement {result.metrics['overall_accuracy']:.3f}")

    with open(out / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"outputs: {out}")


if __name__ == "__main__":
    main()
