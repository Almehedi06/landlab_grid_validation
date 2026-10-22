"""Score gridded layers against mapped features from a YAML config.

    python -m landlab_grid_validation config.yaml [--overwrite]

Paths in the config are relative to the config's own folder. Every key must be
known; the ones marked optional have the defaults shown.

    run_name: baseline
    output_dir: evaluation              # results go to output_dir/run_name
    crs: EPSG:6339                      # optional; for files that declare no CRS
    predicted:                          # the layers to score; the first one sets the grid
      P_Landslide: P_Landslide.asc
      P_Combined_H: P_Combined_H.asc
    observed:
      features: initiation_points.shp   # points, lines or polygons
      tolerance_cells: 1                # optional; 1 = a 3 x 3 block per point
      group_by: [FailureTyp]            # optional; scores for all, then per value, nested
    domain:                             # optional; cells both classes may come from (all rules hold)
      - {slope_of: dem.asc, min: 17}
      - {raster: ddem.tif, finite: true, resampling: average}
    negatives:                          # optional
      rules: []                         # extra rules for absence cells only (all hold)
      exclude:                          # absence cells matching any of these are dropped
        - {raster: ddem.tif, abs_min: 0.5, buffer_cells: 1, resampling: average}
      min_distance_m: 50                # from any feature
      n: all                            # or a number, drawn at random
      seed: 42
      stratify: null                    # or {raster: classes.tif, resampling: nearest}
    bootstrap: 1000                     # optional; resamples of the features for the 95% interval
    seed: 42                            # optional
    thresholds: [0.5]                   # optional; cutoffs for precision, recall, f1, csi, area flagged
    optimal_threshold: true             # optional; also score at the Youden cutoff (max TPR - FPR)
    fpr_levels: [0.05, 0.10]            # optional; report TPR at these false-positive rates

Rules are described in ``landlab_grid_validation.rules``; the scores in
``evaluate_features``. The run writes metrics.csv, roc.png, sampling_map.png,
roc_curves.csv, negatives.gpkg, settings.json and run_info.json.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .evaluate import evaluate_features
from .readers import grid_from_raster, raster_to_node_field
from .rules import Absences, mask_from_rules, rule_mask

SCHEMA = {
    "run_name": None, "output_dir": None, "crs": None, "predicted": None,
    "observed": {"features", "tolerance_cells", "group_by"},
    "domain": None,
    "negatives": {"rules", "exclude", "min_distance_m", "n", "seed", "stratify"},
    "bootstrap": None, "seed": None, "thresholds": None, "optimal_threshold": None, "fpr_levels": None,
}
REQUIRED = ("run_name", "output_dir", "predicted")


def load_config(path: Path) -> dict:
    """Read the YAML config and check every key; report all problems at once."""
    import yaml

    cfg = yaml.safe_load(Path(path).read_text()) or {}
    problems = [f"unknown key: {k}" for k in cfg if k not in SCHEMA]
    problems += [f"missing key: {k}" for k in REQUIRED if k not in cfg]
    if "features" not in (cfg.get("observed") or {}):
        problems.append("missing key: observed.features")
    for section, keys in SCHEMA.items():
        if keys and isinstance(cfg.get(section), dict):
            problems += [f"unknown key: {section}.{k}" for k in cfg[section] if k not in keys]
    if problems:
        raise SystemExit(f"{path}:\n  " + "\n  ".join(problems))
    cfg.setdefault("crs", None)
    cfg["observed"] = {"tolerance_cells": 1, "group_by": None, **cfg["observed"]}
    cfg.setdefault("domain", [])
    cfg["negatives"] = {"rules": [], "exclude": [], "min_distance_m": 0.0, "n": "all", "seed": 0,
                        "stratify": None, **(cfg.get("negatives") or {})}
    cfg.setdefault("bootstrap", 1000)
    cfg.setdefault("seed", 42)
    cfg.setdefault("thresholds", [0.5])
    cfg.setdefault("optimal_threshold", True)
    cfg.setdefault("fpr_levels", [0.05, 0.10])
    return cfg


def run(config_path, *, overwrite=False) -> Path:
    config_path = Path(config_path)
    cfg = load_config(config_path)
    base = config_path.resolve().parent

    def resolve(p):
        return Path(p) if Path(p).is_absolute() else base / p

    out = resolve(cfg["output_dir"]) / cfg["run_name"]
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise SystemExit(f"{out} already holds a run; choose a new run_name or pass --overwrite")

    crs = cfg["crs"]
    names = list(cfg["predicted"])
    grid, grid_crs = grid_from_raster(resolve(cfg["predicted"][names[0]]), crs=crs)
    layers = {name: raster_to_node_field(grid, grid_crs, resolve(p), raster_crs=crs)
              for name, p in cfg["predicted"].items()}
    neg = cfg["negatives"]
    domain = mask_from_rules(grid, grid_crs, cfg["domain"], base_dir=base)
    absences = Absences(
        keep=mask_from_rules(grid, grid_crs, neg["rules"], base_dir=base) if neg["rules"] else None,
        exclude=mask_from_rules(grid, grid_crs, neg["exclude"], how="any", base_dir=base) if neg["exclude"] else None,
        min_distance_m=float(neg["min_distance_m"]), n=neg["n"], seed=int(neg["seed"]),
        stratify=raster_to_node_field(grid, grid_crs, resolve(neg["stratify"]["raster"]),
                                      resampling=neg["stratify"].get("resampling"), raster_crs=crs)
        if neg["stratify"] else None,
    )
    obs = cfg["observed"]
    ev = evaluate_features(grid, grid_crs, layers, resolve(obs["features"]),
                           tolerance_cells=int(obs["tolerance_cells"]), group_by=obs["group_by"],
                           domain=domain, absences=absences, bootstrap=int(cfg["bootstrap"]), seed=int(cfg["seed"]),
                           thresholds=tuple(cfg["thresholds"]), optimal_threshold=bool(cfg["optimal_threshold"]),
                           fpr_levels=tuple(cfg["fpr_levels"]))
    inputs = [resolve(p) for p in cfg["predicted"].values()] + [resolve(obs["features"])]
    inputs += [resolve(r[k]) for r in [*cfg["domain"], *neg["rules"], *neg["exclude"]]
               for k in ("raster", "slope_of", "vector") if k in r]
    import geopandas as gpd

    ev.write(out, settings=cfg, inputs=inputs, points=gpd.read_file(resolve(obs["features"])).to_crs(grid_crs),
             background=next((raster_to_node_field(grid, grid_crs, resolve(r["slope_of"]), raster_crs=crs)
                              for r in cfg["domain"] if "slope_of" in r), None))
    print(ev.summary())
    print(f"wrote {out}")
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m landlab_grid_validation", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path)
    ap.add_argument("--overwrite", action="store_true", help="replace an existing run folder")
    args = ap.parse_args(argv)
    run(args.config, overwrite=args.overwrite)


if __name__ == "__main__":
    main(sys.argv[1:])

__all__ = ["load_config", "main", "run", "rule_mask"]
