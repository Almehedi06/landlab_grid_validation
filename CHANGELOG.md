# Changelog

## 1.2.0 - 2026-10-19

### Added

- `plot_comparison(..., points=gdf, colour_by="column")`: the observations drawn
  in one colour per value of that column (for example failure type), with the
  count of each in a legend on the observed map. Missing values are
  "unlabelled"; a column that does not exist is an error.
- Feature-level scoring for point and polygon inventories:
  `vector_to_feature_cells(grid, crs, features, tolerance_cells=k)` gives each
  feature's cells, `feature_hits(result.predicted_yes, cells)` says whether
  anything was flagged at each one, and `plot_comparison(..., feature_hits=hits)`
  draws the flagged cells with each feature marked hit or miss, in place of the
  cell-by-cell agreement map, which under-reads points.
- `evaluate_features(grid, crs, layers, features, ...)`: AUC with a 95% interval
  (resampling the features) for any number of layers, for all features and per
  nested group (`group_by` columns), with the absence cells drawn by `Absences`
  (keep and exclude masks, distance from the features, all or n at random,
  stratified by a class raster). `Evaluation.write` saves metrics, ROC curves,
  `roc.png`, the absences and a run record. `roc_curve`, `bootstrap_auc_interval`
  and `feature_groups` are the pieces.
- Rules as plain dicts: `rule_mask` and `mask_from_rules` turn rasters, a DEM's
  slope (`slope_of`) and polygons into cell masks (`min`, `max`, `abs_min`,
  `abs_max`, `values`, `finite`, `inside`, `buffer_cells`); `sample_absences`
  draws the absence cells; `slope_degrees` is the slope.
- `python -m landlab_grid_validation config.yaml`: the same evaluation from a
  YAML config, for any site; unknown keys are errors.
- `plot_roc(curves)`: ROC panels, built without pyplot so a notebook's backend is
  never changed.
- `evaluate_features` also reports the cost at a cutoff (`thresholds`, default 0.5,
  and the Youden cutoff: confusion counts, precision, recall, f1, csi, precision as a
  multiple of the base rate, share of the domain flagged), `tpr_at_fpr_<level>`,
  `average_precision`, `brier`, `positives_at_zero` and `domain_above_zero`;
  `threshold_scores` and `youden_threshold` are the pieces; `Evaluation.summary()`
  prints the short version: AUC (95%), the optimal cutoff and the precision there.
- `plot_sampling(...)` and `Evaluation.plot_sampling()`: the map of the domain,
  positive, absence and excluded cells, with counts; `Evaluation.write` saves it
  as `sampling_map.png`.
- `chance_hit_rate(grid, predicted_yes, tolerance_cells)`: the hit rate a random
  location gets under the feature rule; `plot_comparison(..., chance_hit_rate=)`
  puts it in the title.
- `compare_fields_on_grid(..., tolerance_cells=k)` for two continuous fields:
  `mae_within_tolerance` and `rmse_within_tolerance`, matching each observed cell
  with the closest predicted value in its block.
- A continuous field against classes gets a box plot per class as its third map.
- `io` extra now includes scipy and pyyaml.

## 1.1.0 - 2026-10-18

### Added

- `write_field_geotiff(grid, crs, path, field)`: any node field to a GeoTIFF on
  the grid, north up, NaN as no data. The counterpart of `write_field_ascii`.
- `ComparisonResult.agreement()`: the map of 0 correct no, 1 false alarm,
  2 miss, 3 hit. `plot_comparison` and `export_layers.ipynb` both use it, so
  the codes are defined in one place.


- `examples/validate_pioneer.ipynb` is shorter: no setup cell, a plain AUC
  table, and one line of reading after each result. It reads `probability_fields/`
  and `obs_data/` in the Pioneer data folder.
- Removed the tutorial notebook, `examples/build_tutorial.py` and
  `run_tutorial.sh`. The Pioneer notebook is the worked example.
- `examples/export_layers.ipynb`: writes the validation layers, the channel and
  slope masks and the G 5-150 absence points to files. Kept apart so running
  the validation notebook never overwrites data.
- README: one "Getting started" block (clone, `.venv`, install, open the
  notebook) and where the Pioneer data come from.

## 1.0.0 - 2026-10-05

First stable release. The API is settled: `FieldSpec`, `compare_fields_on_grid`,
the readers, and `plot_comparison`.

### Added

- Confidence intervals for the AUC, by the DeLong method, reported with every
  score-against-yes/no comparison as `roc_auc_ci_low` and `roc_auc_ci_high`.
  A perfect separation or all-tied scores report `nan` rather than a
  zero-width interval that would claim false certainty.
- `roc_auc` and `roc_auc_ci` are importable from the package directly.
- `examples/tutorial.ipynb`: a run-anywhere tutorial that makes its own example
  data, with every output saved. Built by `examples/build_tutorial.py`.
- MIT license, `CITATION.cff` and this changelog.
- CI on Linux, macOS and Windows for Python 3.10 to 3.13, plus `ruff`.

### Fixed

- Tests run from a clone whether or not the package is installed. They
  previously failed to import unless `pip install -e .` had been run first.

## 0.3.0 - 2026-09-28

- Half-cell guard, raster tolerance, and one Pioneer notebook.
- Readers for rasters and vectors, value checks, map plots.
- `FieldSpec` and `compare_fields_on_grid`.
