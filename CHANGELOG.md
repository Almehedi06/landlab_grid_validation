# Changelog

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
