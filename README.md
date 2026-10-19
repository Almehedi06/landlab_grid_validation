# landlab-grid-validation

Compare modelled and observed landslide or mass-wasting fields on one Landlab
grid, whatever format the observations come in, and map where they agree.

```text
raster / shapefile / array  ->  node field on the grid  ->  FieldSpec  ->  compare  ->  plot
```

## Getting started

```bash
git clone https://github.com/Almehedi06/landlab_grid_validation
cd landlab_grid_validation
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[io,plot,notebook]"
```

If `python -m venv` fails on Ubuntu, run `sudo apt install python3-venv` first, or
use a conda Python.

Open `examples/validate_pioneer.ipynb`, choose the `.venv` kernel, set `PIONEER`
in the first cell to your data folder, and Run All. It scores the Landlab layers
for the Pioneer Fire at Stehekin against 89 mapped debris-flow initiation points
and the dDEM, and its results are saved in it. In VS Code, open the repository
folder itself so the `.venv` kernel is listed.

`examples/export_layers.ipynb` runs that notebook, then writes its layers to files
for QGIS: the yes/no and dDEM fields, hit/miss maps, the channel and slope masks,
and absence points away from ridges and channels. `validate_pioneer.ipynb` itself
writes nothing.

`examples/landslide_notebook.ipynb` is a shorter example: one 30 m landslide
probability field against the mapped landslide points.

The Pioneer data are not in the repository. The initiation points are in the USGS
data release (doi:10.5066/P1GNNSZU); the Landlab layers and the dDEM are available
on request.

To use the package in your own code without cloning:
`pip install "landlab-grid-validation[io,plot] @ git+https://github.com/Almehedi06/landlab_grid_validation"`.
Python 3.10 to 3.13. The core (`FieldSpec`, `compare_fields_on_grid`) needs only
numpy and Landlab; `io` adds the rasterio and geopandas readers, `plot` adds
matplotlib.

## Using it in your own code

The notebook's calls, in short:

```python
from landlab_grid_validation import (FieldSpec, compare_fields_on_grid, grid_from_raster,
                                     plot_comparison, raster_to_node_field, vector_to_node_field,
                                     write_field_geotiff)

grid, crs = grid_from_raster("P_Landslide.asc", crs="EPSG:6339")      # nodes at cell centres
grid.add_field("P_Landslide", raster_to_node_field(grid, crs, "P_Landslide.asc",
                                                   raster_crs="EPSG:6339"), at="node")
observed = vector_to_node_field(grid, crs, "initiation_points.shp", tolerance_cells=1)

result = compare_fields_on_grid(
    grid, "P_Landslide", observed,
    FieldSpec(name="Landslide", kind="continuous", units="probability", threshold=0.5),
    FieldSpec(name="mapped initiations", kind="binary"),
)
print(result.metrics["roc_auc"])
plot_comparison(grid, result)

# keep the maps as files for QGIS
write_field_geotiff(grid, crs, "observed.tif", observed)              # 1 = yes, 0 = no
write_field_geotiff(grid, crs, "agreement.tif", result.agreement())   # 0 correct no, 1 false alarm, 2 miss, 3 hit
```

## What goes in

| source | reader | notes |
|---|---|---|
| GeoTIFF, ESRI ASCII, any GDAL raster | `raster_to_node_field` | read as-is if already on the grid; otherwise you choose `resampling` |
| points, lines, polygons (any geopandas format) | `vector_to_node_field` | a yes/no map, or class codes from a column; optional tolerance in cells |
| NumPy array | pass it directly | 1D in Landlab node order, or 2D with `origin=` |
| Landlab field | pass its name | |

A Landlab grid carries no coordinate system, so the readers take its CRS
explicitly and never guess.

## Kinds of field and what is computed

Each field gets a `FieldSpec`: `continuous` (probability, elevation change,
depth), `binary` (yes/no) or `categorical` (e.g. stable / erosion / deposition).

| predicted vs observed | metrics | third map |
|---|---|---|
| continuous vs yes/no | confusion counts at `threshold`, AUC with a confidence interval, average precision, Brier (probabilities) | hit / miss / false alarm / correct no |
| yes/no vs yes/no | confusion counts, recall, precision, CSI, F1 | hit / miss / false alarm / correct no |
| continuous vs continuous | bias, MAE, RMSE, Pearson r | predicted − observed |
| classes vs classes | confusion matrix, per-class recall and precision | class agreement |

Scores must increase towards "yes": for erosion, pass depth or `-dz`, and set
`sign_convention="positive_is_loss"` so maps keep loss red.

`tolerance_cells=k` in `compare_fields_on_grid` adds recall and precision that
forgive offsets of up to k cells, for yes/no comparisons.

## Rules that stop silent errors

- **Build grids with `grid_from_raster`.** `landlab.io.esri_ascii.load` puts
  nodes on cell corners for files with `XLLCORNER`, half a cell off; the
  readers detect such a grid and refuse it.
- **2D arrays need `origin`**: `"upper"` when row 0 is north (GeoTIFF,
  rasterio), `"lower"` when row 0 is south (Landlab).
- **No default threshold.** Turning a continuous field into yes/no needs one.
- **Values are checked before scoring**: probabilities in [0, 1], binary
  fields 0 and 1 only, categorical fields their declared codes. Nothing is clipped.
- **Drifting flags** (9999.137 in a model output) are removed by
  `FieldSpec.valid_range` and counted in `result.masked_out_of_range`.
- **Resampling is always your choice**, and features outside the grid are
  reported, never dropped silently.
- **The AUC comes with a confidence interval** (`roc_auc_ci_low` and
  `roc_auc_ci_high`, by the DeLong method). With few observed points an AUC of
  0.66 may not differ from 0.57, and the interval says so. A perfect separation
  reports `nan` rather than a zero-width interval. The interval assumes
  independent cells, so with clustered points or `tolerance_cells` it is a
  lower bound on the real uncertainty.

## Tests

```bash
pip install -e ".[test]"
python -m pytest -q
```

They run from a clone, installed or not. CI runs them on Linux, macOS and
Windows for Python 3.10 to 3.13, with `ruff check .` for lint.

## Citation and license

Cite with the metadata in `CITATION.cff`. MIT license, see `LICENSE`.
Changes are recorded in `CHANGELOG.md`.
