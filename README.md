# landlab-grid-validation

Compare modelled and observed landslide or mass-wasting fields on one Landlab
grid, whatever format the observations come in, and map where they agree.

```text
raster / shapefile / array  ->  node array on the grid  ->  FieldSpec  ->  compare  ->  plot
```

## Install

```bash
pip install -e ".[io,plot]"        # io: rasterio + geopandas readers; plot: matplotlib
```

The core (`FieldSpec`, `compare_fields_on_grid`) needs only numpy and Landlab.

## Example

```python
import geopandas as gpd
from landlab_grid_validation import (FieldSpec, compare_fields_on_grid, grid_from_raster,
                                     plot_comparison, raster_to_node_field, vector_to_node_field)

grid, crs = grid_from_raster("P_Landslide.tif")               # one node per raster cell
predicted = raster_to_node_field(grid, crs, "P_Landslide.tif")
points = gpd.read_file("initiation_points.shp")
observed = vector_to_node_field(grid, crs, points, tolerance_cells=1)   # 3 x 3 blocks

result = compare_fields_on_grid(
    grid, predicted, observed,
    FieldSpec(name="Landslide", kind="continuous", units="probability", threshold=0.5),
    FieldSpec(name="mapped initiations", kind="binary"),
)
print(result.metrics["roc_auc"])
plot_comparison(grid, result, points=points.to_crs(crs)).savefig("landslide.png")
```

## What goes in

| source | reader | notes |
|---|---|---|
| GeoTIFF, ESRI ASCII, any GDAL raster | `raster_to_node_field` | read as-is if already on the grid; otherwise you must pick `resampling` |
| points, lines, polygons (any geopandas format) | `vector_to_node_field` | a yes/no map, or class codes from a column; optional tolerance in cells |
| NumPy array | pass it directly | 1D in Landlab node order, or 2D with `origin=` |
| Landlab field | pass its name | |

`grid_from_raster` builds the grid from any raster. For an existing grid, pass
its CRS to the readers: a Landlab grid carries none, so the readers never guess.

## Kinds of field and what is computed

Each field gets a `FieldSpec`: `continuous` (probability, elevation change,
depth), `binary` (yes/no) or `categorical` (classes such as stable / erosion /
deposition).

| predicted vs observed | metrics | third map |
|---|---|---|
| continuous vs yes/no | confusion counts at `threshold`, AUC, average precision, Brier (probabilities) | hit / miss / false alarm / correct no |
| yes/no vs yes/no | confusion counts, recall, precision, CSI, F1 | hit / miss / false alarm / correct no |
| continuous vs continuous | bias, MAE, RMSE, Pearson r | predicted − observed |
| classes vs classes | confusion matrix, per-class recall and precision | class agreement |

Scores must increase towards "yes": for erosion, pass depth or `-dz`.

## Rules that stop silent errors

- **2D arrays need `origin`.** `"upper"` when row 0 is north (GeoTIFF,
  rasterio), `"lower"` when row 0 is south (Landlab). The readers return node
  order already, so this only matters for arrays you pass yourself.
- **No default threshold.** Turning a continuous field into yes/no needs
  `FieldSpec.threshold` or `threshold=`.
- **Values are checked before scoring.** Probabilities must lie in [0, 1],
  binary fields may hold only 0 and 1, categorical fields only their declared
  codes. Nothing is clipped.
- **Flags that drift** (9999.137 in a model output) are caught by
  `FieldSpec.valid_range`; the count removed is reported in
  `result.masked_out_of_range`.
- **Resampling is always your choice**: `nearest`/`mode` for classes and
  yes/no maps, `average`/`bilinear` for continuous values.
- **Features outside the grid are reported**, never dropped silently.
- `mask=` restricts scoring to chosen cells, for example evidence plus
  pseudo-absence blocks.

## Pioneer example

`examples/pioneer.py` scores the three Landlab probability layers for the
Pioneer Fire against all 89 mapped initiation points, and Landlab MWR against
the dDEM. It stops unless the AUCs equal the earlier scoring (0.573818,
0.652238, 0.664850), so it doubles as an end-to-end check.

```bash
python examples/pioneer.py --data /path/to/Downloads
```

## Tests

```bash
python -m pytest -q
```
