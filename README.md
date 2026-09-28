# landlab-grid-validation

Prototype utility for comparing modelled and observed landslide or mass-wasting
fields on a common Landlab grid.

Core idea:

```text
any input format -> Landlab node field -> FieldSpec -> compare_fields_on_grid()
```

The core function does not care whether data originally came from ASC, GeoTIFF,
shapefile, points, a NumPy array, or a Landlab field. Format-specific readers
should stay as adapters around this core.

## Supported Field Kinds

- `continuous`: probability, elevation change, erosion depth, deposition depth,
  factor of safety.
- `binary`: landslide/no landslide, erosion/no erosion, deposition/no deposition.
- `categorical`: stable/erosion/deposition, source/runout/deposition, or
  severity classes.

## Example

```python
from landlab import RasterModelGrid
from landlab_grid_validation import FieldSpec, compare_fields_on_grid

grid = RasterModelGrid((100, 100), xy_spacing=10.0)

result = compare_fields_on_grid(
    grid,
    predicted="landslide__probability_of_failure",
    observed="mapped_landslide_inventory",
    predicted_spec=FieldSpec(
        name="landslide__probability_of_failure",
        kind="continuous",
        variable="landslide_probability",
        units="probability",
        threshold=0.5,
    ),
    observed_spec=FieldSpec(
        name="mapped_landslide_inventory",
        kind="binary",
        variable="landslide_presence",
        positive_class="landslide",
    ),
)

print(result.metrics)
```

## Design Rules

- Keep the core Landlab-facing comparison dependency-light.
- Treat file formats as adapters, not as the scientific abstraction.
- Require explicit metadata for meaning: kind, variable, units, positive class,
  class labels, threshold and nodata.
- Use a common valid-data mask before computing metrics.
- Do not infer geomorphic meaning from numeric values alone.

