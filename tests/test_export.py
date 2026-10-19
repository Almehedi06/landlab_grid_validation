"""Writing results back to files: GeoTIFFs that land in the right place, and the agreement map."""

import numpy as np
import pytest
from landlab import RasterModelGrid

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402

from landlab_grid_validation import (  # noqa: E402
    FieldSpec,
    compare_fields_on_grid,
    grid_from_raster,
    raster_to_node_field,
    write_field_geotiff,
)

CRS = "EPSG:32610"


def test_a_written_geotiff_reads_back_onto_the_same_cells(tmp_path):
    """Write a field, read it back with the package's own reader: every value in its cell."""
    src = tmp_path / "src.tif"
    arr = np.arange(20, dtype="float32").reshape(4, 5)     # row 0 is north, as GeoTIFFs are
    with rasterio.open(src, "w", driver="GTiff", height=4, width=5, count=1, dtype="float32",
                       crs=CRS, transform=from_origin(1000.0, 2000.0, 10.0, 10.0)) as dst:
        dst.write(arr, 1)
    grid, crs = grid_from_raster(src)
    values = raster_to_node_field(grid, crs, src)
    values[3] = np.nan                                     # no data must survive the round trip

    out = write_field_geotiff(grid, crs, tmp_path / "out.tif", values)
    with rasterio.open(out) as written, rasterio.open(src) as original:
        assert written.transform == original.transform
        assert written.crs == original.crs
    again = raster_to_node_field(grid, crs, out)
    np.testing.assert_array_equal(np.isnan(again), np.isnan(values))
    np.testing.assert_allclose(again[np.isfinite(again)], values[np.isfinite(values)])


def test_writer_rejects_the_wrong_length(tmp_path):
    grid = RasterModelGrid((3, 3))
    with pytest.raises(ValueError, match="9 nodes"):
        write_field_geotiff(grid, CRS, tmp_path / "bad.tif", np.zeros(4))


def test_agreement_codes_each_outcome_once():
    grid = RasterModelGrid((2, 2))
    predicted = np.array([0.0, 0.9, 0.1, 0.8])             # no, yes, no, yes at threshold 0.5
    observed = np.array([0.0, 0.0, 1.0, 1.0])
    result = compare_fields_on_grid(
        grid, predicted, observed,
        FieldSpec(name="p", kind="continuous", units="probability", threshold=0.5),
        FieldSpec(name="seen", kind="binary"),
    )
    # 0 correct no, 1 false alarm, 2 miss, 3 hit
    np.testing.assert_array_equal(result.agreement(), [0, 1, 2, 3])


def test_agreement_is_nan_outside_the_compared_cells_and_none_for_two_numbers():
    grid = RasterModelGrid((2, 2))
    spec = FieldSpec(name="p", kind="continuous", units="probability", threshold=0.5)
    result = compare_fields_on_grid(grid, np.array([0.2, 0.9, 0.4, 0.7]),
                                    np.array([0.0, 1.0, np.nan, 1.0]),
                                    spec, FieldSpec(name="seen", kind="binary"))
    assert np.isnan(result.agreement()[2])

    both_numbers = compare_fields_on_grid(grid, np.ones(4), np.ones(4),
                                          FieldSpec(name="a", kind="continuous", units="m"),
                                          FieldSpec(name="b", kind="continuous", units="m"))
    assert both_numbers.agreement() is None
