"""The checks that stop silent errors: one test per failure mode."""

import numpy as np
import pytest
from landlab import RasterModelGrid

from landlab_grid_validation import FieldSpec, compare_fields_on_grid, field_to_node_array, write_field_ascii
from landlab_grid_validation.metrics import average_precision, roc_auc

PROB = FieldSpec(name="p", kind="continuous", units="probability", threshold=0.5)
YESNO = FieldSpec(name="o", kind="binary")


def test_2d_array_needs_origin():
    grid = RasterModelGrid((2, 3))
    with pytest.raises(ValueError, match="origin"):
        field_to_node_array(grid, np.zeros((2, 3)))


def test_upper_origin_puts_row_0_on_the_north_edge():
    grid = RasterModelGrid((2, 3))
    top_down = np.array([[1, 2, 3], [4, 5, 6]])          # row 0 = north, as rasterio reads it
    nodes = field_to_node_array(grid, top_down, origin="upper")
    north = grid.y_of_node == grid.y_of_node.max()
    assert list(nodes[north]) == [1, 2, 3]
    assert list(field_to_node_array(grid, top_down, origin="lower")[north]) == [4, 5, 6]


def test_threshold_is_required_to_make_yes_no():
    grid = RasterModelGrid((2, 2))
    dz = FieldSpec(name="dz", kind="continuous", units="m")
    with pytest.raises(ValueError, match="threshold"):
        compare_fields_on_grid(grid, np.array([0.1, 0.9, 0.2, 0.8]), np.array([0, 1, 0, 1]), dz, YESNO)
    result = compare_fields_on_grid(grid, np.array([0.1, 0.9, 0.2, 0.8]), np.array([0, 1, 0, 1]),
                                    dz, YESNO, threshold=0.5)
    assert result.comparison_type == "continuous_vs_binary"
    assert result.metrics["roc_auc"] == 1.0


def test_valid_range_masks_drifting_flags_and_reports_them():
    grid = RasterModelGrid((2, 2))
    spec = FieldSpec(name="mwr", kind="continuous", units="m", nodata=9999, valid_range=(-100, 100))
    obs = FieldSpec(name="ddem", kind="continuous", units="m")
    with pytest.warns(UserWarning, match="1 values outside valid_range"):
        result = compare_fields_on_grid(grid, np.array([0.1, 9999.137, 9999.0, 0.3]),
                                        np.array([0.1, 0.2, 0.3, 0.3]), spec, obs)
    assert result.metrics["n"] == 2
    assert result.masked_out_of_range == {"predicted": 1, "observed": 0}


def test_probability_outside_0_1_is_an_error_not_clipped():
    grid = RasterModelGrid((2, 2))
    with pytest.raises(ValueError, match="outside \\[0, 1\\]"):
        compare_fields_on_grid(grid, np.array([0.2, 1.3, -9999.0, 0.5]), np.array([0, 1, 0, 1]),
                               PROB, YESNO)


def test_binary_field_may_hold_only_0_and_1():
    grid = RasterModelGrid((2, 2))
    with pytest.raises(ValueError, match="expected only 0 and 1"):
        compare_fields_on_grid(grid, np.array([0.2, 0.8, 0.4, 0.6]), np.array([0, 2, 0, 1]),
                               PROB, YESNO)


def test_categorical_values_must_be_declared_classes():
    grid = RasterModelGrid((2, 2))
    spec = FieldSpec(name="c", kind="categorical", classes={0: "stable", 1: "erosion"})
    with pytest.raises(ValueError, match="not among its classes"):
        compare_fields_on_grid(grid, np.array([0, 1, 255, 0]), np.array([0, 1, 1, 0]), spec, spec)


def test_mask_with_nan_excludes_cells():
    grid = RasterModelGrid((2, 2))
    mask = np.array([1.0, np.nan, 1.0, 1.0])
    result = compare_fields_on_grid(grid, np.array([0.9, 0.9, 0.1, 0.8]), np.array([1, 0, 0, 1]),
                                     PROB, YESNO, mask=mask)
    assert result.metrics["n"] == 3


def test_auc_ties_share_their_mean_rank():
    # positive at 1 ties one negative (half credit) and beats the other: (0.5 + 1) / 2
    assert roc_auc(np.array([1.0, 1.0, 0.0]), np.array([1, 0, 0])) == pytest.approx(0.75)


def test_average_precision_treats_ties_as_one_step():
    scores = np.array([0.9, 0.5, 0.5, 0.1])
    y = np.array([1, 0, 1, 0])
    expected = 0.5 * 1.0 + 0.5 * (2 / 3)                   # steps at 0.9 and at the 0.5 tie
    assert average_precision(scores, y) == pytest.approx(expected)
    assert average_precision(scores[[0, 2, 1, 3]], y[[0, 2, 1, 3]]) == pytest.approx(expected)


def test_write_field_ascii_writes_the_given_values_and_leaves_the_grid_alone(tmp_path):
    grid = RasterModelGrid((2, 3))
    grid.add_field("dz", np.zeros(6), at="node")
    path = write_field_ascii(grid, tmp_path / "dz.asc", np.array([0, 1, 2, 3, 4, np.nan]), name="dz")
    values = [float(v) for v in path.read_text().split()[-6:]]
    assert values == [3.0, 4.0, -9999.0, 0.0, 1.0, 2.0]      # north row first, NaN as -9999
    assert not grid.at_node["dz"].any()
