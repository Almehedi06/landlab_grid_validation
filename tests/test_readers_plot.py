"""Readers (raster, vector) and the comparison plot."""

import numpy as np
import pytest
from landlab import RasterModelGrid

rasterio = pytest.importorskip("rasterio")
gpd = pytest.importorskip("geopandas")
from rasterio.transform import from_origin  # noqa: E402
from shapely.geometry import LineString, Point  # noqa: E402

from landlab_grid_validation import (  # noqa: E402
    FieldSpec,
    compare_fields_on_grid,
    grid_from_raster,
    plot_comparison,
    raster_to_node_field,
    vector_to_node_field,
)

CRS = "EPSG:32610"


def write_tif(path, arr, transform, crs=CRS, nodata=None):
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
                       count=1, dtype="float64", crs=crs, transform=transform,
                       nodata=nodata) as dst:
        dst.write(arr, 1)
    return path


def test_raster_round_trip_keeps_north_up(tmp_path):
    arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])      # row 0 = north
    path = write_tif(tmp_path / "a.tif", arr, from_origin(500000, 5000020, 10, 10))
    grid, crs = grid_from_raster(path)
    assert (grid.x_of_node[0], grid.y_of_node[0]) == (500005.0, 5000005.0)
    values = raster_to_node_field(grid, crs, path)
    north = grid.y_of_node == grid.y_of_node.max()
    assert list(values[north]) == [1.0, 2.0, 3.0]


def test_raster_nodata_becomes_nan(tmp_path):
    arr = np.array([[1.0, -9999.0, 3.0], [4.0, 5.0, 6.0]])
    path = write_tif(tmp_path / "a.tif", arr, from_origin(0, 20, 10, 10), nodata=-9999.0)
    grid, crs = grid_from_raster(path)
    assert np.isnan(raster_to_node_field(grid, crs, path)).sum() == 1


def test_raster_on_another_grid_needs_a_resampling_choice(tmp_path):
    grid, crs = grid_from_raster(write_tif(tmp_path / "g.tif", np.zeros((2, 3)),
                                           from_origin(0, 20, 10, 10)))
    fine = write_tif(tmp_path / "fine.tif", np.ones((4, 6)), from_origin(0, 20, 5, 5))
    with pytest.raises(ValueError, match="resampling"):
        raster_to_node_field(grid, crs, fine)
    assert np.allclose(raster_to_node_field(grid, crs, fine, resampling="average"), 1.0)


def test_raster_without_crs_must_be_given_one(tmp_path):
    path = write_tif(tmp_path / "a.tif", np.zeros((2, 3)), from_origin(0, 20, 10, 10), crs=None)
    with pytest.raises(ValueError, match="declares no CRS"):
        grid_from_raster(path)
    grid, crs = grid_from_raster(path, crs=CRS)
    assert crs.to_epsg() == 32610


def test_grid_built_on_cell_corners_is_refused(tmp_path):
    from landlab.io import esri_ascii
    path = tmp_path / "field.asc"
    path.write_text("ncols 3\nnrows 2\nxllcorner 500000\nyllcorner 5000000\ncellsize 10\n"
                    "NODATA_value -9999\n1 2 3\n4 5 6\n")
    with open(path) as f:
        corner_grid = esri_ascii.load(f, name="field")        # nodes on cell corners
    with pytest.raises(ValueError, match="half a cell"):
        raster_to_node_field(corner_grid, CRS, path, raster_crs=CRS, resampling="nearest")
    grid, crs = grid_from_raster(path, crs=CRS)                # nodes on cell centres
    assert list(raster_to_node_field(grid, crs, path, raster_crs=CRS)) == [4.0, 5.0, 6.0, 1.0, 2.0, 3.0]


def test_points_with_a_one_cell_tolerance_burn_3x3_blocks():
    grid = RasterModelGrid((5, 5), xy_spacing=10.0, xy_of_lower_left=(5.0, 5.0))
    pts = gpd.GeoDataFrame(geometry=[Point(25, 25)], crs=CRS)
    burned = vector_to_node_field(grid, CRS, pts, tolerance_cells=1).reshape(5, 5)
    assert burned.sum() == 9 and burned[1:4, 1:4].all()


def test_features_outside_the_grid_are_reported():
    grid = RasterModelGrid((3, 3), xy_spacing=10.0, xy_of_lower_left=(5.0, 5.0))
    pts = gpd.GeoDataFrame(geometry=[Point(15, 15), Point(900, 900)], crs=CRS)
    with pytest.warns(UserWarning, match="1 of 2 features lie outside"):
        burned = vector_to_node_field(grid, CRS, pts)
    assert burned.sum() == 1


def test_text_classes_are_burned_through_codes():
    grid = RasterModelGrid((3, 3), xy_spacing=10.0, xy_of_lower_left=(5.0, 5.0))
    gdf = gpd.GeoDataFrame({"kind": ["LS", "Runoff"]},
                           geometry=[Point(5, 5), LineString([(5, 25), (25, 25)])], crs=CRS)
    with pytest.raises(ValueError, match="not numeric"):
        vector_to_node_field(grid, CRS, gdf, value="kind")
    burned = vector_to_node_field(grid, CRS, gdf, value="kind", codes={"LS": 1, "Runoff": 2})
    assert burned[0] == 1 and (burned.reshape(3, 3)[2] == 2).all()


def test_colour_scales_follow_sign_and_ignore_zeros():
    from landlab_grid_validation.plot import _continuous_style
    depth = np.r_[np.zeros(95), np.linspace(0.5, 1.5, 5)]          # sparse erosion depth
    cmap, norm = _continuous_style(depth, "m")
    assert norm.vmin == 0.0 and norm.vmax > 1.4
    signed = np.r_[-2.0, 0.0, 3.0]
    assert _continuous_style(signed, "m")[0] == "RdBu"
    assert _continuous_style(signed, "m", "positive_is_loss")[0] == "RdBu_r"


@pytest.mark.parametrize("kind", ["probability", "change", "classes"])
def test_plot_comparison_draws_three_maps(kind):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    grid = RasterModelGrid((4, 4), xy_spacing=10.0)
    rng = np.random.default_rng(1)
    if kind == "probability":
        pred, obs = rng.random(16), (rng.random(16) > 0.5).astype(float)
        ps = FieldSpec(name="P", kind="continuous", units="probability", threshold=0.5)
        os_ = FieldSpec(name="mapped", kind="binary")
    elif kind == "change":
        pred, obs = rng.normal(size=16), rng.normal(size=16)
        ps = FieldSpec(name="model dz", kind="continuous", units="m")
        os_ = FieldSpec(name="dDEM", kind="continuous", units="m")
    else:
        pred, obs = rng.integers(0, 3, 16).astype(float), rng.integers(0, 3, 16).astype(float)
        ps = os_ = FieldSpec(name="c", kind="categorical",
                             classes={0: "stable", 1: "erosion", 2: "deposition"})
    result = compare_fields_on_grid(grid, pred, obs, ps, os_)
    fig = plot_comparison(grid, result, points=(np.array([15.0]), np.array([15.0])))
    assert len(fig.axes) == 6                                  # three maps, three colour bars
    assert "cells compared" in fig.get_suptitle()
