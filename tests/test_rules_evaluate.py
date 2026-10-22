"""Cell rules, absence sampling, grouped evaluation, ROC plot, chance hit rate,
continuous tolerance, and the YAML runner."""

import json

import numpy as np
import pytest
from landlab import RasterModelGrid

rasterio = pytest.importorskip("rasterio")
gpd = pytest.importorskip("geopandas")
from rasterio.transform import from_origin  # noqa: E402
from shapely.geometry import Point, box  # noqa: E402

from landlab_grid_validation import (  # noqa: E402
    Absences,
    FieldSpec,
    chance_hit_rate,
    compare_fields_on_grid,
    evaluate_features,
    feature_groups,
    grid_from_raster,
    mask_from_rules,
    roc_curve,
    rule_mask,
    sample_absences,
    slope_degrees,
    vector_to_feature_cells,
)

CRS = "EPSG:32610"
TRANSFORM = from_origin(0, 60, 10, 10)          # 6 x 6 cells, x 0..60, y 0..60


def write_tif(path, arr, nodata=None):
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                       dtype="float64", crs=CRS, transform=TRANSFORM, nodata=nodata) as dst:
        dst.write(arr, 1)
    return path


@pytest.fixture
def grid(tmp_path):
    return grid_from_raster(write_tif(tmp_path / "g.tif", np.zeros((6, 6))))[0]


def test_rules_keep_what_they_say(tmp_path, grid):
    values = np.arange(36, dtype=float).reshape(6, 6)          # row 0 = north: 0..5, ..., row 5: 30..35
    values[0, 0] = -9999
    path = write_tif(tmp_path / "v.tif", values, nodata=-9999)
    m = rule_mask(grid, CRS, {"raster": path, "min": 10, "max": 20})
    assert m.sum() == 9                                         # 11..19
    assert rule_mask(grid, CRS, {"raster": path, "values": [3, 4, 99]}).sum() == 2
    assert rule_mask(grid, CRS, {"raster": path, "finite": True}).sum() == 35
    assert rule_mask(grid, CRS, {"raster": path, "finite": False}).sum() == 1
    assert rule_mask(grid, CRS, {"raster": path, "abs_min": 34}).sum() == 1      # only 35: exclusive, nodata is NaN
    inside = rule_mask(grid, CRS, {"vector": gpd.GeoDataFrame(geometry=[box(0, 0, 20, 20)], crs=CRS)})
    assert inside.sum() == 4
    assert rule_mask(grid, CRS, {"vector": gpd.GeoDataFrame(geometry=[box(0, 0, 20, 20)], crs=CRS),
                                 "inside": False}).sum() == 32
    grown = rule_mask(grid, CRS, {"raster": path, "values": [14], "buffer_cells": 1})
    assert grown.sum() == 9
    assert np.array_equal(mask_from_rules(grid, CRS, []), np.ones(36, bool))
    assert mask_from_rules(grid, CRS, [], how="any").sum() == 0
    both = mask_from_rules(grid, CRS, [{"raster": path, "min": 10}, {"raster": path, "max": 20}])
    assert np.array_equal(both, m)
    either = mask_from_rules(grid, CRS, [{"raster": path, "max": 2}, {"raster": path, "min": 33}], how="any")
    assert either.sum() == 3


def test_rules_refuse_ambiguity(tmp_path, grid):
    path = write_tif(tmp_path / "v.tif", np.ones((6, 6)))
    with pytest.raises(ValueError, match="unknown keys"):
        rule_mask(grid, CRS, {"raster": path, "minimum": 1})
    with pytest.raises(ValueError, match="say what to keep"):
        rule_mask(grid, CRS, {"raster": path})
    with pytest.raises(ValueError, match="exactly one"):
        rule_mask(grid, CRS, {"raster": path, "slope_of": path, "min": 1})
    with pytest.raises(ValueError, match="takes inside"):
        rule_mask(grid, CRS, {"vector": gpd.GeoDataFrame(geometry=[box(0, 0, 20, 20)], crs=CRS), "min": 1})


def test_slope_rule_uses_degrees(tmp_path, grid):
    z = np.tile(np.arange(6, dtype=float) * 10, (6, 1))        # rises 10 m per 10 m cell: 45 degrees
    assert np.allclose(slope_degrees(grid, z[::-1].reshape(-1))[7], 45.0)
    path = write_tif(tmp_path / "dem.tif", z)
    assert rule_mask(grid, CRS, {"slope_of": path, "min": 40}).sum() == 36
    assert rule_mask(grid, CRS, {"slope_of": path, "min": 50}).sum() == 0


def test_absences_respect_blocks_distance_exclusion_and_count(grid):
    pts = gpd.GeoDataFrame(geometry=[Point(25, 25)], crs=CRS)
    own = vector_to_feature_cells(grid, CRS, pts)
    blocks = vector_to_feature_cells(grid, CRS, pts, tolerance_cells=1)
    domain = np.ones(36, bool)
    assert sample_absences(grid, domain, own, block_cells=blocks).size == 36 - 9
    far = sample_absences(grid, domain, own, Absences(min_distance_m=21), block_cells=blocks)
    assert far.size == 36 - 9 - 4                                # also drops the four cells 20 m from the point
    exclude = np.zeros(36, bool)
    exclude[:6] = True
    assert sample_absences(grid, domain, own, Absences(exclude=exclude), block_cells=blocks).size == 36 - 9 - 6
    picked = sample_absences(grid, domain, own, Absences(n=5, seed=1), block_cells=blocks)
    assert picked.size == 5 and np.array_equal(picked, sample_absences(grid, domain, own, Absences(n=5, seed=1),
                                                                        block_cells=blocks))
    with pytest.raises(ValueError, match="only"):
        sample_absences(grid, domain, own, Absences(n=100), block_cells=blocks)
    with pytest.raises(ValueError, match="no absence cells"):
        sample_absences(grid, np.zeros(36, bool), own)


def test_stratified_absences_follow_the_positives_classes(grid):
    pts = gpd.GeoDataFrame(geometry=[Point(25, 25)], crs=CRS)
    blocks = vector_to_feature_cells(grid, CRS, pts, tolerance_cells=1)
    classes = np.zeros(36)
    classes[grid.x_of_node >= 25] = 1                           # the block has 3 cells of class 0 and 6 of class 1
    picked = sample_absences(grid, np.ones(36, bool), blocks, Absences(n=9, seed=0, stratify=classes))
    assert picked.size == 9 and np.sum(classes[picked] == 1) == 6
    with pytest.raises(ValueError, match="stratify needs"):
        sample_absences(grid, np.ones(36, bool), blocks, Absences(stratify=classes))


def test_feature_groups_nest():
    gdf = gpd.GeoDataFrame({"process": ["df", "df", "ls", None], "type": ["a", "b", "a", "a"]},
                           geometry=[Point(5, 5)] * 4, crs=CRS)
    groups = feature_groups(gdf, ["process", "type"])
    assert [g for g, _ in groups] == ["All", "df", "ls", "unlabelled", "df / a", "df / b", "ls / a", "unlabelled / a"]
    assert groups[1][1].tolist() == [0, 1]
    assert feature_groups(gdf)[0][1].size == 4
    with pytest.raises(ValueError, match="not in the features"):
        feature_groups(gdf, "missing")


def test_roc_curve_area_is_the_rank_auc():
    x, y = roc_curve([0.9, 0.8, 0.4], [0.1, 0.5, 0.4, 0.2])
    from landlab_grid_validation import roc_auc
    from landlab_grid_validation.evaluate import area_under
    assert np.isclose(area_under(x, y), roc_auc(np.r_[0.9, 0.8, 0.4, 0.1, 0.5, 0.4, 0.2], np.r_[1, 1, 1, 0, 0, 0, 0]))


def test_evaluate_features_scores_groups_and_writes_a_run(tmp_path, grid):
    pts = gpd.GeoDataFrame({"kind": ["LS", "Runoff"]}, geometry=[Point(15, 45), Point(45, 15)], crs=CRS)
    layer = np.zeros(36)
    layer[vector_to_feature_cells(grid, CRS, pts[:1], tolerance_cells=1)[0]] = 1.0    # perfect on the LS point
    ev = evaluate_features(grid, CRS, {"P": layer}, pts, tolerance_cells=1, group_by="kind",
                           absences=Absences(min_distance_m=15), bootstrap=50)
    rows = {(r["layer"], r["group"]): r for r in ev.table}
    assert rows[("P", "LS")]["auc"] == 1.0                      # its block scores 1, every absence cell 0
    assert rows[("P", "Runoff")]["auc"] == 0.5                  # its block and the absences all score 0: ties
    assert rows[("P", "All")]["auc"] == 0.75 and rows[("P", "All")]["features"] == 2
    assert ev.groups == ["All", "LS", "Runoff"] and ev.negatives.size == 36 - 18   # both blocks are never absences
    out = ev.write(tmp_path / "run", settings={"note": "test"}, inputs=[])
    assert {p.name for p in out.iterdir()} >= {"metrics.csv", "roc_curves.csv", "negatives.gpkg",
                                               "run_info.json", "settings.json"}
    info = json.loads((out / "run_info.json").read_text())
    assert info["cells"]["absences"] == ev.negatives.size and info["features"]["LS"]["listed"] == 1
    assert "All" in ev.summary()
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    assert (out / "roc.png").exists() and (out / "sampling_map.png").exists()
    assert len([a for a in ev.plot_roc().axes if a.get_visible()]) == 3
    fig = ev.plot_sampling(points=pts, background=np.zeros(36))
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert labels == ["outside the domain (0)", "domain, not used (0)", "excluded from absences (0)",
                      "absence cells (18)", "positive cells (18)", "features"]


def test_chance_hit_rate_counts_blocks_not_cells(grid):
    yes = np.zeros(36)
    yes[14] = 1.0                                               # one flagged cell
    assert np.isclose(chance_hit_rate(grid, yes), 1 / 36)
    assert np.isclose(chance_hit_rate(grid, yes, tolerance_cells=1), 9 / 36)
    yes[:6] = np.nan                                            # cells outside the comparison do not count
    assert np.isclose(chance_hit_rate(grid, yes, tolerance_cells=1), 9 / 30)


def test_continuous_tolerance_forgives_a_one_cell_offset(grid):
    obs = np.zeros(36)
    obs[grid.x_of_node == 25] = 2.0                             # a channel in column 2
    pred = np.zeros(36)
    pred[grid.x_of_node == 35] = 2.0                            # modelled one column east
    spec = FieldSpec(name="dz", kind="continuous", units="m")
    exact = compare_fields_on_grid(grid, pred, obs, spec, spec)
    assert exact.metrics["mae"] > 0
    tol = compare_fields_on_grid(grid, pred, obs, spec, spec, tolerance_cells=1)
    assert tol.metrics["mae_within_tolerance"] == 0.0 and tol.metrics["rmse_within_tolerance"] == 0.0
    with pytest.raises(ValueError, match="tolerance_cells applies"):
        compare_fields_on_grid(grid, pred, (obs > 1).astype(float) * 1,
                               FieldSpec(name="c", kind="categorical", classes={0: "a", 2: "b"}),
                               FieldSpec(name="d", kind="categorical", classes={0: "a", 1: "b"}),
                               tolerance_cells=1)


def test_continuous_against_classes_gets_a_box_panel(grid):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from landlab_grid_validation import plot_comparison
    classes = (grid.x_of_node >= 30).astype(float)
    values = np.where(classes == 1, 0.8, 0.2)
    result = compare_fields_on_grid(grid, values, classes,
                                    FieldSpec(name="P", kind="continuous", units="probability"),
                                    FieldSpec(name="burn", kind="categorical", classes={0: "low", 1: "high"}))
    fig = plot_comparison(grid, result)
    assert fig.axes[2].get_title() == "Predicted by observed class"
    assert [t.get_text() for t in fig.axes[2].get_xticklabels()] == ["low (18)", "high (18)"]


def test_yaml_runner_end_to_end(tmp_path):
    yaml = pytest.importorskip("yaml")
    from landlab_grid_validation.__main__ import run
    p = np.zeros((6, 6))
    p[0:3, 0:3] = 0.9                                           # the whole 3 x 3 block of the point at (15, 45)
    write_tif(tmp_path / "P.tif", p)
    z = np.tile(np.arange(6, dtype=float) * 10, (6, 1))
    write_tif(tmp_path / "dem.tif", z)
    gpd.GeoDataFrame({"kind": ["a"]}, geometry=[Point(15, 45)], crs=CRS).to_file(tmp_path / "pts.gpkg")
    cfg = {"run_name": "r1", "output_dir": "runs", "predicted": {"P": "P.tif"},
           "observed": {"features": "pts.gpkg", "tolerance_cells": 1, "group_by": ["kind"]},
           "domain": [{"slope_of": "dem.tif", "min": 40}],
           "negatives": {"exclude": [{"raster": "P.tif", "min": 0.5, "buffer_cells": 0}], "min_distance_m": 0},
           "bootstrap": 20}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg))
    out = run(tmp_path / "config.yaml")
    rows = list(__import__("csv").DictReader(open(out / "metrics.csv")))
    assert {r["group"] for r in rows} == {"All", "a"} and float(rows[0]["auc"]) == 1.0
    with pytest.raises(SystemExit, match="already holds a run"):
        run(tmp_path / "config.yaml")
    (tmp_path / "bad.yaml").write_text(yaml.safe_dump({**cfg, "typo": 1, "negatives": {"nn": 2}}))
    with pytest.raises(SystemExit, match="unknown key: typo"):
        run(tmp_path / "bad.yaml")


def test_threshold_scores_and_youden_cutoff():
    from landlab_grid_validation import threshold_scores, youden_threshold
    pos, neg = np.array([0.9, 0.8, 0.4, 0.0]), np.array([0.7, 0.3, 0.0, 0.0, 0.0, 0.0])
    r = threshold_scores(pos, neg, np.r_[pos, neg], 0.5, "at_0.5")
    assert (r["tp_at_0.5"], r["fp_at_0.5"], r["fn_at_0.5"], r["tn_at_0.5"]) == (2, 1, 2, 5)
    assert r["precision_at_0.5"] == round(2 / 3, 4) and r["recall_at_0.5"] == 0.5
    assert r["csi_at_0.5"] == 0.4 and r["flagged_share_at_0.5"] == 0.3
    assert r["precision_lift_at_0.5"] == round((2 / 3) / 0.4, 2)       # base rate 4 of 10
    assert youden_threshold(pos, neg) == 0.4                             # TPR 0.75 at FPR 1/6: J = 0.58, the best


def test_evaluate_features_reports_cost_and_calibration(tmp_path, grid):
    pts = gpd.GeoDataFrame(geometry=[Point(15, 45)], crs=CRS)
    layer = np.full(36, 0.1)
    layer[vector_to_feature_cells(grid, CRS, pts, tolerance_cells=1)[0]] = 0.9
    layer[0] = 0.0
    ev = evaluate_features(grid, CRS, {"P": layer}, pts, tolerance_cells=1, bootstrap=0,
                           thresholds=(0.5, 0.05), fpr_levels=(0.10,))
    r = ev.table[0]
    assert r["auc"] == 1.0 and r["tpr_at_fpr_10pct"] == 1.0 and r["average_precision"] == 1.0
    assert r["flagged_share_at_0.5"] == 0.25 and r["precision_at_0.5"] == 1.0 and r["recall_at_0.5"] == 1.0
    assert r["flagged_share_at_0.05"] == round(35 / 36, 4) and r["precision_at_0.05"] == round(9 / 35, 4)
    assert r["optimal_threshold"] == 0.9 and r["precision_at_optimal"] == 1.0
    assert r["positives_at_zero"] == 0.0 and r["domain_above_zero"] == round(35 / 36, 4)
    assert 0 < r["brier"] < 0.02
    assert "optimal P" in ev.summary() and "precision there" in ev.summary()
