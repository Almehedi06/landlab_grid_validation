"""Validation utilities for mapped Landlab grid fields."""

__version__ = "1.2.0"

from .core import ComparisonResult, chance_hit_rate, compare_fields_on_grid, feature_hits, field_to_node_array
from .evaluate import (Evaluation, bootstrap_auc_interval, evaluate_features, feature_groups, roc_curve,
                       threshold_scores, youden_threshold)
from .export import write_field_ascii, write_field_geotiff
from .metrics import roc_auc, roc_auc_ci
from .plot import grid_extent, plot_comparison, plot_roc, plot_sampling
from .readers import (grid_from_raster, grid_transform, raster_to_node_field, vector_to_feature_cells,
                      vector_to_node_field)
from .rules import Absences, mask_from_rules, rule_mask, sample_absences, slope_degrees
from .spec import FieldSpec

__all__ = [
    "Absences",
    "ComparisonResult",
    "Evaluation",
    "FieldSpec",
    "bootstrap_auc_interval",
    "chance_hit_rate",
    "compare_fields_on_grid",
    "evaluate_features",
    "feature_groups",
    "feature_hits",
    "field_to_node_array",
    "grid_extent",
    "grid_from_raster",
    "grid_transform",
    "mask_from_rules",
    "plot_comparison",
    "plot_roc",
    "plot_sampling",
    "raster_to_node_field",
    "roc_auc",
    "roc_auc_ci",
    "roc_curve",
    "rule_mask",
    "sample_absences",
    "slope_degrees",
    "threshold_scores",
    "vector_to_feature_cells",
    "vector_to_node_field",
    "youden_threshold",
    "write_field_ascii",
    "write_field_geotiff",
]
