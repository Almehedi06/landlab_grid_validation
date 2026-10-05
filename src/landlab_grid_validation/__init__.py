"""Validation utilities for mapped Landlab grid fields."""

__version__ = "1.0.0"

from .core import ComparisonResult, compare_fields_on_grid, field_to_node_array
from .export import write_field_ascii
from .metrics import roc_auc, roc_auc_ci
from .plot import grid_extent, plot_comparison
from .readers import grid_from_raster, grid_transform, raster_to_node_field, vector_to_node_field
from .spec import FieldSpec

__all__ = [
    "ComparisonResult",
    "FieldSpec",
    "compare_fields_on_grid",
    "field_to_node_array",
    "grid_extent",
    "grid_from_raster",
    "grid_transform",
    "plot_comparison",
    "roc_auc",
    "roc_auc_ci",
    "raster_to_node_field",
    "vector_to_node_field",
    "write_field_ascii",
]
