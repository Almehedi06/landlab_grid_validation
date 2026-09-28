"""Core comparison function for mapped fields on a Landlab grid."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .metrics import (
    binary_metrics,
    categorical_metrics,
    continuous_by_category,
    continuous_metrics,
    finite_pair_mask,
    probability_metrics,
)
from .spec import FieldSpec


@dataclass
class ComparisonResult:
    """Result from comparing predicted and observed fields."""

    comparison_type: str
    metrics: dict[str, Any]
    predicted: np.ndarray
    observed: np.ndarray
    valid_mask: np.ndarray
    predicted_spec: FieldSpec
    observed_spec: FieldSpec

    def to_dict(self) -> dict[str, Any]:
        return {
            "comparison_type": self.comparison_type,
            "metrics": self.metrics,
            "predicted_spec": self.predicted_spec.__dict__,
            "observed_spec": self.observed_spec.__dict__,
            "n_valid": int(np.sum(self.valid_mask)),
        }


def field_to_node_array(grid, values, *, nodata=None) -> np.ndarray:
    """Return a node-length array from a Landlab field name or array-like values.

    Parameters
    ----------
    grid : landlab RasterModelGrid-like
        Grid whose nodes define the comparison domain.
    values : str or array-like
        If a string, it is interpreted as a field name in ``grid.at_node``.
        Arrays may be one-dimensional node arrays or two-dimensional rasters
        matching the node row/column shape.
    nodata : int or float, optional
        Values equal to nodata are converted to ``np.nan``.
    """
    if isinstance(values, str):
        if values not in grid.at_node:
            raise KeyError(f"{values!r} is not a field at grid nodes")
        arr = np.asarray(grid.at_node[values])
    else:
        arr = np.asarray(values)

    if arr.ndim == 2:
        expected = (grid.number_of_node_rows, grid.number_of_node_columns)
        if arr.shape != expected:
            raise ValueError(f"2D array shape {arr.shape} does not match grid {expected}")
        arr = arr.reshape(grid.number_of_nodes)
    elif arr.ndim != 1:
        raise ValueError("field values must be a node-length 1D array or grid-shaped 2D array")

    if arr.size != grid.number_of_nodes:
        raise ValueError(
            f"field has {arr.size} values but grid has {grid.number_of_nodes} nodes"
        )

    out = arr.astype(float, copy=True)
    if nodata is not None:
        out[out == nodata] = np.nan
    return out


def compare_fields_on_grid(
    grid,
    predicted,
    observed,
    predicted_spec: FieldSpec,
    observed_spec: FieldSpec,
    *,
    mask=None,
    threshold: float | None = None,
    plot: bool = False,
) -> ComparisonResult:
    """Compare predicted and observed mapped fields on one Landlab grid.

    This is the core abstraction. File formats should be handled upstream by
    adapters that add fields to the grid or pass node arrays into this function.
    """
    pred = field_to_node_array(grid, predicted, nodata=predicted_spec.nodata)
    obs = field_to_node_array(grid, observed, nodata=observed_spec.nodata)

    valid = finite_pair_mask(pred, obs)
    if mask is not None:
        mask_arr = field_to_node_array(grid, mask).astype(bool)
        valid &= mask_arr

    if not np.any(valid):
        raise ValueError("no overlapping valid cells between predicted and observed fields")

    pred_v = pred[valid]
    obs_v = obs[valid]

    comparison_type, metrics = _compute_metrics(
        pred_v,
        obs_v,
        predicted_spec,
        observed_spec,
        threshold=threshold,
    )

    result = ComparisonResult(
        comparison_type=comparison_type,
        metrics=metrics,
        predicted=pred,
        observed=obs,
        valid_mask=valid,
        predicted_spec=predicted_spec,
        observed_spec=observed_spec,
    )

    if plot:
        from .plot import plot_comparison

        result.metrics["figure"] = plot_comparison(grid, result)

    return result


def _compute_metrics(
    predicted: np.ndarray,
    observed: np.ndarray,
    predicted_spec: FieldSpec,
    observed_spec: FieldSpec,
    *,
    threshold: float | None,
) -> tuple[str, dict[str, Any]]:
    pk = predicted_spec.kind
    ok = observed_spec.kind

    if pk == "continuous" and ok == "continuous":
        if predicted_spec.units == "probability" and _looks_binary(observed):
            use_threshold = _threshold(predicted_spec, threshold)
            return (
                "probability_vs_binary",
                probability_metrics(predicted, observed.astype(bool), use_threshold),
            )
        return "continuous_vs_continuous", continuous_metrics(predicted, observed)

    if pk == "continuous" and ok == "binary":
        use_threshold = _threshold(predicted_spec, threshold)
        return (
            "probability_vs_binary"
            if predicted_spec.units == "probability"
            else "continuous_threshold_vs_binary",
            probability_metrics(predicted, observed.astype(bool), use_threshold)
            if predicted_spec.units == "probability"
            else binary_metrics(predicted >= use_threshold, observed.astype(bool)),
        )

    if pk == "binary" and ok == "binary":
        return "binary_vs_binary", binary_metrics(predicted.astype(bool), observed.astype(bool))

    if pk == "categorical" and ok == "categorical":
        labels = _class_labels(predicted_spec, observed_spec)
        return "categorical_vs_categorical", categorical_metrics(predicted, observed, labels)

    if pk == "continuous" and ok == "categorical":
        return "continuous_by_observed_category", continuous_by_category(predicted, observed)

    if pk == "categorical" and ok == "continuous":
        return "observed_continuous_by_predicted_category", continuous_by_category(observed, predicted)

    if pk == "binary" and ok == "continuous":
        use_threshold = _threshold(observed_spec, threshold)
        return "binary_vs_thresholded_continuous", binary_metrics(
            predicted.astype(bool), observed >= use_threshold
        )

    if pk == "categorical" and ok == "binary":
        positive = _positive_value(predicted_spec)
        return "category_positive_vs_binary", binary_metrics(
            predicted == positive, observed.astype(bool)
        )

    if pk == "binary" and ok == "categorical":
        positive = _positive_value(observed_spec)
        return "binary_vs_category_positive", binary_metrics(
            predicted.astype(bool), observed == positive
        )

    raise ValueError(f"unsupported comparison: predicted {pk}, observed {ok}")


def _threshold(spec: FieldSpec, override: float | None) -> float:
    value = override if override is not None else spec.threshold
    if value is None:
        value = 0.5
    return float(value)


def _looks_binary(values: np.ndarray) -> bool:
    finite = values[np.isfinite(values)]
    return set(np.unique(finite).tolist()).issubset({0.0, 1.0})


def _class_labels(predicted_spec: FieldSpec, observed_spec: FieldSpec) -> list[Any]:
    labels: list[Any] = []
    for spec in (observed_spec, predicted_spec):
        if spec.classes:
            for key in spec.classes:
                if key not in labels:
                    labels.append(key)
    return labels


def _positive_value(spec: FieldSpec):
    if spec.positive_class is None:
        raise ValueError(
            f"{spec.name}: positive_class is required to convert categorical to binary"
        )
    if spec.classes is None:
        return spec.positive_class
    for key, label in spec.classes.items():
        if key == spec.positive_class or label == spec.positive_class:
            return key
    raise ValueError(f"{spec.name}: positive_class {spec.positive_class!r} not in classes")

