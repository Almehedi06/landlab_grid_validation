"""Core comparison function for mapped fields on a Landlab grid."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .metrics import (
    binary_metrics,
    categorical_metrics,
    continuous_by_category,
    continuous_metrics,
    score_vs_binary_metrics,
)
from .spec import FieldSpec


@dataclass
class ComparisonResult:
    """Result from comparing predicted and observed fields.

    ``predicted_yes`` and ``observed_yes`` are the yes/no maps the metrics were
    computed from (NaN outside the compared cells), set whenever the comparison
    reduces to yes/no. ``masked_out_of_range`` counts values removed by each
    FieldSpec's ``valid_range``.
    """

    comparison_type: str
    metrics: dict[str, Any]
    predicted: np.ndarray
    observed: np.ndarray
    valid_mask: np.ndarray
    predicted_spec: FieldSpec
    observed_spec: FieldSpec
    predicted_yes: np.ndarray | None = None
    observed_yes: np.ndarray | None = None
    masked_out_of_range: dict[str, int] = field(default_factory=dict)
    figure: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "comparison_type": self.comparison_type,
            "metrics": self.metrics,
            "predicted_spec": self.predicted_spec.__dict__,
            "observed_spec": self.observed_spec.__dict__,
            "n_valid": int(np.sum(self.valid_mask)),
            "masked_out_of_range": self.masked_out_of_range,
        }


def field_to_node_array(grid, values, *, nodata=None, origin=None) -> np.ndarray:
    """Return a node-length float array from a Landlab field name or array-like values.

    Parameters
    ----------
    grid : landlab RasterModelGrid-like
        Grid whose nodes define the comparison domain.
    values : str or array-like
        A field name in ``grid.at_node``, a 1D array in Landlab node order, or a
        2D array of the grid's (rows, columns) shape.
    nodata : int or float, optional
        Values equal to nodata are converted to ``np.nan``.
    origin : {"upper", "lower"}, required for 2D arrays
        "upper" when row 0 is the north edge (GeoTIFF, rasterio, most images);
        "lower" when row 0 is the south edge (Landlab). There is no default:
        guessing wrong flips the map upside down without any error.
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
        if origin not in ("upper", "lower"):
            raise ValueError(
                "2D arrays need origin='upper' (row 0 = north, as in GeoTIFF/rasterio) "
                "or origin='lower' (row 0 = south, as in Landlab)"
            )
        if origin == "upper":
            arr = arr[::-1]
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
    origin: str | None = None,
    plot: bool = False,
) -> ComparisonResult:
    """Compare predicted and observed mapped fields on one Landlab grid.

    This is the core abstraction. File formats are handled upstream by the
    readers in ``landlab_grid_validation.readers``, which return node arrays.

    ``mask`` limits the comparison to cells where it is non-zero, for example
    evidence plus pseudo-absence cells. ``origin`` applies to any 2D arrays.
    Values are checked against each FieldSpec before any metric is computed:
    probabilities must lie in [0, 1], binary fields may hold only 0 and 1, and
    categorical fields only their declared class codes.
    """
    pred, n_pred = _prepare(grid, predicted, predicted_spec, origin)
    obs, n_obs = _prepare(grid, observed, observed_spec, origin)

    valid = np.isfinite(pred) & np.isfinite(obs)
    if mask is not None:
        mask_arr = field_to_node_array(grid, mask, origin=origin)
        valid &= np.nan_to_num(mask_arr, nan=0.0) != 0

    if not np.any(valid):
        raise ValueError("no overlapping valid cells between predicted and observed fields")

    comparison_type, metrics, pred_yes, obs_yes = _compute_metrics(
        pred[valid],
        obs[valid],
        predicted_spec,
        observed_spec,
        threshold=threshold,
    )

    def on_grid(values):
        if values is None:
            return None
        out = np.full(pred.size, np.nan)
        out[valid] = values
        return out

    result = ComparisonResult(
        comparison_type=comparison_type,
        metrics=metrics,
        predicted=pred,
        observed=obs,
        valid_mask=valid,
        predicted_spec=predicted_spec,
        observed_spec=observed_spec,
        predicted_yes=on_grid(pred_yes),
        observed_yes=on_grid(obs_yes),
        masked_out_of_range={"predicted": n_pred, "observed": n_obs},
    )

    if plot:
        from .plot import plot_comparison

        result.figure = plot_comparison(grid, result)

    return result


def _prepare(grid, values, spec: FieldSpec, origin) -> tuple[np.ndarray, int]:
    arr = field_to_node_array(grid, values, nodata=spec.nodata, origin=origin)
    n_out = 0
    if spec.valid_range is not None:
        low, high = spec.valid_range
        outside = np.isfinite(arr) & ((arr < low) | (arr > high))
        n_out = int(outside.sum())
        arr[outside] = np.nan
        if n_out:
            warnings.warn(
                f"{spec.name}: {n_out} values outside valid_range {spec.valid_range} "
                "set to no-data",
                stacklevel=3,
            )
    _check_values(arr, spec)
    return arr, n_out


def _check_values(arr: np.ndarray, spec: FieldSpec) -> None:
    finite = arr[np.isfinite(arr)]
    if spec.units == "probability":
        bad = int(np.sum((finite < 0) | (finite > 1)))
        if bad:
            raise ValueError(
                f"{spec.name}: {bad} probability values outside [0, 1]; "
                "declare nodata or valid_range"
            )
    if spec.kind == "binary":
        extra = sorted(set(np.unique(finite).tolist()) - {0.0, 1.0})
        if extra:
            raise ValueError(
                f"{spec.name}: binary field holds {extra[:5]}; expected only 0 and 1"
            )
    if spec.kind == "categorical":
        known = {_code(spec, key) for key in spec.classes}
        unknown = sorted(set(np.unique(finite).tolist()) - known)
        if unknown:
            raise ValueError(
                f"{spec.name}: values {unknown[:5]} are not among its classes {sorted(known)}"
            )


def _compute_metrics(
    predicted: np.ndarray,
    observed: np.ndarray,
    predicted_spec: FieldSpec,
    observed_spec: FieldSpec,
    *,
    threshold: float | None,
) -> tuple[str, dict[str, Any], np.ndarray | None, np.ndarray | None]:
    pk = predicted_spec.kind
    ok = observed_spec.kind

    if pk == "continuous" and ok == "continuous":
        return "continuous_vs_continuous", continuous_metrics(predicted, observed), None, None

    if pk == "continuous" and ok == "binary":
        cut = _threshold(predicted_spec, threshold)
        probability = predicted_spec.units == "probability"
        return (
            "probability_vs_binary" if probability else "continuous_vs_binary",
            score_vs_binary_metrics(predicted, observed, cut, probability=probability),
            (predicted >= cut).astype(float),
            observed,
        )

    if pk == "binary" and ok == "binary":
        return "binary_vs_binary", binary_metrics(predicted, observed), predicted, observed

    if pk == "categorical" and ok == "categorical":
        labels = _class_labels(predicted_spec, observed_spec)
        return (
            "categorical_vs_categorical",
            categorical_metrics(predicted, observed, labels),
            None,
            None,
        )

    if pk == "continuous" and ok == "categorical":
        return (
            "continuous_by_observed_category",
            continuous_by_category(predicted, observed),
            None,
            None,
        )

    if pk == "categorical" and ok == "continuous":
        return (
            "observed_continuous_by_predicted_category",
            continuous_by_category(observed, predicted),
            None,
            None,
        )

    if pk == "binary" and ok == "continuous":
        observed_yes = (observed >= _threshold(observed_spec, threshold)).astype(float)
        return (
            "binary_vs_thresholded_continuous",
            binary_metrics(predicted, observed_yes),
            predicted,
            observed_yes,
        )

    if pk == "categorical" and ok == "binary":
        predicted_yes = (predicted == _positive_value(predicted_spec)).astype(float)
        return (
            "category_positive_vs_binary",
            binary_metrics(predicted_yes, observed),
            predicted_yes,
            observed,
        )

    if pk == "binary" and ok == "categorical":
        observed_yes = (observed == _positive_value(observed_spec)).astype(float)
        return (
            "binary_vs_category_positive",
            binary_metrics(predicted, observed_yes),
            predicted,
            observed_yes,
        )

    raise ValueError(f"unsupported comparison: predicted {pk}, observed {ok}")


def _threshold(spec: FieldSpec, override: float | None) -> float:
    value = override if override is not None else spec.threshold
    if value is None:
        raise ValueError(
            f"{spec.name}: turning this continuous field into yes/no needs a threshold; "
            "set FieldSpec.threshold or pass threshold="
        )
    return float(value)


def _code(spec: FieldSpec, key) -> float:
    try:
        return float(key)
    except (TypeError, ValueError):
        raise ValueError(
            f"{spec.name}: class key {key!r} is not a numeric code; use {{code: label}}"
        ) from None


def _class_labels(predicted_spec: FieldSpec, observed_spec: FieldSpec) -> list[Any]:
    labels: list[Any] = []
    for spec in (observed_spec, predicted_spec):
        if spec.classes:
            for key in spec.classes:
                if key not in labels:
                    labels.append(key)
    return labels


def _positive_value(spec: FieldSpec) -> float:
    if spec.positive_class is None:
        raise ValueError(
            f"{spec.name}: positive_class is required to convert categorical to binary"
        )
    if spec.classes is None:
        return _code(spec, spec.positive_class)
    for key, label in spec.classes.items():
        if key == spec.positive_class or label == spec.positive_class:
            return _code(spec, key)
    raise ValueError(f"{spec.name}: positive_class {spec.positive_class!r} not in classes")
