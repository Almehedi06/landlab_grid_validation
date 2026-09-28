"""Maps of a comparison: predicted, observed, and where they agree."""

from __future__ import annotations

import numpy as np

from .core import ComparisonResult

OUTCOMES = (("correct no", "#e6e6e6"), ("false alarm", "#fdae61"),
            ("miss", "#2c7bb6"), ("hit", "#b2182b"))


def grid_extent(grid) -> tuple[float, float, float, float]:
    """Left, right, bottom, top of the grid's cells, for imshow(origin='lower')."""
    x0, y0 = grid.xy_of_lower_left
    return (x0 - grid.dx / 2, x0 + (grid.number_of_node_columns - 0.5) * grid.dx,
            y0 - grid.dy / 2, y0 + (grid.number_of_node_rows - 0.5) * grid.dy)


def plot_comparison(grid, result: ComparisonResult, *, points=None, marker_size=16,
                    class_colours=None):
    """Three maps on the grid's own coordinates: predicted, observed, and agreement.

    The third map shows hit / miss / false alarm / correct no for any comparison
    that reduces to yes/no, predicted minus observed for two continuous fields,
    and class agreement for two categorical fields. Two continuous fields in the
    same units share one colour scale. ``points`` (a GeoDataFrame in the grid's
    CRS, or an (x, y) pair of arrays) are drawn on the predicted and observed
    maps, leaving the agreement map readable. ``class_colours``
    ({code: colour}) sets categorical colours; otherwise matplotlib's tab10 is used.

    Returns the matplotlib figure; matplotlib is imported only here.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize

    shape = (grid.number_of_node_rows, grid.number_of_node_columns)
    extent = grid_extent(grid)
    ps, os_ = result.predicted_spec, result.observed_spec
    shared = None
    if ps.kind == os_.kind == "continuous" and ps.units == os_.units:
        shared = _continuous_style(np.r_[result.predicted, result.observed], ps.units)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), constrained_layout=True)

    def show(ax, values, cmap, norm, ticks=None, labels=None, label=""):
        im = ax.imshow(np.ma.masked_invalid(values.reshape(shape)), origin="lower",
                       extent=extent, cmap=cmap, norm=norm, interpolation="nearest")
        cb = fig.colorbar(im, ax=ax, shrink=0.8, ticks=ticks)
        if labels is not None:
            cb.ax.set_yticklabels(labels)
        cb.set_label(label)

    for ax, values, spec, role in ((axes[0], result.predicted, ps, "Predicted"),
                                   (axes[1], result.observed, os_, "Observed")):
        cmap, norm, ticks, labels = _style(spec, values, shared, class_colours)
        show(ax, values, cmap, norm, ticks, labels,
             spec.units if spec.kind == "continuous" and spec.units else "")
        ax.set_title(f"{role}: {spec.name}")

    ax = axes[2]
    if result.predicted_yes is not None:
        outcome = np.where(np.isnan(result.predicted_yes) | np.isnan(result.observed_yes),
                           np.nan, result.predicted_yes + 2 * result.observed_yes)
        show(ax, outcome, ListedColormap([c for _, c in OUTCOMES]),
             BoundaryNorm(np.arange(-0.5, 4), 4), list(range(4)), [n for n, _ in OUTCOMES])
        cut = result.metrics.get("threshold")
        ax.set_title("Agreement" + (f" at threshold {cut:g}" if cut is not None else ""))
    elif result.comparison_type == "continuous_vs_continuous":
        diff = np.where(result.valid_mask, result.predicted - result.observed, np.nan)
        lim = _symmetric_limit(diff)
        show(ax, diff, "RdBu", Normalize(-lim, lim), label=ps.units or "")
        ax.set_title("Predicted − observed")
    elif result.comparison_type == "categorical_vs_categorical":
        agree = np.where(result.valid_mask,
                         (result.predicted == result.observed).astype(float), np.nan)
        show(ax, agree, ListedColormap(["#b2182b", "#e6e6e6"]),
             BoundaryNorm([-0.5, 0.5, 1.5], 2), [0, 1], ["differ", "agree"])
        ax.set_title("Class agreement")
    else:
        show(ax, np.where(result.valid_mask, 1.0, np.nan), ListedColormap(["#4d4d4d"]),
             BoundaryNorm([0.5, 1.5], 1), [1], ["compared"])
        ax.set_title("Cells compared")

    for i, ax in enumerate(axes):
        if points is not None and i < 2:
            x, y = _xy(points)
            ax.scatter(x, y, s=marker_size, facecolor="none", edgecolor="black",
                       linewidth=0.8, zorder=5)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(_summary(result), fontsize=12)
    return fig


def _style(spec, values, shared, class_colours=None):
    from matplotlib import colormaps
    from matplotlib.colors import BoundaryNorm, ListedColormap

    if spec.kind == "binary":
        yes = str(spec.positive_class) if spec.positive_class is not None else "yes"
        return (ListedColormap(["#e6e6e6", "#b2182b"]), BoundaryNorm([-0.5, 0.5, 1.5], 2),
                [0, 1], ["no", yes])
    if spec.kind == "categorical":
        names = {float(k): v for k, v in spec.classes.items()}
        codes = sorted(names)
        edges = ([codes[0] - 0.5] + [(a + b) / 2 for a, b in zip(codes, codes[1:])]
                 + [codes[-1] + 0.5])
        if class_colours:
            given = {float(k): v for k, v in class_colours.items()}
            colours = [given[c] for c in codes]
        else:
            colours = colormaps["tab10"](np.arange(len(codes)) % 10)
        return (ListedColormap(colours), BoundaryNorm(edges, len(codes)), codes,
                [names[c] for c in codes])
    cmap, norm = shared or _continuous_style(values, spec.units)
    return cmap, norm, None, None


def _continuous_style(values, units):
    from matplotlib.colors import Normalize

    if units == "probability":
        return "YlOrRd", Normalize(0.0, 1.0)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return "viridis", Normalize(0.0, 1.0)
    if finite.min() < 0 < finite.max():           # signed change: red loss, blue gain
        lim = _symmetric_limit(finite)
        return "RdBu", Normalize(-lim, lim)
    low, high = np.percentile(finite, [2, 98])
    if high <= low:
        low, high = finite.min(), max(finite.max(), finite.min() + 1.0)
    return "viridis", Normalize(low, high)


def _symmetric_limit(values) -> float:
    v = np.abs(values[np.isfinite(values) & (values != 0)])
    if v.size == 0:
        return 1.0
    lim = float(np.percentile(v, 98))
    return lim if lim > 0 else float(v.max())


def _xy(points):
    if hasattr(points, "geometry"):
        geom = points.geometry
        if not (geom.geom_type == "Point").all():
            geom = geom.representative_point()
        return geom.x.to_numpy(), geom.y.to_numpy()
    x, y = points
    return np.asarray(x), np.asarray(y)


def _summary(result: ComparisonResult) -> str:
    m = result.metrics
    if "roc_auc" in m:
        parts = [f"AUC {m['roc_auc']:.2f}", f"recall {m['recall']:.2f}",
                 f"precision {m['precision']:.2f}"]
    elif "recall" in m:
        parts = [f"recall {m['recall']:.2f}", f"precision {m['precision']:.2f}",
                 f"CSI {m['iou_csi']:.2f}"]
    elif "rmse" in m:
        parts = [f"bias {m['bias']:.2f}", f"RMSE {m['rmse']:.2f}", f"r {m['pearson_r']:.2f}"]
    elif "overall_accuracy" in m:
        parts = [f"overall accuracy {m['overall_accuracy']:.2f}"]
    else:
        parts = []
    head = (f"{result.predicted_spec.name} vs {result.observed_spec.name}  ·  "
            f"{int(result.valid_mask.sum()):,} cells compared")
    return "  ·  ".join([head] + parts)
