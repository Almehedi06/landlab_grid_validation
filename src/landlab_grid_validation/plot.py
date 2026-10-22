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


ROC_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6c5ce7", "#8a8a85")


def plot_roc(curves, *, cols=3):
    """ROC curves, one line per layer. ``curves`` is {panel title: {label: (fpr, tpr, auc)}};
    a flat {label: (fpr, tpr, auc)} gives one panel. Up to ``cols`` panels per row.

    Built with Figure directly, not pyplot, so calling it inside a notebook never
    changes the notebook's backend or leaves a figure open. Returns the figure.
    """
    from itertools import cycle

    from matplotlib.figure import Figure

    if not curves:
        raise ValueError("no curves to plot")
    first = next(iter(curves.values()))
    panels = curves if isinstance(first, dict) else {"": curves}
    n = len(panels)
    ncols = min(cols, n)
    nrows = -(-n // ncols)
    fig = Figure(figsize=(4.8 * ncols, 4.4 * nrows), constrained_layout=True)
    axes = fig.subplots(nrows, ncols, squeeze=False).ravel()
    for ax in axes[n:]:
        ax.set_visible(False)
    for ax, (title, lines) in zip(axes, panels.items()):
        ax.plot([0, 1], [0, 1], "--", color="#8a8a85", lw=1)
        for (label, (x, y, auc)), colour in zip(lines.items(), cycle(ROC_COLOURS)):
            ax.plot(x, y, color=colour, lw=2, label=f"{label}  (AUC {auc:.2f})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("false-positive rate (share of absence cells flagged)")
        ax.set_ylabel("true-positive rate (share of feature cells flagged)")
        ax.set_title(title, loc="left", fontsize=10)
        ax.legend(loc="lower right", fontsize=8, frameon=False)
        ax.grid(color="#e6e6e3", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    return fig


SAMPLING_COLOURS = {"positive": "#b2182b", "absence": "#2a78d6", "excluded": "#7b3294", "domain": "#d9d9d6"}


def plot_sampling(grid, *, domain, positives, absences, excluded=None, points=None, background=None,
                  title="Where the two classes come from"):
    """Map of the cells an evaluation used: the domain (cells either class may come
    from), the positive cells, the absence cells, and cells excluded from the
    absences (an event footprint). ``positives`` is a list of cell arrays or a
    boolean node mask; ``absences`` node indices or a mask; ``excluded`` a mask;
    ``points`` the features, drawn on top; ``background`` an elevation node array
    drawn as a hillshade. Counts go in the legend. Built with Figure directly, so
    it never changes a notebook's backend; returns the figure.
    """
    from matplotlib.colors import LightSource, ListedColormap
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch

    shape = (grid.number_of_node_rows, grid.number_of_node_columns)
    extent = grid_extent(grid)
    domain = np.asarray(domain, dtype=bool)
    pos = _as_mask(grid, positives)
    neg = _as_mask(grid, absences)
    exc = np.zeros(grid.number_of_nodes, bool) if excluded is None else np.asarray(excluded, dtype=bool)
    # draw order: domain, excluded (only inside the domain), absences, positives
    classes = np.full(grid.number_of_nodes, np.nan)
    classes[domain] = 0
    classes[domain & exc] = 1
    classes[neg] = 2
    classes[pos] = 3
    names = ["domain, not used", "excluded from absences", "absence cells", "positive cells"]
    colours = [SAMPLING_COLOURS[k] for k in ("domain", "excluded", "absence", "positive")]
    counts = [int(np.sum(classes == i)) for i in range(4)]

    fig = Figure(figsize=(9, 8), constrained_layout=True)
    ax = fig.subplots()
    if background is not None:
        z = np.asarray(background, dtype=float).reshape(shape)
        z = np.where(np.isfinite(z), z, np.nanmean(z))
        shade = LightSource(azdeg=315, altdeg=45).hillshade(z, vert_exag=1.5, dx=grid.dx, dy=grid.dy)
        ax.imshow(shade, origin="lower", extent=extent, cmap="gray", vmin=0, vmax=1, alpha=0.6)
    ax.imshow(np.ma.masked_invalid(classes.reshape(shape)), origin="lower", extent=extent,
              cmap=ListedColormap(colours), vmin=-0.5, vmax=3.5, interpolation="nearest", alpha=0.9)
    if points is not None:
        x, y = _xy(points)
        ax.scatter(x, y, s=14, facecolor="white", edgecolor="black", linewidth=0.6, zorder=5, label="features")
    handles = [Patch(facecolor=c, edgecolor="none", label=f"{n} ({k:,})") for n, c, k in zip(names, colours, counts)]
    handles.insert(0, Patch(facecolor="white", edgecolor="#8a8a85", label=f"outside the domain ({int(np.sum(~domain)):,})"))
    if points is not None:
        handles.append(ax.get_legend_handles_labels()[0][0])
    ax.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.92)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, loc="left")
    return fig


def _as_mask(grid, cells):
    arr = np.asarray(cells, dtype=object) if isinstance(cells, list) else np.asarray(cells)
    if isinstance(cells, list):
        mask = np.zeros(grid.number_of_nodes, bool)
        for c in cells:
            mask[np.asarray(c, dtype=int)] = True
        return mask
    if arr.dtype == bool:
        return arr
    mask = np.zeros(grid.number_of_nodes, bool)
    mask[arr.astype(int)] = True
    return mask


def plot_comparison(grid, result: ComparisonResult, *, points=None, colour_by=None,
                    feature_hits=None, chance_hit_rate=None, marker_size=16, class_colours=None):
    """Three maps on the grid's own coordinates: predicted, observed, and agreement.

    The third map shows hit / miss / false alarm / correct no for any comparison
    that reduces to yes/no, predicted minus observed for two continuous fields,
    and class agreement for two categorical fields. Two continuous fields in the
    same units share one colour scale. ``points`` (a GeoDataFrame in the grid's
    CRS, or an (x, y) pair of arrays) are drawn on the predicted and observed
    maps, leaving the agreement map readable. ``colour_by`` names a column of
    ``points``: each distinct value gets its own colour and a legend entry with
    its count, so one map shows every observation type; missing values are
    "unlabelled". ``feature_hits`` (one True/False per entry of ``points``, from
    ``feature_hits``) replaces the cell-by-cell agreement map with the
    feature-level view: the cells predicted yes, each feature marked hit or
    miss, and the share of cells flagged and of features hit in the title;
    ``chance_hit_rate`` (from ``chance_hit_rate``) adds the hit rate of a random
    location to that title. A continuous field against classes gets a box plot
    of the values per class as its third panel. ``class_colours``
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
        shared = _continuous_style(np.r_[result.predicted, result.observed], ps.units,
                                   ps.sign_convention)

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
    box_axis = None
    if feature_hits is not None:
        hits = _hits_array(feature_hits, points)
        if result.predicted_yes is None:
            raise ValueError("feature_hits needs a comparison that reduces to yes/no")
        show(ax, result.predicted_yes, ListedColormap(["#e6e6e6", "#fdae61"]),
             BoundaryNorm([-0.5, 0.5, 1.5], 2), [0, 1], ["not flagged", "flagged"])
        x, y = _xy(points)
        for hit, colour, name in ((True, OUTCOMES[3][1], "hit"), (False, OUTCOMES[2][1], "miss")):
            sel = hits == hit
            if sel.any():
                ax.scatter(x[sel], y[sel], s=marker_size * 1.5, facecolor=colour, edgecolor="black",
                           linewidth=0.5, label=f"{name} ({int(sel.sum())})", zorder=5)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        cut = result.metrics.get("threshold")
        chance = f"\nrandom location hit {chance_hit_rate:.0%}" if chance_hit_rate is not None else ""
        ax.set_title(f"Flagged{f' at {cut:g}' if cut is not None else ''}: "
                     f"{np.nanmean(result.predicted_yes):.0%} of cells, "
                     f"{hits.mean():.0%} of {hits.size} features hit{chance}")
    elif result.comparison_type in ("continuous_by_observed_category",
                                    "observed_continuous_by_predicted_category"):
        by_observed = result.comparison_type == "continuous_by_observed_category"
        values, classes, spec = ((result.predicted, result.observed, os_) if by_observed
                                 else (result.observed, result.predicted, ps))
        names = {float(k): v for k, v in spec.classes.items()}
        codes = sorted(names)
        data = [values[result.valid_mask & (classes == c)] for c in codes]
        ax.boxplot(data, showfliers=False)
        ax.set_xticks(range(1, len(codes) + 1), [f"{names[c]} ({d.size})" for c, d in zip(codes, data)])
        ax.set_ylabel((ps if by_observed else os_).units or "")
        ax.set_title("Predicted by observed class" if by_observed else "Observed by predicted class")
        box_axis = ax
    elif result.predicted_yes is not None:
        show(ax, result.agreement(), ListedColormap([c for _, c in OUTCOMES]),
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

    groups = _point_groups(points, colour_by) if points is not None else []
    for i, ax in enumerate(axes):
        if ax is box_axis:
            continue
        if i < 2:
            for label, x, y, colour in groups:
                ax.scatter(x, y, s=marker_size, facecolor=colour, edgecolor="black",
                           linewidth=0.8 if colour == "none" else 0.5, label=label, zorder=5)
            if colour_by is not None and i == 1:
                ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
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
        # Class boundaries sit halfway between codes, so the pairs are one shorter.
        midpoints = [(a + b) / 2 for a, b in zip(codes, codes[1:], strict=False)]
        edges = [codes[0] - 0.5] + midpoints + [codes[-1] + 0.5]
        if class_colours:
            given = {float(k): v for k, v in class_colours.items()}
            colours = [given[c] for c in codes]
        else:
            colours = colormaps["tab10"](np.arange(len(codes)) % 10)
        return (ListedColormap(colours), BoundaryNorm(edges, len(codes)), codes,
                [names[c] for c in codes])
    cmap, norm = shared or _continuous_style(values, spec.units, spec.sign_convention)
    return cmap, norm, None, None


def _continuous_style(values, units, sign_convention=None):
    from matplotlib.colors import Normalize

    if units == "probability":
        return "YlOrRd", Normalize(0.0, 1.0)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return "viridis", Normalize(0.0, 1.0)
    if finite.min() < 0 < finite.max():           # signed change: loss red, gain blue
        lim = _symmetric_limit(finite)
        cmap = "RdBu_r" if sign_convention == "positive_is_loss" else "RdBu"
        return cmap, Normalize(-lim, lim)
    nonzero = finite[finite != 0]                 # sparse fields: scale on the non-zero cells
    base = nonzero if nonzero.size else finite
    low = 0.0 if finite.min() >= 0 else float(np.percentile(base, 2))
    high = float(np.percentile(base, 98))
    if high <= low:
        high = low + 1.0
    return "viridis", Normalize(low, high)


def _symmetric_limit(values) -> float:
    v = np.abs(values[np.isfinite(values) & (values != 0)])
    if v.size == 0:
        return 1.0
    lim = float(np.percentile(v, 98))
    return lim if lim > 0 else float(v.max())


def _hits_array(feature_hits, points):
    if points is None:
        raise ValueError("feature_hits needs points, in the same order")
    hits = np.asarray(feature_hits, dtype=bool).ravel()
    n = _xy(points)[0].size
    if hits.size != n:
        raise ValueError(f"feature_hits has {hits.size} entries for {n} points")
    return hits


def _point_groups(points, colour_by):
    """The points as (label, x, y, face colour) groups: one hollow group when
    ``colour_by`` is None, otherwise one filled group per distinct value of that
    column, in sorted order, labelled with the value and its count."""
    x, y = _xy(points)
    if colour_by is None:
        return [(None, x, y, "none")]
    if not hasattr(points, "columns") or colour_by not in points.columns:
        raise ValueError(f"colour_by={colour_by!r} is not a column of points")
    from matplotlib import colormaps

    column = points[colour_by]
    labels = np.where(column.isna(), "unlabelled", column.astype(str))
    tab10 = colormaps["tab10"]
    return [(f"{value} ({int((labels == value).sum())})", x[labels == value], y[labels == value],
             tab10(i % 10)) for i, value in enumerate(sorted(set(labels)))]


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
