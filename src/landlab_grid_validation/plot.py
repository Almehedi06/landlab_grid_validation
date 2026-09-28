"""Simple visualization helpers for Landlab field comparisons."""

from __future__ import annotations

import numpy as np

from .core import ComparisonResult


def plot_comparison(grid, result: ComparisonResult):
    """Plot predicted, observed and residual/valid-mask panels.

    Returns the matplotlib figure. Matplotlib is imported lazily so the core
    comparison utility can run without plotting dependencies.
    """
    import matplotlib.pyplot as plt

    shape = (grid.number_of_node_rows, grid.number_of_node_columns)
    predicted = result.predicted.reshape(shape)
    observed = result.observed.reshape(shape)
    valid = result.valid_mask.reshape(shape)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

    im0 = axes[0].imshow(predicted, origin="lower")
    axes[0].set_title(f"Predicted: {result.predicted_spec.name}")
    fig.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].imshow(observed, origin="lower")
    axes[1].set_title(f"Observed: {result.observed_spec.name}")
    fig.colorbar(im1, ax=axes[1], shrink=0.8)

    if result.predicted_spec.kind == "continuous" and result.observed_spec.kind == "continuous":
        panel = np.where(valid, result.predicted - result.observed, np.nan).reshape(shape)
        title = "Predicted - observed"
    else:
        panel = valid.astype(float)
        title = "Valid comparison mask"

    im2 = axes[2].imshow(panel, origin="lower")
    axes[2].set_title(title)
    fig.colorbar(im2, ax=axes[2], shrink=0.8)

    for ax in axes:
        ax.set_xlabel("column")
        ax.set_ylabel("row")

    return fig

