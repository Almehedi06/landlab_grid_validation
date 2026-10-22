"""Score gridded layers against mapped features: AUC by group with an interval,
ROC curves, and the absence cells used, all from one call.

    ev = evaluate_features(grid, crs, {"P_Landslide": p}, points, tolerance_cells=1,
                           group_by=["FailureTyp"], domain=domain, absences=Absences(min_distance_m=50))
    print(ev.summary())
    ev.write("evaluation/run1")

Protocol (Strauch et al. 2018, adapted to any mapped feature): the positives are
the cells of each feature grown by the tolerance, inside the domain; the
absences are the other domain cells, filtered as ``Absences`` says. The AUC
ranks every positive cell against every absence cell, so it needs no threshold
and does not depend on how many absences there are. The 95% interval resamples
the features, not the cells: features are the independent observations.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import average_precision, roc_auc
from .readers import vector_to_feature_cells
from .rules import Absences, sample_absences


def roc_curve(positive_scores, reference_scores):
    """ROC curve: for every threshold, x = share of the reference cells at or above
    it, y = share of the positive cells at or above it. Ties join with a straight
    line, so the area under it is the rank AUC."""
    pos, ref = np.sort(np.asarray(positive_scores, float)), np.sort(np.asarray(reference_scores, float))
    thresholds = np.unique(np.r_[pos, ref])[::-1]
    y = 1 - np.searchsorted(pos, thresholds, "left") / pos.size
    x = 1 - np.searchsorted(ref, thresholds, "left") / ref.size
    return np.r_[0.0, x], np.r_[0.0, y]


def area_under(x, y) -> float:
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2))


def _rates(positive_scores, reference_scores, thresholds):
    """True- and false-positive rates at each threshold (score >= threshold)."""
    pos, ref = np.sort(np.asarray(positive_scores, float)), np.sort(np.asarray(reference_scores, float))
    tpr = 1 - np.searchsorted(pos, thresholds, "left") / pos.size
    fpr = 1 - np.searchsorted(ref, thresholds, "left") / ref.size
    return tpr, fpr


def youden_threshold(positive_scores, reference_scores) -> float:
    """The cutoff that maximises TPR - FPR (Youden's J): the point of the ROC curve
    farthest above the diagonal. Unlike "max precision + recall", it does not
    depend on the ratio of positives to absences, so it is the same whether every
    absence cell is used or a 1:1 draw."""
    thresholds = np.unique(np.r_[positive_scores, reference_scores])[::-1]
    tpr, fpr = _rates(positive_scores, reference_scores, thresholds)
    return float(thresholds[np.argmax(tpr - fpr)])


def threshold_scores(positive_scores, reference_scores, domain_scores, threshold, tag) -> dict:
    """Confusion counts and scores when cells with score >= threshold are flagged:
    tp, fp, fn, tn, precision, recall, f1, csi, the share of the domain flagged,
    and precision as a multiple of the base rate (positives / (positives +
    absences)); 1 means no better than flagging at random. Keys end in ``tag``."""
    pos, ref = np.asarray(positive_scores, float), np.asarray(reference_scores, float)
    tp, fp = int(np.sum(pos >= threshold)), int(np.sum(ref >= threshold))
    fn, tn = pos.size - tp, ref.size - fp
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / pos.size if pos.size else np.nan
    f1 = 2 * precision * recall / (precision + recall) if np.isfinite(precision + recall) and precision + recall else np.nan
    base = pos.size / (pos.size + ref.size)
    return {f"tp_{tag}": tp, f"fp_{tag}": fp, f"fn_{tag}": fn, f"tn_{tag}": tn,
            f"precision_{tag}": round(float(precision), 4), f"recall_{tag}": round(float(recall), 4),
            f"f1_{tag}": round(float(f1), 4), f"csi_{tag}": round(float(tp / (tp + fp + fn)) if tp + fp + fn else np.nan, 4),
            f"precision_lift_{tag}": round(float(precision / base), 2) if np.isfinite(precision) else np.nan,
            f"flagged_share_{tag}": round(float(np.mean(np.asarray(domain_scores, float) >= threshold)), 4)}


def bootstrap_auc_interval(scores, feature_cells, absence_scores, n_boot=1000, seed=42):
    """95% interval of the AUC from resampling the features with replacement.

    Each positive cell's placement (share of absence cells below it, ties half)
    is computed once, so each resample is a weighted mean. The same seed for
    every layer compares the layers on the same resamples. The interval reflects
    how few features there are; it does not cover the choice of absence rules.
    """
    neg = np.sort(np.asarray(absence_scores, float))
    sums, counts = [], []
    for cells in feature_cells:
        s = np.asarray(scores, float)[np.asarray(cells, dtype=int)]
        placement = (np.searchsorted(neg, s, "left") + np.searchsorted(neg, s, "right")) / 2 / neg.size
        sums.append(placement.sum())
        counts.append(s.size)
    sums, counts = np.array(sums), np.array(counts)
    if sums.size == 0 or n_boot <= 0:
        return np.nan, np.nan
    pick = np.random.default_rng(seed).integers(0, sums.size, size=(n_boot, sums.size))
    aucs = sums[pick].sum(1) / counts[pick].sum(1)
    low, high = np.percentile(aucs, [2.5, 97.5])
    return float(low), float(high)


def feature_groups(features, group_by=None) -> list[tuple[str, np.ndarray]]:
    """Groups of features to score: "All", then each value of the first column,
    then each combination with the next column, and so on ("landslide / LS").
    Missing values are "unlabelled"."""
    n = len(features)
    groups: list[tuple[str, np.ndarray]] = [("All", np.arange(n))]
    columns = [group_by] if isinstance(group_by, str) else list(group_by or [])
    missing = [c for c in columns if c not in features.columns]
    if missing:
        raise ValueError(f"group_by columns {missing} are not in the features")
    labels = []
    for column in columns:
        values = features[column]
        labels.append(np.where(values.isna(), "unlabelled", values.astype(str)).astype(str))
    for level in range(1, len(columns) + 1):
        keys = np.array([" / ".join(parts) for parts in zip(*labels[:level])])
        for key in sorted(set(keys.tolist())):
            groups.append((key, np.flatnonzero(keys == key)))
    return groups


@dataclass
class Evaluation:
    """What ``evaluate_features`` returns. ``table`` has one row per layer and group."""

    table: list[dict[str, Any]]
    curves: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, float]]
    groups: list[str]
    layers: list[str]
    group_counts: dict[str, tuple[int, int]]      # group -> (features listed, with cells in the domain)
    feature_cells: list[np.ndarray]               # each feature's scored cells, inside the domain
    negatives: np.ndarray                         # node indices of the absence cells
    domain: np.ndarray                            # boolean node mask
    grid: Any = field(repr=False)
    crs: Any = field(repr=False)
    excluded: np.ndarray | None = field(default=None, repr=False)   # cells kept out of the absences

    def plot_sampling(self, *, points=None, background=None):
        """Map of the domain, the positive cells, the absence cells and the excluded cells."""
        from .plot import plot_sampling

        return plot_sampling(self.grid, domain=self.domain, positives=self.feature_cells, absences=self.negatives,
                             excluded=self.excluded, points=points, background=background)

    def summary(self) -> str:
        """One line per group and layer, the three numbers to compare runs by: AUC
        (95% interval), the optimal cutoff (Youden: highest true-positive rate minus
        false-positive rate) and the precision at that cutoff, also as a multiple of
        the base rate. Every other score is in ``table`` (metrics.csv)."""
        width = max(len(g) for g in self.groups) + 2
        lw = max(len(n) for n in self.layers) + 2
        has_opt = "optimal_threshold" in self.table[0]
        head = f"{'group':{width}s}{'layer':{lw}s}{'AUC (95%)':>18s}"
        if has_opt:
            head += f"{'optimal P':>11s}{'precision there':>17s}{'x base':>8s}"
        lines = [head]
        for group in self.groups:
            for name in self.layers:
                r = next(r for r in self.table if r["group"] == group and r["layer"] == name)
                line = f"{group:{width}s}{name:{lw}s}{r['auc']:>8.3f} ({r['auc_low']:.2f}-{r['auc_high']:.2f})"
                if has_opt:
                    line += (f"{r['optimal_threshold']:>11.3g}{r['precision_at_optimal']:>17.1%}"
                             f"{r['precision_lift_at_optimal']:>8.1f}")
                lines.append(line)
        lines.append(f"{int(self.domain.sum())} domain cells, {self.negatives.size} absence cells; "
                     "every other score is in metrics.csv")
        return "\n".join(lines)

    def plot_roc(self):
        """One ROC panel per group, one line per layer."""
        from .plot import plot_roc

        panels = {f"{group}, {self.group_counts[group][1]} features":
                  {layer: self.curves[(group, layer)] for layer in self.layers if (group, layer) in self.curves}
                  for group in self.groups}
        return plot_roc(panels)

    def write(self, out_dir, *, settings=None, inputs=(), points=None, background=None) -> Path:
        """metrics.csv, roc_curves.csv, roc.png, sampling_map.png (the domain, positive,
        absence and excluded cells; ``points`` drawn on top, ``background`` elevation as
        a hillshade), negatives.gpkg, run_info.json and, when given, settings.json.
        ``inputs`` are files to checksum into run_info."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "metrics.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.table[0]))
            writer.writeheader()
            writer.writerows(self.table)
        with open(out / "roc_curves.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["group", "layer", "auc", "fpr", "tpr"])
            for (group, layer), (x, y, auc) in self.curves.items():
                writer.writerows([group, layer, round(auc, 6), round(float(a), 6), round(float(b), 6)]
                                 for a, b in zip(x, y))
        try:
            self.plot_roc().savefig(out / "roc.png", dpi=150)
            self.plot_sampling(points=points, background=background).savefig(out / "sampling_map.png", dpi=150)
        except ImportError:
            warnings.warn("matplotlib is not installed; roc.png and sampling_map.png not written", stacklevel=2)
        try:
            import geopandas as gpd

            gpd.GeoDataFrame({"node": self.negatives},
                             geometry=gpd.points_from_xy(self.grid.x_of_node[self.negatives],
                                                         self.grid.y_of_node[self.negatives]),
                             crs=self.crs).to_file(out / "negatives.gpkg")
        except ImportError:
            warnings.warn("geopandas is not installed; negatives.gpkg not written", stacklevel=2)
        if settings is not None:
            (out / "settings.json").write_text(json.dumps(settings, indent=2, default=str))
        import landlab

        from . import __version__

        info = {
            "created": datetime.now().isoformat(timespec="seconds"),
            "cells": {"grid": int(self.grid.number_of_nodes), "domain": int(self.domain.sum()),
                      "absences": int(self.negatives.size)},
            "features": {g: {"listed": listed, "with_cells_in_domain": scored}
                         for g, (listed, scored) in self.group_counts.items()},
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "landlab": landlab.__version__, "landlab_grid_validation": __version__},
            "inputs": {str(p): {"bytes": Path(p).stat().st_size, "sha256": _sha256(p)} for p in inputs},
        }
        (out / "run_info.json").write_text(json.dumps(info, indent=2))
        return out


def evaluate_features(grid, crs, predicted, features, *, tolerance_cells=1, group_by=None, domain=None,
                      absences: Absences | None = None, bootstrap=1000, seed=42, thresholds=(0.5,),
                      optimal_threshold=True, fpr_levels=(0.05, 0.10), top_shares=(0.10, 0.20)) -> Evaluation:
    """Score node-array layers against mapped features (points, lines or polygons).

    predicted : {name: node array}; cells where any layer has no data are left out.
    features : a GeoDataFrame or a file geopandas can read.
    tolerance_cells : each feature's cells grown by this many cells are its positives.
    group_by : column name(s) of the features; scores for all features, then per
        value, then per nested combination.
    domain : boolean node mask of the cells both classes may come from (None: all).
    absences : the rules for the absence cells (see ``Absences``).
    bootstrap, seed : resamples of the features for the 95% interval of the AUC.

    Every row of the table carries three kinds of score:

    - ranking, threshold-free: ``auc`` with its interval, ``tpr_at_fpr_<level>``
      (the share of positive cells caught when that share of absence cells is
      wrongly flagged, for each of ``fpr_levels``), ``average_precision``, and
      ``captured_in_top_<share>pct_domain`` for each of ``top_shares`` (the
      success-rate reading: its reference is the whole domain, positives
      included, so it is not a point on the ROC curve);
    - cost at a cutoff, for each of ``thresholds`` (``at_0.5``) and, with
      ``optimal_threshold``, at the Youden cutoff (``at_optimal``, with the value
      in ``optimal_threshold``): tp, fp, fn, tn, precision, recall, f1, csi,
      precision as a multiple of the base rate, and the share of the domain flagged;
    - calibration and silence: ``brier`` (for probability layers; NaN otherwise),
      ``positives_at_zero`` (share of positive cells the model scores exactly 0)
      and ``domain_above_zero`` (share of the domain it scores above 0).

    Compare runs only with the same features, domain and absence rules: the
    absence cells never depend on the layers, so with fixed rules every run is
    scored against the same cells, and the interval covers the sampling of the
    features only, not the choice of rules. Precision, f1, csi and Brier do
    depend on the ratio of positives to absences; AUC, TPR at an FPR level and
    the Youden cutoff do not.
    """
    import geopandas as gpd

    absences = absences or Absences()
    gdf = features if isinstance(features, gpd.GeoDataFrame) else gpd.read_file(features)
    layers = {name: np.asarray(values, dtype=float).reshape(-1) for name, values in predicted.items()}
    if not layers:
        raise ValueError("no predicted layers")
    for name, values in layers.items():
        if values.size != grid.number_of_nodes:
            raise ValueError(f"{name} has {values.size} values; the grid has {grid.number_of_nodes} nodes")
    valid = np.all([np.isfinite(v) for v in layers.values()], axis=0)
    dom = valid if domain is None else (valid & np.asarray(domain, dtype=bool))
    if not dom.any():
        raise ValueError("the domain holds no cells")

    own = vector_to_feature_cells(grid, crs, gdf, tolerance_cells=0)
    blocks = vector_to_feature_cells(grid, crs, gdf, tolerance_cells=tolerance_cells) if tolerance_cells else own
    scored = [b[dom[b]] for b in blocks]
    negatives = sample_absences(grid, dom, own, absences, block_cells=blocks)

    rows, curves, counts, groups = [], {}, {}, []
    for label, index in feature_groups(gdf, group_by):
        cells = [scored[i] for i in index if scored[i].size]
        counts[label] = (int(index.size), len(cells))
        if not cells:
            warnings.warn(f"group {label!r}: none of its {index.size} features has cells in the domain", stacklevel=2)
            continue
        groups.append(label)
        yes = np.zeros(grid.number_of_nodes, dtype=bool)
        yes[np.concatenate(cells)] = True
        for name, values in layers.items():
            pos, neg = values[yes], values[negatives]
            auc = roc_auc(np.r_[pos, neg], np.r_[np.ones(pos.size), np.zeros(neg.size)])
            x, y = roc_curve(pos, neg)
            if not np.isclose(area_under(x, y), auc):
                raise RuntimeError(f"ROC area {area_under(x, y)} disagrees with the rank AUC {auc}")
            low, high = bootstrap_auc_interval(values, cells, neg, bootstrap, seed)
            sx, sy = roc_curve(pos, values[dom])
            row = {"layer": name, "group": label, "features": len(cells), "yes_cells": int(yes.sum()),
                   "no_cells": int(negatives.size), "auc": round(auc, 4),
                   "auc_low": round(low, 4), "auc_high": round(high, 4)}
            for level in fpr_levels:
                row[f"tpr_at_fpr_{int(round(level * 100))}pct"] = round(float(np.interp(level, x, y)), 4)
            row["average_precision"] = round(average_precision(np.r_[pos, neg], np.r_[np.ones(pos.size), np.zeros(neg.size)]), 4)
            for share in top_shares:
                row[f"captured_in_top_{int(round(share * 100))}pct_domain"] = round(float(np.interp(share, sx, sy)), 4)
            for t in thresholds:
                row.update(threshold_scores(pos, neg, values[dom], float(t), f"at_{t:g}"))
            if optimal_threshold:
                best = youden_threshold(pos, neg)
                row["optimal_threshold"] = round(best, 6)
                row.update(threshold_scores(pos, neg, values[dom], best, "at_optimal"))
            scores_all = np.r_[pos, neg]
            labels_all = np.r_[np.ones(pos.size), np.zeros(neg.size)]
            row["brier"] = (round(float(np.mean((scores_all - labels_all) ** 2)), 4)
                            if scores_all.min() >= 0 and scores_all.max() <= 1 else np.nan)
            row["positives_at_zero"] = round(float(np.mean(pos == 0)), 4)
            row["domain_above_zero"] = round(float(np.mean(values[dom] > 0)), 4)
            rows.append(row)
            curves[(label, name)] = (x, y, auc)
    if not rows:
        raise ValueError("no feature has cells in the domain")
    return Evaluation(table=rows, curves=curves, groups=groups, layers=list(layers), group_counts=counts,
                      feature_cells=scored, negatives=negatives, domain=dom, grid=grid, crs=crs,
                      excluded=absences.exclude)


def _sha256(path, chunk=1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()
