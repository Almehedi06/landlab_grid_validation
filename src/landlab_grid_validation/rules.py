"""Cell rules and absence sampling: where a comparison may draw its cells from.

A rule is a plain dict, so a site config can hold it. It names one source
(``raster``, ``slope_of`` or ``vector``) and what to keep from it:

    {"slope_of": "dem.asc", "min": 17}                                  slope steeper than 17 degrees
    {"raster": "G.tif", "min": 5, "max": 150, "resampling": "nearest"}  5 < G < 150
    {"raster": "ddem.tif", "finite": True, "resampling": "average"}     cells the survey covers
    {"raster": "ddem.tif", "abs_min": 0.5, "buffer_cells": 1}           |change| > 0.5 m, grown by one cell
    {"raster": "severity.tif", "values": [3, 4], "resampling": "nearest"}
    {"vector": "perimeter.shp", "inside": True}

``min`` and ``max`` are exclusive (min < value < max); ``abs_min`` and
``abs_max`` apply to the absolute value; ``values`` keeps the listed codes;
``finite`` keeps cells that have data (``False``: cells without). A cell with no
data never satisfies a value rule. ``buffer_cells`` grows the kept cells by that
many cells, for footprints to exclude. ``resampling``, ``band``, ``nodata`` and
``raster_crs`` go to ``raster_to_node_field``; a raster that declares no CRS is
taken to be in the grid's CRS.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .readers import _grow, raster_to_node_field, vector_to_node_field

RULE_KEYS = frozenset({"raster", "slope_of", "vector", "band", "resampling", "nodata", "raster_crs",
                       "min", "max", "abs_min", "abs_max", "values", "finite", "inside", "all_touched",
                       "buffer_cells"})
VALUE_KEYS = ("min", "max", "abs_min", "abs_max", "values", "finite")


def slope_degrees(grid, elevation) -> np.ndarray:
    """Slope in degrees at every node, from an elevation node array (NaN stays NaN)."""
    z = np.asarray(elevation, dtype=float).reshape(grid.shape)
    gy, gx = np.gradient(z, grid.dy, grid.dx)
    return np.degrees(np.arctan(np.hypot(gx, gy))).reshape(-1)


def rule_mask(grid, crs, rule: dict, *, base_dir=None) -> np.ndarray:
    """The cells that satisfy one rule, as a boolean node array."""
    unknown = sorted(set(rule) - RULE_KEYS)
    if unknown:
        raise ValueError(f"rule {rule}: unknown keys {unknown}; allowed: {sorted(RULE_KEYS)}")
    sources = [key for key in ("raster", "slope_of", "vector") if key in rule]
    if len(sources) != 1:
        raise ValueError(f"rule {rule}: name exactly one of raster, slope_of, vector")
    source = sources[0]
    path = rule[source]
    if base_dir is not None and isinstance(path, (str, Path)) and not Path(path).is_absolute():
        path = Path(base_dir) / path
    given = [key for key in VALUE_KEYS if key in rule]

    if source == "vector":
        if given:
            raise ValueError(f"rule {rule}: a vector rule takes inside, not {given}")
        mask = vector_to_node_field(grid, crs, path, all_touched=rule.get("all_touched", True)) == 1
        if not rule.get("inside", True):
            mask = ~mask
    else:
        if "inside" in rule:
            raise ValueError(f"rule {rule}: inside is for vector rules")
        if not given:
            raise ValueError(f"rule {rule}: say what to keep: min, max, abs_min, abs_max, values or finite")
        values = raster_to_node_field(grid, crs, path, band=rule.get("band", 1),
                                      resampling=rule.get("resampling"),
                                      raster_crs=rule.get("raster_crs", crs), nodata=rule.get("nodata"))
        if source == "slope_of":
            values = slope_degrees(grid, values)
        has_data = np.isfinite(values)
        if rule.get("finite") is False:
            if len(given) > 1:
                raise ValueError(f"rule {rule}: finite: false cannot be combined with value limits")
            return ~has_data
        mask = has_data.copy()
        if "min" in rule:
            mask &= values > rule["min"]
        if "max" in rule:
            mask &= values < rule["max"]
        if "abs_min" in rule:
            mask &= np.abs(values) > rule["abs_min"]
        if "abs_max" in rule:
            mask &= np.abs(values) < rule["abs_max"]
        if "values" in rule:
            mask &= np.isin(values, np.asarray(rule["values"], dtype=float))

    if rule.get("buffer_cells"):
        mask = _grow(mask.reshape(grid.shape), int(rule["buffer_cells"])).reshape(-1)
    return mask


def mask_from_rules(grid, crs, rules, *, how: str = "all", base_dir=None) -> np.ndarray:
    """Cells that satisfy all the rules (``how="all"``) or any of them (``how="any"``).

    No rules at all keeps every cell under "all" and no cell under "any".
    """
    if how not in ("all", "any"):
        raise ValueError("how must be 'all' or 'any'")
    masks = [rule_mask(grid, crs, rule, base_dir=base_dir) for rule in rules]
    if not masks:
        return np.full(grid.number_of_nodes, how == "all")
    return np.logical_and.reduce(masks) if how == "all" else np.logical_or.reduce(masks)


@dataclass(frozen=True)
class Absences:
    """How absence (no) cells are drawn from the domain.

    keep : boolean node mask; rules for absence cells only (for example 5 < G < 150).
    exclude : boolean node mask of cells to drop, such as the footprint the event
        changed, already grown by its buffer.
    min_distance_m : no absence cell closer than this to any feature's own cells.
    n : "all" for every qualifying cell, or a number drawn at random with ``seed``.
    stratify : class codes per node (a categorical raster on the grid). The ``n``
        cells are drawn per class in the proportions the positive cells have.
    """

    keep: np.ndarray | None = None
    exclude: np.ndarray | None = None
    min_distance_m: float = 0.0
    n: int | str = "all"
    seed: int = 0
    stratify: np.ndarray | None = None


def sample_absences(grid, domain, feature_cells, absences: Absences | None = None, *,
                    block_cells=None) -> np.ndarray:
    """Node indices of the absence cells: the domain minus every feature's block,
    minus ``absences.exclude``, limited to ``absences.keep`` and to cells at least
    ``min_distance_m`` from the features' own cells; then all of them or ``n`` at random.

    ``feature_cells`` are the features' own cells (no tolerance); ``block_cells``
    (default: the same) are the cells scored as positive, which are never absences.
    """
    from scipy.ndimage import distance_transform_edt

    absences = absences or Absences()
    pool = np.asarray(domain, dtype=bool).copy()
    blocks = feature_cells if block_cells is None else block_cells
    for cells in blocks:
        pool[np.asarray(cells, dtype=int)] = False
    if absences.keep is not None:
        pool &= np.asarray(absences.keep, dtype=bool)
    if absences.exclude is not None:
        pool &= ~np.asarray(absences.exclude, dtype=bool)
    if absences.min_distance_m > 0:
        at = np.zeros(grid.number_of_nodes, dtype=bool)
        for cells in feature_cells:
            at[np.asarray(cells, dtype=int)] = True
        distance = distance_transform_edt(~at.reshape(grid.shape), sampling=(grid.dy, grid.dx)).reshape(-1)
        pool &= distance >= absences.min_distance_m
    candidates = np.flatnonzero(pool)
    if candidates.size == 0:
        raise ValueError("no absence cells qualify; loosen the rules")

    if absences.n == "all":
        if absences.stratify is not None:
            raise ValueError("stratify needs a number of absence cells, n")
        return candidates
    n = int(absences.n)
    if n > candidates.size:
        raise ValueError(f"n = {n} absence cells asked for, but only {candidates.size} qualify")
    rng = np.random.default_rng(absences.seed)
    if absences.stratify is None:
        return np.sort(rng.choice(candidates, n, replace=False))

    classes = np.asarray(absences.stratify)
    positive = np.unique(np.concatenate([np.asarray(c, dtype=int) for c in blocks]))
    kinds, counts = np.unique(classes[positive], return_counts=True)
    chosen = []
    for kind, count in zip(kinds, counts):
        pool_k = candidates[classes[candidates] == kind]
        want = int(round(n * count / counts.sum()))
        if want > pool_k.size:
            warnings.warn(f"stratify: class {kind} has {pool_k.size} candidate cells, {want} wanted", stacklevel=2)
            want = pool_k.size
        if want:
            chosen.append(rng.choice(pool_k, want, replace=False))
    if not chosen:
        raise ValueError("stratify: no absence cells in the classes the positives occupy")
    return np.sort(np.concatenate(chosen))
