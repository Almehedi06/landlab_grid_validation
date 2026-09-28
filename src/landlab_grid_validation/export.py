"""Export helpers for Landlab grid validation fields."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_field_ascii(grid, path: str | Path, field, *, name: str | None = None,
                      clobber: bool = True) -> Path:
    """Write a node field, or node-length values, to ESRI ASCII with Landlab's writer.

    The values are written through a scratch copy of the grid, so the caller's
    grid is never modified and an existing field of the same name is never
    written by mistake. NaN is written as -9999, the NODATA_VALUE Landlab puts
    in the header.
    """
    from landlab import RasterModelGrid
    from landlab.io import esri_ascii

    path = Path(path)
    if isinstance(field, str):
        values = np.asarray(grid.at_node[field], dtype=float)
        name = name or field
    else:
        values = np.asarray(field, dtype=float).reshape(-1)
        name = name or "field"
    if values.size != grid.number_of_nodes:
        raise ValueError(
            f"field has {values.size} values but grid has {grid.number_of_nodes} nodes"
        )
    if path.exists() and not clobber:
        raise FileExistsError(path)

    scratch = RasterModelGrid(
        grid.shape, xy_spacing=(grid.dx, grid.dy), xy_of_lower_left=grid.xy_of_lower_left
    )
    scratch.add_field(name, np.where(np.isfinite(values), values, -9999.0), at="node")
    with path.open("w", encoding="utf-8") as stream:
        esri_ascii.dump(scratch, stream=stream, at="node", name=name)
    return path
