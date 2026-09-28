"""Export helpers for Landlab grid validation fields."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_field_ascii(grid, path: str | Path, field, *, name: str = "field", clobber: bool = True):
    """Write a node field to ESRI ASCII using Landlab's writer.

    This is intentionally a thin wrapper around Landlab I/O so ASC stays an
    exchange format rather than the internal representation.
    """
    path = Path(path)
    values = np.asarray(grid.at_node[field] if isinstance(field, str) else field)
    if values.size != grid.number_of_nodes:
        raise ValueError(
            f"field has {values.size} values but grid has {grid.number_of_nodes} nodes"
        )
    if path.exists() and not clobber:
        raise FileExistsError(path)

    if name not in grid.at_node:
        grid.add_field(name, values, at="node", clobber=True)

    try:
        from landlab.io.esri_ascii import dump

        with path.open("w", encoding="utf-8") as stream:
            dump(grid, stream=stream, at="node", name=name)
    except ImportError:  # pragma: no cover - older Landlab versions
        from landlab.io import write_esri_ascii

        write_esri_ascii(str(path), grid, names=[name], clobber=clobber)
    return path
