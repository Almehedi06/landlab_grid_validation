"""Readers: rasters and vector files onto the nodes of a Landlab grid.

A Landlab grid carries no coordinate system, so every reader takes the grid's
CRS explicitly and checks the input against it. Nothing is reprojected or
resampled unless you say how, and every array comes back in Landlab node order
(row 0 = south), ready for ``compare_fields_on_grid``.

rasterio and geopandas are imported inside the functions, so the core package
works without them.
"""

from __future__ import annotations

import warnings

import numpy as np

RESAMPLING = ("nearest", "mode", "bilinear", "average", "max", "min")


def grid_from_raster(path, *, crs=None):
    """A RasterModelGrid with one node at the centre of each raster cell, and its CRS.

    ``crs`` is only needed when the file declares none (an ESRI ASCII grid, for
    example). If the file declares one, a different ``crs`` is an error.
    """
    import rasterio
    from landlab import RasterModelGrid

    with rasterio.open(path) as src:
        t = src.transform
        if t.b != 0 or t.d != 0:
            raise ValueError(f"{path}: rotated rasters are not supported")
        if t.e >= 0:
            raise ValueError(f"{path}: rows must run north to south")
        grid_crs = _resolve_crs(src.crs, crs, path)
        dx, dy = t.a, -t.e
        grid = RasterModelGrid(
            (src.height, src.width),
            xy_spacing=(dx, dy),
            xy_of_lower_left=(src.bounds.left + dx / 2, src.bounds.bottom + dy / 2),
        )
    return grid, grid_crs


def grid_transform(grid):
    """The affine transform of the grid's cells, top-left origin, as rasterio uses."""
    from affine import Affine

    x0, y0 = grid.xy_of_lower_left
    return Affine(grid.dx, 0.0, x0 - grid.dx / 2,
                  0.0, -grid.dy, y0 + (grid.number_of_node_rows - 0.5) * grid.dy)


def raster_to_node_field(grid, crs, path, *, band=1, resampling=None, raster_crs=None,
                         nodata=None) -> np.ndarray:
    """Raster values at the grid's nodes, NaN where there is no data.

    A raster already on this grid is read as it is. Any other raster is
    reprojected onto the grid, and you must say how: 'nearest' or 'mode' for
    classes and yes/no maps, 'average' or 'bilinear' for continuous values.
    ``raster_crs`` is for files that declare no CRS; ``nodata`` overrides the
    file's own no-data value.
    """
    import rasterio
    import rasterio.warp
    from rasterio.crs import CRS
    from rasterio.enums import Resampling

    dst_crs = CRS.from_user_input(crs)
    dst_transform = grid_transform(grid)
    shape = (grid.number_of_node_rows, grid.number_of_node_columns)
    with rasterio.open(path) as src:
        src_crs = _resolve_crs(src.crs, raster_crs, path)
        nd = nodata if nodata is not None else src.nodata
        aligned = (_same_crs(src_crs, dst_crs) and src.shape == shape
                   and src.transform.almost_equals(dst_transform))
        if aligned:
            arr = src.read(band).astype("float64")
            if nd is not None:
                arr[arr == nd] = np.nan
        else:
            if _half_cell_apart(src.transform, src.shape, dst_transform, shape,
                                _same_crs(src_crs, dst_crs)):
                raise ValueError(
                    f"{path} sits exactly half a cell from this grid's nodes. The grid was "
                    "probably built with landlab.io.esri_ascii.load from a file with "
                    "XLLCORNER, which puts nodes on cell corners; build it with "
                    "grid_from_raster instead"
                )
            if resampling not in RESAMPLING:
                raise ValueError(
                    f"{path} is not on this grid, so it must be resampled: pass "
                    "resampling='nearest' or 'mode' (classes, yes/no) or "
                    "'average' or 'bilinear' (continuous values)"
                )
            arr = np.full(shape, np.nan)
            rasterio.warp.reproject(
                rasterio.band(src, band), arr, src_transform=src.transform,
                src_crs=src_crs, src_nodata=nd, dst_transform=dst_transform,
                dst_crs=dst_crs, dst_nodata=np.nan, resampling=Resampling[resampling],
            )
    return arr[::-1].reshape(-1).copy()


def vector_to_node_field(grid, crs, features, *, value=1.0, codes=None, tolerance_cells=0,
                         background=0.0, all_touched=True) -> np.ndarray:
    """Burn points, lines or polygons onto the grid's nodes.

    features : a file geopandas can read, or a GeoDataFrame. Filter it first to
        burn a subset, for example one mechanism.
    value : a number for a yes/no map (burned cells get it, all others
        ``background``), or the name of a column holding class codes. A text
        column needs ``codes``, for example {"LS": 1, "Runoff": 2}.
    tolerance_cells : grow each burned cell into a (2k+1) x (2k+1) block, a
        positional tolerance for remotely mapped features (1 gives 3 x 3 cells).
    all_touched : burn every cell a line or polygon touches, not only cells whose
        centre it covers. A point always burns the cell it falls in.

    Features that lie outside the grid are reported, never dropped silently.
    """
    import geopandas as gpd
    from rasterio.features import rasterize
    from shapely.geometry import box

    gdf = features if isinstance(features, gpd.GeoDataFrame) else gpd.read_file(features)
    if gdf.crs is None:
        raise ValueError("features declare no CRS")
    gdf = gdf.to_crs(crs)
    empty = gdf.geometry.isna() | gdf.geometry.is_empty
    if empty.any():
        warnings.warn(f"{int(empty.sum())} features have no geometry and were skipped",
                      stacklevel=2)
        gdf = gdf[~empty]

    transform = grid_transform(grid)
    shape = (grid.number_of_node_rows, grid.number_of_node_columns)
    extent = box(transform.c, transform.f - shape[0] * grid.dy,
                 transform.c + shape[1] * grid.dx, transform.f)
    outside = ~gdf.intersects(extent)
    if outside.all():
        raise ValueError("none of the features overlap the grid")
    if outside.any():
        warnings.warn(f"{int(outside.sum())} of {len(gdf)} features lie outside the grid "
                      "and were not burned", stacklevel=2)
        gdf = gdf[~outside]

    if isinstance(value, str):
        if tolerance_cells:
            raise ValueError("tolerance_cells works for a single burn value, not class codes")
        vals = gdf[value]
        if codes is not None:
            missing = sorted(set(vals) - set(codes))
            if missing:
                raise ValueError(f"no code for {missing} in column {value!r}")
            vals = vals.map(codes)
        try:
            vals = vals.astype(float).to_numpy()
        except (TypeError, ValueError):
            raise ValueError(f"column {value!r} is not numeric; pass codes={{label: code}}") from None
    else:
        if float(value) == float(background):
            raise ValueError("value must differ from background")
        vals = np.full(len(gdf), float(value))

    burned = rasterize(zip(gdf.geometry, vals, strict=True), out_shape=shape, transform=transform,
                       fill=background, all_touched=all_touched, dtype="float64")
    if tolerance_cells:
        grown = _grow(burned != background, int(tolerance_cells))
        burned = np.where(grown, float(value), float(background))
    return burned[::-1].reshape(-1).copy()


def _grow(mask: np.ndarray, k: int) -> np.ndarray:
    """Square dilation by k cells in every direction, without wrapping at the edges."""
    rows, cols = mask.shape
    padded = np.pad(mask, k)
    out = np.zeros_like(mask)
    for dr in range(2 * k + 1):
        for dc in range(2 * k + 1):
            out |= padded[dr:dr + rows, dc:dc + cols]
    return out


def _half_cell_apart(src_transform, src_shape, dst_transform, dst_shape, same_crs) -> bool:
    """True when two grids of one size and resolution are offset by exactly half a cell."""
    a, b = src_transform, dst_transform
    if not same_crs or src_shape != dst_shape:
        return False
    if not (np.isclose(a.a, b.a) and np.isclose(a.e, b.e)):
        return False
    return bool(np.isclose(abs(a.c - b.c), abs(a.a) / 2)
                and np.isclose(abs(a.f - b.f), abs(a.e) / 2))


def _same_crs(a, b) -> bool:
    return a == b or (a.to_epsg() is not None and a.to_epsg() == b.to_epsg())


def _resolve_crs(file_crs, given, path):
    from rasterio.crs import CRS

    if file_crs is None:
        if given is None:
            raise ValueError(f"{path} declares no CRS; pass it explicitly")
        return CRS.from_user_input(given)
    if given is not None and not _same_crs(file_crs, CRS.from_user_input(given)):
        raise ValueError(f"{path} declares {file_crs}, not {given}")
    return file_crs
