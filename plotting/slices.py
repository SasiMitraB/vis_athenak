"""
2D slices: cutting a slice out of a frame, and drawing it on an axes.

None of these use global settings, so they can be combined freely, e.g. one
panel per run when comparing cooling prescriptions:

    fig, axes = plt.subplots(1, 2)
    for ax, frame in zip(axes, frames):
        plane = Plane(frame, axis="z")                 # midplane
        T = get(plane, "temp", params, units="K")      # computed on the 2D slice only
        draw_slice(ax, plane.x, plane.y, T, cmap="inferno", norm="log", vmin=1e4, vmax=1e7)

`Plane` (``simulation_data.plane.FramePlane``) reads each raw field's slice
the first time it is used, from only the meshblocks crossing the slice, so
neither reading nor derived quantities touch the rest of the box.
"""

from __future__ import annotations

import numpy as np
from matplotlib import colors

if "." in __package__:  # imported as vis_athenak.plotting
    from ..simulation_data.plane import FramePlane as Plane
else:  # run from the repo root
    from simulation_data.plane import FramePlane as Plane

__all__ = ["AXIS_LABELS", "Plane", "draw_slice", "slice_index", "take_slice"]

# Slice normal -> (array axis cut, horizontal coordinate, vertical coordinate).
# Arrays are (nx3, nx2, nx1), so axis 0 is x3 (z).
_GEOMETRY = {
    "z": (0, "x1", "x2"),
    "y": (1, "x1", "x3"),
    "x": (2, "x2", "x3"),
}
AXIS_LABELS = {"x1": "x", "x2": "y", "x3": "z"}


def _check_axis(axis: str) -> None:
    if axis not in _GEOMETRY:
        raise ValueError(f"axis must be one of {list(_GEOMETRY)}, got {axis!r}")


def slice_index(frame, axis: str = "z", position: float | None = None) -> int:
    """
    Index along ``axis`` of the cell containing ``position`` (code units);
    the middle cell if ``position`` is None.
    """
    _check_axis(axis)
    normal = {0: "x3", 1: "x2", 2: "x1"}[_GEOMETRY[axis][0]]
    faces = frame[f"{normal}f"]
    ncells = len(faces) - 1
    if position is None:
        return ncells // 2
    if not faces[0] <= position <= faces[-1]:
        raise ValueError(
            f"{AXIS_LABELS[normal]} = {position} outside [{faces[0]}, {faces[-1]}]"
        )
    return min(int(np.searchsorted(faces, position, side="right")) - 1, ncells - 1)


def take_slice(frame, data: np.ndarray, axis: str = "z", position: float | None = None):
    """
    Slice an already computed array ``data`` (shape ``(nx3, nx2, nx1)``)
    normal to ``axis``; see `Plane` to slice before computing instead.

    ``position`` is a coordinate in code units along ``axis``; the cell
    containing it is used.  Defaults to the middle of the domain.  For 2D
    runs use ``axis="z"``.

    Returns ``(x_faces, y_faces, slice)`` ready for `draw_slice`, with the
    slice of shape ``(len(y_faces) - 1, len(x_faces) - 1)``.
    """
    _check_axis(axis)
    cut, h, v = _GEOMETRY[axis]
    index = slice_index(frame, axis, position)
    plane = np.take(data, index, axis=cut)  # rows follow v, columns follow h
    return frame[f"{h}f"], frame[f"{v}f"], plane


def _uniform(faces) -> bool:
    widths = np.diff(np.asarray(faces, dtype=float))
    return widths.size > 0 and np.allclose(widths, widths[0], rtol=1e-6, atol=0)


def draw_slice(ax, x, y, data, cmap="viridis", norm=None, vmin=None, vmax=None,
               colorbar=True, label=None, axis="z", length_units=None):
    """
    Draw a slice on ``ax``; ``x`` and ``y`` are cell faces (as from `Plane`
    or `take_slice`).  Returns the image, for a colorbar.

    ``norm`` is "log" (non-positive values masked) or None/"linear".
    ``length_units`` is only used in the axis labels; scale ``x`` and ``y``
    to match.  Uniform grids are drawn with imshow (one image, fast); other
    grids with pcolormesh (one quad per cell).
    """
    if norm == "log":
        data = np.ma.masked_less_equal(data, 0)
        mpl_norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    elif norm in (None, "linear"):
        mpl_norm = colors.Normalize(vmin=vmin, vmax=vmax)
    else:
        raise ValueError(f"norm must be 'log' or None, got {norm!r}")

    if _uniform(x) and _uniform(y):
        mesh = ax.imshow(data, origin="lower", extent=(x[0], x[-1], y[0], y[-1]),
                         cmap=cmap, norm=mpl_norm, aspect="equal", interpolation="nearest")
    else:
        mesh = ax.pcolormesh(x, y, data, cmap=cmap, norm=mpl_norm, shading="flat")
        ax.set_aspect("equal")
    _, h, v = _GEOMETRY[axis]
    suffix = f" [{length_units}]" if length_units and length_units != "code" else ""
    ax.set_xlabel(AXIS_LABELS[h] + suffix)
    ax.set_ylabel(AXIS_LABELS[v] + suffix)
    if colorbar:
        ax.figure.colorbar(mesh, ax=ax, label=label)
    return mesh
