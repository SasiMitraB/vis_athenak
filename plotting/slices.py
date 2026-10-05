"""
2D slices: cutting a slice out of a frame, and drawing it on an axes.

Both are plain functions with no file I/O or global settings, so they can be
combined freely, e.g. one panel per run when comparing cooling prescriptions:

    fig, axes = plt.subplots(1, 2)
    for ax, frame in zip(axes, frames):
        x, y, data = take_slice(frame, get(frame, "temp", params), axis="z")
        im = draw_slice(ax, x, y, data, cmap="inferno", norm="log", vmin=1e4, vmax=1e7)
"""

from __future__ import annotations

import numpy as np
from matplotlib import colors

# Slice normal -> (array axis cut, horizontal coordinate, vertical coordinate).
# Arrays are (nx3, nx2, nx1), so axis 0 is x3 (z).
_GEOMETRY = {
    "z": (0, "x1", "x2"),
    "y": (1, "x1", "x3"),
    "x": (2, "x2", "x3"),
}
AXIS_LABELS = {"x1": "x", "x2": "y", "x3": "z"}


def take_slice(frame, data: np.ndarray, axis: str = "z", position: float | None = None):
    """
    Slice ``data`` (shape ``(nx3, nx2, nx1)``) normal to ``axis``.

    ``position`` is a coordinate in code units along ``axis``; the cell
    containing it is used.  Defaults to the middle of the domain.  For 2D
    runs use ``axis="z"``.

    Returns ``(x_faces, y_faces, slice)`` ready for ``pcolormesh``, with the
    slice of shape ``(len(y_faces) - 1, len(x_faces) - 1)``.
    """
    if axis not in _GEOMETRY:
        raise ValueError(f"axis must be one of {list(_GEOMETRY)}, got {axis!r}")
    cut, h, v = _GEOMETRY[axis]
    normal = {0: "x3", 1: "x2", 2: "x1"}[cut]

    faces = frame[f"{normal}f"]
    if position is None:
        index = data.shape[cut] // 2
    else:
        if not faces[0] <= position <= faces[-1]:
            raise ValueError(
                f"{AXIS_LABELS[normal]} = {position} outside [{faces[0]}, {faces[-1]}]"
            )
        index = min(int(np.searchsorted(faces, position, side="right")) - 1,
                    data.shape[cut] - 1)

    plane = np.take(data, index, axis=cut)  # rows follow v, columns follow h
    return frame[f"{h}f"], frame[f"{v}f"], plane


def draw_slice(ax, x, y, data, cmap="viridis", norm=None, vmin=None, vmax=None,
               colorbar=True, label=None, axis="z", length_units=None):
    """
    Draw a slice from `take_slice` on ``ax``; returns the QuadMesh.

    ``norm`` is "log" (non-positive values masked) or None/"linear".
    ``length_units`` is only used in the axis labels; scale ``x`` and ``y``
    to match.
    """
    if norm == "log":
        data = np.ma.masked_less_equal(data, 0)
        mpl_norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    elif norm in (None, "linear"):
        mpl_norm = colors.Normalize(vmin=vmin, vmax=vmax)
    else:
        raise ValueError(f"norm must be 'log' or None, got {norm!r}")

    mesh = ax.pcolormesh(x, y, data, cmap=cmap, norm=mpl_norm, shading="flat")
    ax.set_aspect("equal")
    _, h, v = _GEOMETRY[axis]
    suffix = f" [{length_units}]" if length_units and length_units != "code" else ""
    ax.set_xlabel(AXIS_LABELS[h] + suffix)
    ax.set_ylabel(AXIS_LABELS[v] + suffix)
    if colorbar:
        ax.figure.colorbar(mesh, ax=ax, label=label)
    return mesh
