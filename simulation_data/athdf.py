"""
Fast reads of single variables from .athdf files.

``athena_read.athdf`` rebuilds the coordinate arrays and re-reads the file
metadata on every call, which costs about 1 ms per call on a small grid,
roughly ten times the h5py read of the variable itself.  Here the block
layout of a file is read once (`AthdfLayout`) and reused, so each further
variable costs one h5py read plus copying blocks into place.

Only the default merge of ``athena_read.athdf`` is reproduced: every
meshblock is prolongated (repeated) up to the finest level present and
placed at its logical location.  Files it does not handle (ghost zones,
slices or sums along an extended dimension) report ``supported = False``,
and the caller should fall back to ``athena_read.athdf``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass(frozen=True)
class AthdfLayout:
    """Where each variable and meshblock of one .athdf file goes."""

    variables: dict[str, tuple[str, int]]  # name -> (dataset, index in dataset)
    levels: np.ndarray                     # (nmb,) refinement level of each block
    locations: np.ndarray                  # (nmb, 3) logical location (x1, x2, x3)
    block_size: tuple[int, int, int]       # (nx1, nx2, nx3) cells per block
    shape: tuple[int, int, int]            # (nx3, nx2, nx1) of the merged array
    max_level: int
    dtype: np.dtype                        # dtype of the stored variables
    supported: bool                        # False: use athena_read.athdf instead

    @classmethod
    def read(cls, f: h5py.File) -> AthdfLayout:
        names = [v.decode("ascii", "replace") for v in f.attrs["VariableNames"]]
        datasets = [d.decode("ascii", "replace") for d in f.attrs["DatasetNames"]]
        variables, start = {}, 0
        for dataset, n in zip(datasets, f.attrs["NumVariables"]):
            for i in range(int(n)):
                variables[names[start + i]] = (dataset, i)
            start += int(n)

        max_level = int(f.attrs["MaxLevel"])
        block_size = tuple(int(n) for n in f.attrs["MeshBlockSize"])
        root = [int(n) for n in f.attrs["RootGridSize"]]
        nx = [r * 2**max_level if b > 1 else 1 for b, r in zip(block_size, root)]

        # athena_read treats these specially; leave them to it.
        slice_or_sum = any(b == 1 and r > 1 for b, r in zip(block_size, root))
        ghosts = f["x1v"][:].min() < f.attrs["RootGridX1"][0]

        return cls(
            variables=variables,
            levels=f["Levels"][:],
            locations=f["LogicalLocations"][:],
            block_size=block_size,
            shape=(nx[2], nx[1], nx[0]),
            max_level=max_level,
            dtype=f[datasets[0]].dtype.newbyteorder("="),
            supported=not (slice_or_sum or ghosts),
        )


def read_layout(path: Path) -> AthdfLayout:
    with h5py.File(path, "r") as f:
        return AthdfLayout.read(f)


def read_variables(path: Path, names: list[str], layout: AthdfLayout,
                   dtype=None) -> dict[str, np.ndarray]:
    """
    Read variables of a .athdf file, merged into arrays of shape
    ``layout.shape`` as ``athena_read.athdf`` would.  `layout` must come
    from the same file and have ``supported`` set.
    """
    unknown = [n for n in names if n not in layout.variables]
    if unknown:
        raise KeyError(f"{unknown} not in {path}; it has {list(layout.variables)}")

    with h5py.File(path, "r") as f:
        blocks = {n: f[layout.variables[n][0]][layout.variables[n][1]] for n in names}
    expected = (len(layout.levels), *layout.block_size[::-1])
    for n, arr in blocks.items():
        if arr.shape != expected:
            raise ValueError(f"{n!r} in {path} has shape {arr.shape}, expected {expected}")

    dtype = layout.dtype if dtype is None else np.dtype(dtype)
    out = {n: np.zeros(layout.shape, dtype=dtype) for n in names}
    extended = [n > 1 for n in layout.shape]  # (x3, x2, x1)
    bs = layout.block_size[::-1]              # (nx3, nx2, nx1)

    for b, (level, loc) in enumerate(zip(layout.levels, layout.locations)):
        s = 2 ** (layout.max_level - int(level))
        loc = loc[::-1]  # (x3, x2, x1), matching the array axes
        dest = tuple(
            slice(loc[a] * bs[a] * s, (loc[a] + 1) * bs[a] * s) if extended[a]
            else slice(0, 1)
            for a in range(3)
        )
        for n in names:
            block = blocks[n][b]
            if s > 1:
                for a in range(3):
                    if extended[a]:
                        block = np.repeat(block, s, axis=a)
            out[n][dest] = block
    return out
