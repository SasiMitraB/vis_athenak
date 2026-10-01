"""A single, lazily loaded output snapshot."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .readers import read_file, read_time


class Frame:
    """
    One output snapshot, possibly spread over several files (one per output
    block).  Nothing is read from disk until a field, ``data``, or ``time``
    is accessed.

    ``frame["dens"]`` returns an array of shape ``(nx3, nx2, nx1)``.  The
    coordinate arrays ``x1f, x1v, x2f, ...`` are available the same way.
    """

    def __init__(self, number: int, paths: dict[str, Path], dtype=None):
        self.number = number
        self.paths = paths  # {output id: path to that output's file}
        self._dtype = dtype
        self._data: dict | None = None
        self._time: float | None = None

    @property
    def time(self) -> float:
        """Simulation time in code units (read from the file header only)."""
        if self._time is None:
            if self._data is not None:
                self._time = float(self._data["Time"])
            else:
                self._time = read_time(next(iter(self.paths.values())))
        return self._time

    @property
    def data(self) -> dict:
        """All variables and coordinates of this frame, loaded on first use."""
        if self._data is None:
            data = {}
            for path in self.paths.values():
                data.update(read_file(path, self._dtype))
            self._data = data
        return self._data

    @property
    def is_loaded(self) -> bool:
        return self._data is not None

    def unload(self) -> None:
        """Drop the cached field data; it is re-read on the next access."""
        self._data = None

    def keys(self):
        return self.data.keys()

    def __getitem__(self, name: str) -> np.ndarray:
        try:
            return self.data[name]
        except KeyError:
            raise KeyError(
                f"{name!r} not in frame {self.number}; "
                f"available: {sorted(self.data)}"
            ) from None

    def __contains__(self, name: str) -> bool:
        return name in self.data

    def __repr__(self) -> str:
        state = "loaded" if self.is_loaded else "not loaded"
        return f"<Frame {self.number}  outputs={list(self.paths)}  {state}>"
