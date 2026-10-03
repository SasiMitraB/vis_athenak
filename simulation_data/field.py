"""A single variable of a Frame, lazily loaded."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .frame import Frame


class Field:
    """
    One variable (``dens``, ``velx``, ``bcc1``, ``s_00``, ...) of a Frame.

    Nothing is read until ``data`` is accessed.  For .athdf files only this
    variable is read; .bin files can only be read whole, so loading one field
    of a .bin file loads every field stored in that file.

    A Field can be passed to numpy directly: ``np.log10(frame.fields["dens"])``.
    """

    def __init__(self, name: str, output: str, path: Path, frame: Frame):
        self.name = name
        self.output = output  # id of the output block the field was written by
        self.path = path      # file the field is read from
        self._frame = frame
        self._data: np.ndarray | None = None

    @property
    def data(self) -> np.ndarray:
        """The field as an array of shape ``(nx3, nx2, nx1)``, read on first use."""
        if self._data is None:
            self._frame.load([self])
        return self._data

    @property
    def is_loaded(self) -> bool:
        return self._data is not None

    def unload(self) -> None:
        """Drop the cached array; it is re-read on the next access."""
        self._data = None

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.data, dtype=dtype)

    def __repr__(self) -> str:
        state = "loaded" if self.is_loaded else "not loaded"
        return f"<Field {self.name!r}  output={self.output!r}  {state}>"
