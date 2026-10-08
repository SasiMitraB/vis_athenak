"""A single variable of a Frame, lazily loaded."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .device import Device, device_of, to_device, to_numpy

if TYPE_CHECKING:
    from .frame import Frame


class Field:
    """
    One variable (``dens``, ``velx``, ``bcc1``, ``s_00``, ...) of a Frame.

    Nothing is read until ``data`` is accessed.  For .athdf files only this
    variable is read; .bin files can only be read whole, so loading one field
    of a .bin file loads every field stored in that file.

    The array is a numpy array on the CPU, or a cupy array after
    ``to_gpu()`` (or when the frame loads to ``device="gpu"``).
    ``is_loaded`` says which: ``Device.CPU``, ``Device.GPU`` or
    ``Device.NOT_LOADED`` (falsy).

    A Field can be passed to numpy directly: ``np.log10(frame.fields["dens"])``
    (a GPU field is copied back to the CPU for this).
    """

    def __init__(self, name: str, output: str, path: Path, frame: Frame):
        self.name = name
        self.output = output  # id of the output block the field was written by
        self.path = path      # file the field is read from
        self._frame = frame
        self._data = None     # numpy or cupy array once loaded

    @property
    def data(self):
        """The field as an array of shape ``(nx3, nx2, nx1)``, read on first use."""
        if self._data is None:
            self._frame.load([self])
        return self._data

    @property
    def is_loaded(self) -> Device:
        """``Device.CPU``, ``Device.GPU`` or ``Device.NOT_LOADED`` (falsy)."""
        return device_of(self._data)

    def to(self, device: str | Device) -> Field:
        """Move the array to ``"cpu"`` or ``"gpu"``, reading it first if needed."""
        if self._data is None:
            self._frame.load([self], device=device)
        else:
            self._data = to_device(self._data, device)
        return self

    def to_gpu(self) -> Field:
        return self.to(Device.GPU)

    def to_cpu(self) -> Field:
        return self.to(Device.CPU)

    def unload(self) -> None:
        """Drop the cached array; it is re-read on the next access."""
        self._data = None

    def __array__(self, dtype=None, copy=None):
        return np.asarray(to_numpy(self.data), dtype=dtype)

    def __repr__(self) -> str:
        state = "not loaded" if not self.is_loaded else f"loaded on {self.is_loaded}"
        return f"<Field {self.name!r}  output={self.output!r}  {state}>"
