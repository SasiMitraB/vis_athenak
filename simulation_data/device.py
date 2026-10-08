"""
Where a loaded array lives: on the CPU (numpy), on the GPU (cupy), or nowhere.

cupy is optional and only imported when an array is moved to the GPU.
"""

from __future__ import annotations

from enum import Enum

import numpy as np


class Device(Enum):
    """
    Where a field's array is held.  ``Device.NOT_LOADED`` is falsy, so
    ``if field.is_loaded:`` still means "held somewhere".
    """

    NOT_LOADED = "not loaded"
    CPU = "cpu"
    GPU = "gpu"

    def __bool__(self) -> bool:
        return self is not Device.NOT_LOADED

    def __str__(self) -> str:
        return self.value


def as_device(device: str | Device | None) -> Device | None:
    """``"cpu"``/``"gpu"``/``Device`` -> ``Device``; None stays None."""
    if device is None or isinstance(device, Device):
        if device is Device.NOT_LOADED:
            raise ValueError("cannot load to Device.NOT_LOADED")
        return device
    try:
        return {"cpu": Device.CPU, "gpu": Device.GPU, "cuda": Device.GPU}[device.lower()]
    except (KeyError, AttributeError):
        raise ValueError(f"device must be 'cpu' or 'gpu', got {device!r}") from None


def cupy():
    """The cupy module, with a clear error if it is not installed."""
    try:
        import cupy
    except ImportError:
        raise ImportError("GPU arrays need cupy (pip install cupy-cuda12x)") from None
    return cupy


def device_of(array) -> Device:
    """Where ``array`` lives, without importing cupy."""
    if array is None:
        return Device.NOT_LOADED
    if type(array).__module__.split(".")[0] == "cupy":
        return Device.GPU
    return Device.CPU


def to_device(array, device: str | Device | None):
    """``array`` on ``device`` (a no-op if it is already there or device is None)."""
    device = as_device(device)
    if device is None or device_of(array) is device:
        return array
    if device is Device.GPU:
        return cupy().asarray(array)
    return array.get()  # cupy -> numpy


def to_numpy(array) -> np.ndarray:
    """A numpy array, copying from the GPU if needed."""
    return array.get() if device_of(array) is Device.GPU else np.asarray(array)


def array_module(array):
    """numpy or cupy, whichever ``array`` belongs to."""
    return cupy() if device_of(array) is Device.GPU else np


__all__ = ["Device", "array_module", "as_device", "cupy", "device_of", "to_device",
           "to_numpy"]
