"""
Particle outputs of an AthenaK run: lazy, per-column access and tracking by tag.

    from simulation_data.particles import ParticleData

    pdata = ParticleData("runs/trml/pbin", basename="TRML")
    len(pdata)                     # number of particle output files
    pf = pdata[-1]                 # one file; only its header is read
    pf.time, pf.n, pf.names        # time, particle count, columns
    tags = pf["tag"]               # one column, read from disk only now
    tr = pdata.track([3, 17, 42], ["x", "y", "temp"])
    tr["x"].shape                  # (n_files, 3), NaN where a particle is missing

Two formats are read:

``.prtclbin``  the group's binary particle output (see
    ``data_processing/read_prtcl_bin.py``): an int64/float64 header with the
    names of the grid fields sampled at the particles, then ``nrdata``
    float64 columns, ``nidata`` int32 columns and ``ngriddata`` float64
    columns.  The real columns are named ``x, y, z, vx, vy, vz`` and the
    integer ones ``gid, tag, lastmove, lastlevel`` unless other names are
    passed (``rdata_names`` / ``idata_names``), e.g. if a pusher stores
    fewer or differently ordered columns.
``.part.vtk``  AthenaK's ``pvtk`` output: positions and the ``gid``/``ptag``
    scalars (big-endian float32).  ``ptag`` is also available as ``tag``.

Columns are memory-mapped, so reading a few particles of a file with
millions only touches the bytes needed, and nothing stays in memory between
calls.  Particles migrate between ranks and meshblocks, so their order
changes from file to file: always match them by ``tag``.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np

DEFAULT_RDATA_NAMES = ("x", "y", "z", "vx", "vy", "vz")
DEFAULT_IDATA_NAMES = ("gid", "tag", "lastmove", "lastlevel")

# <stem>.<NNNNN>.<ext>; the stem is "<basename>" or "<basename>.<id>".
_FILE_RE = re.compile(r"^(?P<stem>.+)\.(?P<num>\d+)\.(?P<ext>prtclbin|part\.vtk)$")
PRTCLBIN_MAGIC = 43


class ParticleFile:
    """One particle output file; the header is read on first use, columns on request."""

    def __init__(self, path: str | Path, number: int,
                 rdata_names=DEFAULT_RDATA_NAMES, idata_names=DEFAULT_IDATA_NAMES):
        self.path = Path(path)
        self.number = number
        self._rdata_names = tuple(rdata_names)
        self._idata_names = tuple(idata_names)
        self._columns: dict[str, tuple[int, str]] | None = None  # name -> (offset, dtype)
        self._n = 0
        self._time = np.nan
        self._cycle = -1

    # ── Header ───────────────────────────────────────────────────────────

    def _read_header(self) -> None:
        if self._columns is not None:
            return
        if self.path.name.endswith(".prtclbin"):
            self._read_prtclbin_header()
        else:
            self._read_pvtk_header()

    def _read_prtclbin_header(self) -> None:
        with open(self.path, "rb") as f:
            ints = np.fromfile(f, dtype="<i8", count=5)
            reals = np.fromfile(f, dtype="<f8", count=3)
            if len(ints) < 5 or ints[0] != PRTCLBIN_MAGIC:
                raise ValueError(f"{self.path}: not a .prtclbin file (magic number "
                                 f"{ints[0] if len(ints) else None}, expected {PRTCLBIN_MAGIC})")
            n, nr, ni, ng = (int(v) for v in ints[1:])
            grid_names = [f.read(16).split(b"\x00", 1)[0].decode("utf-8", "replace").strip()
                          for _ in range(ng)]
            offset = f.tell()
        self._n, self._time, self._cycle = n, float(reals[0]), int(reals[2])

        columns = {}
        for i in range(nr):
            name = self._rdata_names[i] if i < len(self._rdata_names) else f"rdata_{i}"
            columns[name] = (offset, "<f8")
            offset += 8 * n
        for i in range(ni):
            name = self._idata_names[i] if i < len(self._idata_names) else f"idata_{i}"
            columns[name] = (offset, "<i4")
            offset += 4 * n
        for name in grid_names:
            # A grid field named like a particle column is kept under "grid/<name>".
            columns[name if name not in columns else f"grid/{name}"] = (offset, "<f8")
            offset += 8 * n

        size = self.path.stat().st_size
        if size != offset:
            raise ValueError(
                f"{self.path}: {size} bytes, but its header ({n} particles, {nr} real, "
                f"{ni} integer, {ng} grid columns) implies {offset}; the file is "
                "truncated or not in the format read_prtcl_bin.py describes")
        self._columns = columns

    def _read_pvtk_header(self) -> None:
        with open(self.path, "rb") as f:
            raw = f.read()  # pvtk files hold only positions and two scalars
        m = re.search(rb"time=\s*([-+0-9.eE]+).*?cycle=\s*(\d+)", raw[:1000])
        if m:
            self._time, self._cycle = float(m.group(1)), int(m.group(2))
        m = re.search(rb"POINTS (\d+) float\n", raw)
        if not m:
            raise ValueError(f"{self.path}: no POINTS section; not an AthenaK pvtk file")
        n = int(m.group(1))
        columns = {}
        start = m.end()
        for i, name in enumerate(("x", "y", "z")):
            # x, y, z interleaved: a strided view is made in _column.
            columns[name] = (start + 4 * i, ">f4/3")
        offset = start + 12 * n
        for m in re.finditer(rb"SCALARS (\w+) float\nLOOKUP_TABLE default\n", raw[offset:]):
            name = m.group(1).decode()
            columns[name] = (offset + m.end(), ">f4")
        if "ptag" in columns:
            columns.setdefault("tag", columns["ptag"])
        self._n, self._columns = n, columns

    @property
    def n(self) -> int:
        """Number of particles in the file."""
        self._read_header()
        return self._n

    @property
    def time(self) -> float:
        """Simulation time (code units); NaN if the file does not record it."""
        self._read_header()
        return self._time

    @property
    def cycle(self) -> int:
        self._read_header()
        return self._cycle

    @property
    def names(self) -> list[str]:
        """Columns of the file: positions, velocities, ids and sampled grid fields."""
        self._read_header()
        return list(self._columns)

    # ── Columns ──────────────────────────────────────────────────────────

    def _column(self, name: str) -> np.ndarray:
        """Memory-mapped view of one column (nothing read yet)."""
        self._read_header()
        if name not in self._columns:
            raise KeyError(f"{name!r} not in {self.path.name}; columns: {self.names}")
        offset, dtype = self._columns[name]
        if self._n == 0:
            return np.empty(0)
        if dtype == ">f4/3":
            mm = np.memmap(self.path, dtype=">f4", mode="r", offset=offset,
                           shape=(3 * self._n - 2,))
            return mm[::3]
        return np.memmap(self.path, dtype=dtype, mode="r", offset=offset, shape=(self._n,))

    def read(self, names, index=None) -> dict[str, np.ndarray]:
        """
        Columns ``names`` as native float64 / int64 arrays, of every particle or
        only those at positions ``index`` (an integer array, in file order).
        """
        out = {}
        for name in names:
            col = self._column(name)
            values = col if index is None else col[index]
            kind = np.int64 if col.dtype.kind in "iu" else np.float64
            out[name] = np.array(values, dtype=kind)
        return out

    def __getitem__(self, name: str) -> np.ndarray:
        return self.read([name])[name]

    def tags(self) -> np.ndarray:
        """Particle tags (int64), in file order."""
        tags = self["tag"]
        return tags.astype(np.int64)

    def find(self, tags) -> tuple[np.ndarray, np.ndarray]:
        """
        Where the particles with ``tags`` are in this file:
        ``(index, which)`` with ``tags[which[k]]`` at file position ``index[k]``.
        Tags not present are left out.
        """
        wanted = np.asarray(tags, dtype=np.int64)
        order = np.argsort(wanted, kind="stable")
        sorted_tags = wanted[order]
        file_tags = self.tags()
        if sorted_tags.size == 0 or file_tags.size == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        pos = np.searchsorted(sorted_tags, file_tags)
        pos_c = np.minimum(pos, sorted_tags.size - 1)
        hit = sorted_tags[pos_c] == file_tags
        index = np.flatnonzero(hit)
        return index, order[pos_c[index]]

    def __repr__(self) -> str:
        return f"<ParticleFile {self.path.name}  n={self.n}  t={self.time:g}>"


class ParticleData:
    """
    The particle output files of one run, in file-number order.

    Parameters
    ----------
    folder : str or Path, or a list of them
        Searched recursively for ``*.prtclbin`` and ``*.part.vtk`` files.
    basename : str, optional
        The run's <job> basename; picks this run's files when the folder
        holds several sets.
    stem : str, optional
        Exact file name stem (``"TRML"``, ``"TRML.prtcl"``, ...) if
        ``basename`` is not enough to tell sets apart.
    rdata_names, idata_names : sequence of str, optional
        Names of the real / integer columns of ``.prtclbin`` files.
    """

    def __init__(self, folder, basename: str | None = None, stem: str | None = None,
                 rdata_names=DEFAULT_RDATA_NAMES, idata_names=DEFAULT_IDATA_NAMES):
        folders = [Path(folder)] if isinstance(folder, (str, Path)) else [Path(f) for f in folder]
        groups: dict[tuple[str, str], dict[int, Path]] = {}
        for root in folders:
            if not root.is_dir():
                continue
            for pattern in ("*.prtclbin", "*.part.vtk"):
                for path in root.rglob(pattern):
                    m = _FILE_RE.match(path.name)
                    if m:
                        key = (m.group("stem"), m.group("ext"))
                        groups.setdefault(key, {}).setdefault(int(m.group("num")), path)
        if not groups:
            raise FileNotFoundError(
                "no particle files (*.prtclbin, *.part.vtk) under "
                + ", ".join(str(f) for f in folders))

        keys = list(groups)
        if stem is not None:
            keys = [k for k in keys if k[0] == stem]
        elif basename is not None and len(keys) > 1:
            mine = [k for k in keys if k[0] == basename or k[0].startswith(basename + ".")]
            keys = mine or keys
        if len(keys) > 1:
            # Prefer the binary output, which carries the sampled grid fields.
            binary = [k for k in keys if k[1] == "prtclbin"]
            keys = binary if len(binary) == 1 else keys
        if len(keys) != 1:
            found = sorted(f"{s}.NNNNN.{e}" for s, e in groups)
            raise ValueError(f"{'no' if not keys else 'several'} particle file sets match: "
                             f"found {found}; pass stem=... to pick one")
        self.stem, self.ext = keys[0]
        files = groups[keys[0]]
        self.files = [ParticleFile(files[n], n, rdata_names, idata_names)
                      for n in sorted(files)]
        self._times: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index):
        return self.files[index]

    def __iter__(self):
        return iter(self.files)

    @property
    def numbers(self) -> list[int]:
        return [f.number for f in self.files]

    @property
    def times(self) -> np.ndarray:
        """Time of every file (header reads only, cached)."""
        if self._times is None:
            self._times = np.array([f.time for f in self.files])
        return self._times

    def track(self, tags, names, files=None) -> dict[str, np.ndarray]:
        """
        Columns ``names`` of the particles ``tags`` in every file (or in
        ``files``, a list of ParticleFiles), as arrays of shape
        ``(n_files, len(tags))``; NaN where a particle is not in a file.
        """
        files = self.files if files is None else files
        out = {name: np.full((len(files), len(tags)), np.nan) for name in names}
        for i, pf in enumerate(files):
            values = track_file(pf, tags, names)
            for name in names:
                out[name][i] = values[name]
        return out

    def __repr__(self) -> str:
        return f"<ParticleData  {self.stem}.NNNNN.{self.ext}  n_files={len(self)}>"


def track_file(pf: ParticleFile, tags, names) -> dict[str, np.ndarray]:
    """Columns ``names`` of the particles ``tags`` in one file; NaN where missing."""
    index, which = pf.find(tags)
    out = {}
    order = np.argsort(index)  # read in file order: sequential access
    values = pf.read(names, index[order])
    for name in names:
        col = np.full(len(tags), np.nan)
        col[which[order]] = values[name]
        out[name] = col
    if len(index) != len(np.unique(which)):
        warnings.warn(f"{pf.path.name}: some tags appear more than once")
    return out
