"""
Self-contained check of the particle reader: writes a few small .prtclbin
files with the particles shuffled between files (as when they migrate between
ranks) and some removed (as at an outflow boundary), then checks that
ParticleData.track matches them by tag.  Needs no simulation data.

    python test/test_particles.py        (from the repo root)
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from simulation_data.particles import ParticleData  # noqa: E402


def write_prtclbin(path, t, x, y, z, tags, temp):
    n = len(tags)
    with open(path, "wb") as f:
        f.write(np.array([43, n, 6, 4, 1], dtype="<i8").tobytes())
        f.write(np.array([t, 1e-3, 0], dtype="<f8").tobytes())
        f.write(b"temp".ljust(16, b"\0"))
        for a in (x, y, z, 0 * x, 0 * x, 0 * x):
            f.write(np.asarray(a, "<f8").tobytes())
        for a in (0 * tags, tags, 0 * tags, 0 * tags):
            f.write(np.asarray(a, "<i4").tobytes())
        f.write(np.asarray(temp, "<f8").tobytes())


def main():
    rng = np.random.default_rng(0)
    n, nfiles = 1000, 5
    tags = np.arange(n)
    with tempfile.TemporaryDirectory() as tmp:
        for k in range(nfiles):
            keep = rng.permutation(n)[: n - 50 * k]  # shuffled, and fewer every file
            # Position x = tag + k encodes which particle and which file it is.
            write_prtclbin(Path(tmp) / f"run.{k:05d}.prtclbin", 0.5 * k,
                           tags[keep] + k, 0 * keep, 0 * keep, tags[keep], 10.0 * tags[keep])
        pdata = ParticleData(tmp, basename="run")
        assert len(pdata) == nfiles and np.allclose(pdata.times, 0.5 * np.arange(nfiles))
        wanted = np.array([7, 999, 3, 512])
        tr = pdata.track(wanted, ["x", "temp"])
        for k in range(nfiles):
            present = np.isin(wanted, pdata[k].tags())
            assert np.allclose(tr["x"][k, present], wanted[present] + k)
            assert np.allclose(tr["temp"][k, present], 10.0 * wanted[present])
            assert np.isnan(tr["x"][k, ~present]).all()
    print("particle reader: ok")


if __name__ == "__main__":
    main()
