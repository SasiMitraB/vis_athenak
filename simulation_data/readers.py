"""
Reading single AthenaK output files (.bin, or .athdf from bin_convert),
built on the repo's ``bin_convert`` and ``athena_read`` modules.
"""

from __future__ import annotations

from pathlib import Path

if "." in __package__:  # imported as vis_athenak.simulation_data
    from ..data_processing import athena_read, bin_convert
else:  # imported as simulation_data, from the repo root
    from data_processing import athena_read, bin_convert

# Output file types that can be read, in order of preference.
EXTENSIONS = ("bin", "athdf")

# With one file per rank, rank 0's directory holds the file handed to the
# reader, which collects the matching files of the other ranks itself.
RANK0_DIR = "rank_00000000"


def read_time(path: Path) -> float:
    """Read only the simulation time from an output file's header."""
    if path.suffix == ".athdf":
        import h5py

        with h5py.File(path, "r") as f:
            return float(f.attrs["Time"])

    with open(path, "rb") as fp:
        fp.readline()  # "Athena ... version=1.1"
        pheader_count = int(fp.readline().split(b"=")[-1])
        for _ in range(pheader_count - 1):
            key, val = fp.readline().decode("utf-8").split("=", 1)
            if key.strip() == "time":
                return float(val)
    raise ValueError(f"no time found in header of {path}")


def read_file(path: Path, dtype=None) -> dict:
    """Read one output file into an athdf-style dict of merged arrays."""
    if path.suffix == ".athdf":
        return athena_read.athdf(str(path), dtype=dtype)
    if path.parent.name == RANK0_DIR:
        return bin_convert.read_all_ranks_binary_as_athdf(str(path), dtype=dtype)
    return bin_convert.read_binary_as_athdf(str(path), dtype=dtype)
