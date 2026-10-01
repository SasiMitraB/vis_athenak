"""
Lazy, frame-by-frame access to the outputs of an AthenaK run.

    from simulation_data import SimulationData

    sim = SimulationData("runs/blast/athinput.blast", "runs/blast")
    len(sim)                    # number of output frames
    for frame in sim:
        rho = frame["dens"]     # files for this frame are read here
    sim[-1].time                # header-only read, no field data loaded

Frames are discovered from the <output> blocks of the athinput file: an
output block with ``file_type = bin`` writes ``{basename}.{id}.{NNNNN}.bin``
(``id`` defaults to ``variable``).  Output blocks that share the same cadence
(e.g. ``mhd_w`` and ``mhd_bcc``) are merged, so a single Frame exposes the
variables of all of them.  ``.athdf`` files produced by ``bin_convert`` are
picked up as well.

Modules
-------
athinput         parse_athinput: athinput file -> {section: {key: value}}
readers          read_time / read_file: one output file -> header time / arrays
frame            Frame: one lazily loaded snapshot
simulation_data  SimulationData: discovers the frames of a run

Run as a module to print a summary of a run:

    python -m simulation_data athinput.blast runs/blast
"""

from .athinput import parse_athinput
from .frame import Frame
from .simulation_data import SimulationData

__all__ = ["Frame", "SimulationData", "parse_athinput"]
