"""SimulationData: the sequence of output frames of one AthenaK run."""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np

from .athinput import parse_athinput
from .frame import Frame
from .readers import EXTENSIONS, RANK0_DIR


class SimulationData:
    """
    Sequence of the output Frames of one AthenaK run.

    Parameters
    ----------
    athinput : str or Path
        The athinput file the run was started with.
    datafolder : str or Path
        Directory holding the outputs.  It is searched recursively, so the
        run directory, its ``bin/`` subdirectory, or a folder of converted
        ``.athdf`` files all work.
    outputs : list of str, optional
        Output ids (``id`` or ``variable`` of the <output> blocks) to include
        in each frame.  They must share the same ``dt``/``dcycle``, since
        frame numbers only line up across outputs written at the same
        cadence.  Defaults to the first bin output block plus every other
        bin block with the same cadence.
    dtype : numpy dtype, optional
        dtype of the loaded field arrays (the readers default to float32).
    device : {"cpu", "gpu"}, optional
        Where fields are loaded: numpy arrays (default) or cupy arrays on the
        GPU.  Individual frames and fields can still be moved with
        ``frame.load(..., device=...)`` and ``field.to_gpu()``/``to_cpu()``.

    Indexing is positional like a list: ``sim[0]``, ``sim[-1]``, ``sim[2:5]``.
    ``frame.number`` is the file number, which differs after restarts or if
    outputs are missing.
    """

    def __init__(
        self,
        athinput: str | Path,
        datafolder: str | Path,
        outputs: list[str] | None = None,
        dtype=None,
        device="cpu",
    ):
        self.athinput = Path(athinput)
        self.datafolder = Path(datafolder)
        if not self.datafolder.is_dir():
            raise FileNotFoundError(f"data folder not found: {self.datafolder}")

        self.params = parse_athinput(self.athinput)
        self.basename = self.params.get("job", {}).get("basename")
        if not self.basename:
            raise ValueError(f"no <job> basename in {self.athinput}")

        self.outputs = self._select_outputs(outputs)
        self._frames = self._discover_frames(dtype, device)
        self._times: np.ndarray | None = None

    # ── Discovery ────────────────────────────────────────────────────────

    def _output_blocks(self) -> dict[str, dict[str, str]]:
        """Bin <output> blocks of the athinput, keyed by output id."""
        blocks = {}
        for section, block in self.params.items():
            if section.startswith("output") and block.get("file_type") == "bin":
                file_id = block.get("id", block.get("variable"))
                if file_id:
                    blocks[file_id] = block
        return blocks

    def _select_outputs(self, outputs: list[str] | None) -> list[str]:
        blocks = self._output_blocks()
        if not blocks:
            raise ValueError(f"no <output> blocks with file_type = bin in {self.athinput}")

        cadences = {
            o: _parse_cadence(b.get("dt"), b.get("dcycle")) for o, b in blocks.items()
        }
        if outputs is None:
            # Default to the outputs written alongside the first bin block.
            first = cadences[next(iter(blocks))]
            outputs = [o for o in blocks if cadences[o] == first]
            skipped = [o for o in blocks if o not in outputs]
            if skipped:
                warnings.warn(
                    f"skipping outputs {skipped}: written at a different cadence "
                    f"than {outputs}.  Pass outputs=[...] to choose."
                )

        unknown = [o for o in outputs if o not in blocks]
        if unknown:
            raise ValueError(f"unknown outputs {unknown}; athinput has {list(blocks)}")

        if len({cadences[o] for o in outputs}) > 1:
            cadences = {o: (blocks[o].get("dt"), blocks[o].get("dcycle")) for o in outputs}
            raise ValueError(
                "outputs are written at different cadences, so their frame "
                f"numbers do not line up: {cadences}.  Pass outputs=[...] to "
                "pick outputs with the same dt."
            )
        return outputs

    def _find_files(self, file_id: str) -> dict[int, Path]:
        """Map frame number -> file for one output id."""
        pattern = re.compile(
            rf"^{re.escape(self.basename)}\.{re.escape(file_id)}\.(\d+)\.(\w+)$"
        )
        for ext in EXTENSIONS:
            found = {}
            for path in self.datafolder.rglob(f"{self.basename}.{file_id}.*.{ext}"):
                m = pattern.match(path.name)
                # With one file per rank, every rank holds a copy of each frame;
                # only rank 0's is kept and the reader collects the others.
                in_other_rank = path.parent.name.startswith("rank_") and \
                    path.parent.name != RANK0_DIR
                if m and not in_other_rank:
                    found[int(m.group(1))] = path
            if found:
                return found
        return {}

    def _discover_frames(self, dtype, device) -> list[Frame]:
        files = {o: self._find_files(o) for o in self.outputs}
        missing = [o for o, found in files.items() if not found]
        if missing:
            raise FileNotFoundError(
                f"no files for outputs {missing} "
                f"({self.basename}.<id>.NNNNN.bin/.athdf) under {self.datafolder}"
            )

        # Keep only frame numbers present for every output.
        numbers = sorted(set.intersection(*(set(f) for f in files.values())))
        return [
            Frame(n, {o: files[o][n] for o in self.outputs}, dtype=dtype, device=device)
            for n in numbers
        ]

    # ── Sequence interface ───────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, index):
        return self._frames[index]

    def __iter__(self):
        # Release each frame's data once the loop moves on, so iterating over
        # a long run does not keep every snapshot in memory.
        for frame in self._frames:
            yield frame
            frame.unload()

    @property
    def frames(self) -> list[Frame]:
        return self._frames

    @property
    def frame_numbers(self) -> list[int]:
        return [f.number for f in self._frames]

    @property
    def times(self) -> np.ndarray:
        """Simulation time of every frame (header reads only, cached)."""
        if self._times is None:
            self._times = np.array([f.time for f in self._frames])
        return self._times

    def __repr__(self) -> str:
        return (
            f"<SimulationData  basename={self.basename!r}  "
            f"outputs={self.outputs}  n_frames={len(self)}>"
        )


def _parse_cadence(dt: str | None, dcycle: str | None) -> tuple:
    return tuple(float(c) if c is not None else None for c in (dt, dcycle))
