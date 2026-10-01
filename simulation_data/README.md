# simulation_data

Lazy, frame-by-frame access to the outputs of an AthenaK run. Give it the
athinput file and the output folder, and you get a sequence of frames. Field
data is read from disk only when a frame is used. Plotting and analysis code
in this repo is meant to build on these classes.

```python
from simulation_data import SimulationData

sim = SimulationData("test_data/kh2d/athinput.kh2d", "test_data/kh2d/bin")
len(sim)                    # 31
for frame in sim:
    rho = frame["dens"]     # files for this frame are read here
sim[-1].time                # 3.0, read from the file header only
```

[`example_script.py`](../example_script.py) in the repo root walks through
every feature. Run it from the repo root with `python example_script.py`.

## Layout

| File | Contents |
|---|---|
| `athinput.py` | `parse_athinput(path)`: athinput file → `{section: {key: value}}`, values as strings |
| `readers.py` | `read_time(path)` (header only) and `read_file(path, dtype)` (full file), built on `data_processing/bin_convert.py` and `athena_read.py` |
| `frame.py` | `Frame`: one snapshot, lazily loaded |
| `simulation_data.py` | `SimulationData`: finds the frames of a run |
| `__main__.py` | `python -m simulation_data <athinput> <datafolder>`: prints the frames and their times |

Import from the package (`from simulation_data import SimulationData, Frame,
parse_athinput`) and run from the repo root, since `readers.py` imports
`data_processing`.

## SimulationData

```python
SimulationData(athinput, datafolder, outputs=None, dtype=None)
```

| | |
|---|---|
| `len(sim)`, `for frame in sim` | number of frames; iterate in order |
| `sim[i]`, `sim[-1]`, `sim[a:b]` | positional indexing, like a list; slices return a list of `Frame` |
| `sim.frames`, `sim.frame_numbers` | the `Frame` list; file numbers (`NNNNN` in the filename) |
| `sim.times` | time of every frame, from file headers only (cached) |
| `sim.params`, `sim.basename`, `sim.outputs` | parsed athinput, job basename, output ids merged into each frame |

### Finding the frames

* AthenaK writes each `<output>` block with `file_type = bin` as
  `{basename}.{id}.{NNNNN}.bin`, where `id` defaults to `variable`.
  `SimulationData` reads these blocks from the athinput file. It then
  searches `datafolder` recursively, so the run directory or its `bin/`
  subdirectory both work.
* `.athdf` files converted with `bin_convert` are found the same way. If
  both formats are present, `.bin` is used.
* One file per rank (`bin/rank_XXXXXXXX/`) is supported. Only rank 0's file
  is listed, and the reader collects the other ranks.
* **Output blocks with the same cadence are merged into one frame.** For
  example, `frame["dens"]` (from `hydro_w`) and `frame["mom1"]` (from
  `hydro_u`) come from the same snapshot. Frame numbers only line up between
  outputs written at the same `dt`/`dcycle`.
  * By default, the first bin output and every other bin output with the
    same cadence are used. Outputs with a different cadence are skipped with
    a warning.
  * Pass `outputs=["mhd_w", "mhd_bcc"]` to choose. Asking for outputs with
    different cadences raises `ValueError`.
* A frame is kept only if every selected output has a file for it.
* Indexing is positional. `frame.number` is the file number. The two differ
  after a restart, or when a frame is missing for one of the outputs.

### Memory

`for frame in sim` frees each frame's data once the loop moves on, so a long
run doesn't fill up memory. Arrays you keep a reference to are not affected.
`sim[a:b]` and `sim.frames` are plain lists, so looping over them keeps data
cached until you call `frame.unload()`.

## Frame

| | |
|---|---|
| `frame["dens"]` | field array, shape `(nx3, nx2, nx1)`; loads all of the frame's files on first access |
| `frame["x1v"]`, `frame["x1f"]`, … | cell-centre and face coordinates |
| `frame.keys()`, `"bcc1" in frame` | available names (loads the data) |
| `frame.time` | simulation time; reads only the header if the data isn't loaded |
| `frame.number`, `frame.paths` | file number; `{output id: path}` |
| `frame.data` | the merged dict itself |
| `frame.is_loaded`, `frame.unload()` | check whether data is cached; drop it (it is re-read on the next access) |

Variable names are AthenaK's own (`dens`, `velx`, `eint`, `mom1`, `ener`,
`bcc1`, `s_00`, …). No renaming or derived fields are added yet. Arrays are
float32 unless you pass `dtype=np.float64`. Frames loaded from `.athdf` also
contain the file's HDF5 attributes (`RootGridSize`, `VariableNames`, …).

## Not supported yet

* Physical units (everything is in code units)
* Derived fields such as temperature
* Coarsened binary output (`cbin`), VTK, and tab outputs

## TODO

1. **`Field` class.** A simulation is made of frames, and a frame is made of
   fields. A `Field` would hold a field's array together with its units, and
   `frame["dens"]` would return a `Field` instead of a bare numpy array. Add
   this once per-field lazy loading is available. At the moment, accessing
   any field loads every file of the frame (see the `frame["dens"]` row
   above), so a per-field object has nothing to be lazy about yet.
2. **Derived fields.** These work like fields, but are defined by a function
   that combines other fields, for example temperature from `eint` and `dens`.
   The result is computed when the frame is loaded and stored with the
   frame's other fields.