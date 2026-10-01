# vis_athenak

Visualization and analysis tools for AthenaK simulation outputs.
Developed by the IISc Computational Astrophysics group.

## Project structure
- `simulation_data/` — `SimulationData` / `Frame`: lazy, frame-by-frame access to a run (athinput + output folder); plotting and utils build on these classes
- `data_processing/` — reads raw simulation files (.bin, .athdf, .hdf5) and converts them into numpy arrays for downstream use
- `plotting/` — visualization scripts that consume processed arrays (matplotlib, yt, etc.)
- `utils/` — shared constants, physical unit conversions, helper functions

## Related work
- Cooling function implementations live in the separate `Athenak_cooling` repo

## Conventions
- Python 3.10+
- Raw simulation data is never committed; it lives locally under a `data/` directory (gitignored)
- Processed arrays may be cached as `.npy` files (also gitignored)
- Each script should be runnable standalone with a clear argparse interface

## Common tasks Claude should help with
- Reading AthenaK HDF5/binary output formats
- Writing efficient numpy-based reduction pipelines
- Plotting 2D slices and radial profiles of MHD quantities (density, temperature, B-field)
- Comparing outputs across runs with different cooling prescriptions
