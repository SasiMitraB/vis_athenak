# vis_athenak

Visualization and analysis tools for [AthenaK](https://github.com/IAS-Astrophysics/athenak) simulation outputs.

Developed by the IISc Computational Astrophysics group.

## Structure

```
vis_athenak/
├── data_processing/   # Read raw simulation files → numpy arrays
├── plotting/          # Visualization scripts
└── utils/             # Constants, unit conversions, helpers
```

## Setup

Requires Python 3.10+.

```bash
git clone git@github.com:meemik-iisc/vis_athenak.git
cd vis_athenak
```

## Usage

Scripts in `plotting/` use arrays produced by `data_processing/`.
Raw simulation data and cached `.npy` arrays are not tracked by git.

## Acknowledgements

`data_processing/athena_read.py`, `bin_convert.py`, and `plot_slice.py` are
taken from [AthenaK's `vis/python`](https://github.com/IAS-Astrophysics/athenak/tree/main/vis/python)
and are distributed under its [BSD-3-Clause license](https://github.com/IAS-Astrophysics/athenak/blob/main/LICENSE).

## Contributors

- Meemik Roy (meemikroy@iisc.ac.in)
