# vis_athenak

Visualization and analysis tools for [AthenaK](https://github.com/IAS-Astrophysics/athenak) simulation outputs.

Developed by the IISc Computational Astrophysics group.

## Structure

```
vis_athenak/
├── simulation_data/   # SimulationData / Frame: lazy access to a run's outputs
├── data_processing/   # Low-level readers for raw simulation files (.bin, .athdf)
├── plotting/          # Visualization scripts
└── utils/             # Constants, unit conversions, helpers
```

## Setup

Requires Python 3.10+.

```bash
git clone git@github.com:meemik-iisc/vis_athenak.git
cd vis_athenak
pip install -r requirements.txt
```

## Usage

```python
from simulation_data import SimulationData

sim = SimulationData("path/to/athinput.kh2d", "path/to/outputs")
len(sim)                    # number of output frames
for frame in sim:
    rho = frame["dens"]     # data is read from disk only here
```

For a full tutorial refer to the [Example Script](./example_script.py)

### Using it from another project

Install it into any virtual environment straight from GitHub:

```bash
pip install "git+https://github.com/meemik-iisc/vis_athenak.git"
pip install "vis_athenak[plot] @ git+https://github.com/meemik-iisc/vis_athenak.git"   # + matplotlib, scipy
```

```python
from vis_athenak import SimulationData
```

To work on the code itself, clone it and install it in editable mode with
`pip install -e .`, so edits take effect without reinstalling.

Without installing, you can also clone the repo into your project folder and
import it from there:

```bash
cd my_project
git clone git@github.com:meemik-iisc/vis_athenak.git
```

```python
# my_project/analysis.py, run from my_project/
from vis_athenak import SimulationData
```

The clone has to be named `vis_athenak` (the default), since a folder name
with a hyphen can't be imported.  Install the dependencies with
`pip install -r vis_athenak/requirements.txt`.

### Inside this repo

Run from the repository root.  `python -m simulation_data <athinput> <datafolder>`
prints a summary of a run.  Scripts in `plotting/` build on `SimulationData`.


## Acknowledgements

`data_processing/athena_read.py`, `bin_convert.py`, and `plot_slice.py` are
taken from [AthenaK's `vis/python`](https://github.com/IAS-Astrophysics/athenak/tree/main/vis/python)
and are distributed under its [BSD-3-Clause license](https://github.com/IAS-Astrophysics/athenak/blob/main/LICENSE).

## Contributors

- Meemik Roy (meemikroy@iisc.ac.in)
- Abhiram K  (abhiram1@iisc.ac.in)
