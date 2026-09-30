# vis_athenak

Visualization and analysis toolkit for AthenaK simulation outputs.

Developed by the IISc Computational Astrophysics group.

## Structure

vis_athenak/
├── data_processing/   # Convert .bin/.athdf/.hdf5 → numpy arrays
├── plotting/          # Visualization scripts
└── utils/             # Constants, unit conversions, helpers

## Setup

git clone https://github.com/<your-org>/vis_athenak
cd vis_athenak
pip install -r requirements.txt

## Usage

Scripts in plotting/ expect preprocessed arrays from data_processing/.
Raw simulation data should be placed locally and is not tracked by git.

## Contributors
- Meemik Roy (meemikroy@iisc.ac.in)