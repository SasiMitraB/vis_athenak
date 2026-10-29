"""
Simulations the scripts can work on, and which one they use by default.

Paths are relative to root folders given by the environment variables
``ATHINPUT_DIR`` and ``DATA_DIR`` (machine-specific; ``DATA_DIR`` defaults to
``ATHENAK_DIR``), so this file is the
same on every machine.  They come from the shell or a ``.env`` file (copy
``.env.example``); see ``utils/env.py``.  An
absolute path in an entry is used as is.  Scripts take ``--run <name>`` to
pick another entry, and flags to override any single value.

The second half sets what the plotting scripts draw: PLOT (figure-wide
settings), PLOT_VARS (one entry per variable that can be plotted) and
PLOT_ORDER (which of them are plotted by default), COMBINED (the grid of
panels of the combined figure and its video) and MAIN (what `python main.py`
makes).

    from config import resolve
    run = resolve()            # paths of RUN
    run = resolve("cbox_02Myr")
    sim = SimulationData(run["athinput"], run["data"], outputs=run["outputs"])
"""

from __future__ import annotations

if __package__:  # imported as vis_athenak.config
    from .utils.env import env_path
else:            # run from the repo root
    from utils.env import env_path
    
# What `python main.py` (no command) makes; each switch can be flipped on the
# command line, e.g. --no-make-slices or --save-png.
MAIN = dict(
    make_slices=True,    # one image per frame and variable in PLOT_ORDER
    make_combined=True,  # one image per frame of COMBINED["layout"]
    make_video=False,     # a video of the combined images
    save_png=False,       # False: no slices, and the combined images are deleted once in the video
    make_particles=False,  # a movie of tracked particles, set up by PARTICLES below
)

# Parallel worker processes for plotting frames (slices and combined figures);
# 1 = serial.  Each worker holds one frame of every run it plots in memory, so
# lower this for large runs.  -w/--workers overrides it for one run.
N_WORKERS = 2

RUN = "clumps"  # default simulation

SIMULATIONS = {
    "clumps": dict(
        athinput="X_ray_H_alpha_test.athinput",          # relative to $ATHINPUT_DIR
        data="bin",       # relative to $DATA_DIR
        out="plots",  # relative to $DATA_DIR
        outputs=None,  # output ids, e.g. ["hydro_w"]; None = auto (see SimulationData)
        frames=None,   # frame numbers, e.g. range(0, 21, 5); None = all
        particles=None,  # folder of particle files (relative to $DATA_DIR); None = search
                         # the data folder and its parent
    ),
    # Runs with tracer particles, e.g.:
    # "trml_fid2d": dict(
    #     athinput="trml/trml_fid2d.athinput",
    #     data="trml_fid2d/bin",
    #     out="trml_fid2d/plots",
    #     particles="trml_fid2d/pbin",
    #     outputs=None,
    #     frames=None,
    # ),
    # "clump": dict(
    #     athinput="clumps/clumps.athinput",
    #     data="clumps_run1/bin",
    #     out="clumps_run1/plots",
    #     particles="clumps_run1/pbin",
    #     outputs=None,
    #     frames=None,
    # ),
    # "cbox_tabcool": dict(
    #     athinput="cooling_box/cbox.athinput",
    #     data="build_cooling_box/src/cbox_tabcool",
    #     out="build_cooling_box/src/cbox_tabcool/plots",
    #     outputs=None,
    #     frames=None,
    # ),
}


def resolve(run: str = RUN) -> dict:
    """Settings of ``run`` with ``athinput``, ``data``, ``out`` as absolute Paths."""
    if run not in SIMULATIONS:
        raise KeyError(f"unknown run {run!r}; config.SIMULATIONS has {list(SIMULATIONS)}")
    sim = dict(SIMULATIONS[run])
    sim["athinput"] = env_path("ATHINPUT_DIR") / sim["athinput"]
    data_root = env_path("DATA_DIR", fallback="ATHENAK_DIR")
    sim["data"] = data_root / sim["data"]
    sim["out"] = data_root / sim["out"]
    sim.setdefault("outputs", None)
    sim.setdefault("frames", None)
    sim["particles"] = data_root / sim["particles"] if sim.get("particles") else None
    return sim


# ── Plotting ─────────────────────────────────────────────────────────────

PLOT = dict(
    axis="z",             # slice normal: x, y or z (z for 2D runs)
    position=None,        # slice position along axis, in length_units; None = midplane
    length_units="kpc",    # axis coordinates: code, cm, pc, kpc
    time_units="Myr",     # time in titles: code, s, yr, kyr, Myr
    time_decimals=0,      # decimal places of the time in titles (0 = whole numbers)
    fig_size_single=(16.0, 3.0),
    panel_size=(16, 3),  # combined figure: size of each panel
    dpi=150,
    format="png",
)

# Fonts of every figure (sizes in points).
FONTS = dict(
    weight="bold",  # "bold" or "normal"; also applies to math text such as 10^{-4}
    title=16,       # panel titles (the variable labels)
    suptitle=18,    # time at the top of the figure
    label=14,       # axis labels
    ticks=12,       # tick labels of axes and colorbars
)

# One entry per plottable variable; the key names the output folder and the
# CLI argument.  Fields:
#   quantity  raw variable (dens, velx, eint, s_00, ...) or derived one
#             (pres, temp, vmag, bmag, beta, entropy, t_cool); see plotting/quantities.py
#   units     "code", or a unit of the quantity's dimension (utils/units.py UNIT_NAMES):
#             density g/cm^3 or cm^-3, pressure dyne/cm^2, velocity km/s or cm/s,
#             temperature K, time s/yr/kyr/Myr, magnetic G/uG
#   norm      "log" or None (linear)
#   vmin/vmax colour limits in those units; None = autoscale each frame
PLOT_VARS = {
    "dens": dict(
        label=r"Density [$\mathbf{m_p/cm^3}$]",
        quantity="dens", units="cm^-3",
        cmap="Greens", norm="log", vmin=1.0e-4, vmax=1.0e1,
    ),
    "pres": dict(
        label=r"Pressure [$\mathbf{dyne/cm^2}$]",
        quantity="pres", units="dyne/cm^2",
        cmap="viridis", norm="log", vmin=1.0e-16, vmax=1.0e-11,
    ),
    "entropy": dict(
        label="Entropy [code]",
        quantity="entropy", units="code",
        cmap="magma", norm=None, vmin=-20.0, vmax=0.0,
    ),
    "velx": dict(
        label="X Velocity [km/s]",
        quantity="velx", units="km/s",
        cmap="Blues_r", norm=None, vmin=-1.0e3, vmax=0.0,
    ),
    "vely": dict(
        label="Y Velocity [km/s]",
        quantity="vely", units="km/s",
        cmap="seismic", norm=None, vmin=-500.0, vmax=500.0,
    ),
    "temp": dict(
        label="Temperature [K]",
        quantity="temp", units="K",
        cmap="coolwarm", norm="log", vmin=1.0e4, vmax=1.0e7,
    ),
    "t_cool": dict(
        label="Cooling Time [Myr]",
        quantity="t_cool", units="Myr",
        cmap="turbo", norm="log", vmin=1.0e-3, vmax=1.0e3,
    ),
    "tracer": dict(
        label="Outflow Tracer",
        quantity="s_00", units="code",
        cmap="winter", norm="log", vmin=1.0e-5, vmax=1.0,
    ),
}

# Variables plotted when none are named on the command line.
PLOT_ORDER = [
    "dens",
    # "pres",
    "temp",
    # "tracer",
    #"velx",
    "vely",
    # "entropy",
    # "t_cool",
]

# Combined figure (python main.py combined) and its video (python main.py video).
# layout is the grid of panels as a list of rows; each cell is
#   "temp"                    a PLOT_VARS key (or quantity), plotted for RUN
#   ("cbox_tabcool", "temp")  a (run, variable) pair, for comparing runs
#   None                      an empty panel
# A row (or column) holding one variable throughout shares one colorbar.
COMBINED = dict(
    name="combined",  # images <name>.<axis>.<NNNNN>.<format>, video <name>.<axis>.mp4
    layout=[
        ["dens"],
        ["temp"],
        # ["velx"],
        ["vely"],
    ],
    fps=24,           # video frames per second
    video_max_size=4096,  # video frames scaled down to at most this many pixels per side
                          # (the encoders fail above 4096); None/0 = full size
)
# Comparing runs, one row per variable and one column per run:
# COMBINED["layout"] = [[(run, var) for run in ["cbox_02Myr", "cbox_tabcool"]]
#                       for var in ["dens", "temp"]]


# ── Particle movies ──────────────────────────────────────────────────────
# python main.py particles (or MAIN["make_particles"]): tracked particles with
# fading trails, one image per particle output file, then a video.  Works for
# any run: domain, periodicity, dimensions and units come from its athinput.
# Every setting has a command-line flag; python main.py particles --help.
# Start with  python main.py particles --info  to see what the files hold.
PARTICLES = dict(
    # What to track
    n_track=10,         # how many particles get markers and trails
    bins=[1.0e5],       # edges in the colour's units; n_track is split evenly over
                        # the bins (here half below 1e5 K, half above); None = random
    tags=None,          # or exactly these tags, e.g. [12, 503, 9981]
    require=None,       # history conditions on the colour quantity, "FILE:LO:HI" with FILE
                        # a particle file number, first, last or last-N; LO/HI may be empty.
                        # Condensing gas (hot at the start, cold at the end):
                        #   require=["first:1e6:", "last::2e4"]
    select_at=None,     # particle file number where they are picked; None = first frame
    region=None,        # pick only inside, in length_units: dict(x=(-1, 1), y=(0, None))
    seed=0,             # random picks are reproducible; change for another set
    background=0,       # this many other particles drawn as faint dots (no trails)
    trail=50,           # trail length in frames; 0 = none, -1 = whole history
    frames=None,        # particle file numbers, e.g. range(0, 1001, 2); None = all

    # Colour: a PLOT_VARS key (its label/units/cmap/norm/vmin/vmax are used) or
    # any particle column / derived quantity; None = one colour.
    color="temp",
    cmap="turbo",       # colormap of the particles; None = the one in PLOT_VARS
    cmap_range=(0.12, 1.0),  # part of the colormap used; drops turbo's near-black end
    vmin=None,          # colour range; None = PLOT_VARS', else the movie's data range
    vmax=None,

    # Gas drawn underneath (PLOT_VARS key), from the grid output nearest in time;
    # None = particles only.
    gas=None,
    gas_cmap="Greys_r",  # None = the colormap in PLOT_VARS
    gas_alpha=0.6,

    # View (axis and slice position come from PLOT)
    xlim=None,          # (lo, hi) in length_units; None = whole domain
    ylim=None,
    slab=None,          # 3D: background dots only within this thickness of the slice
    length_units=None,  # None = PLOT's; "code" to keep code units
    time_units=None,    # None = PLOT's
    time_decimals=2,
    style="dark",       # "dark" or "light"
    marker_size=120,
    fig_size=None,      # (w, h) in inches; None = fitted to the domain
    dpi=100,
    title=None,         # None = the run name
    name="particles",   # file prefix: <out>/particles/<name>.<axis>.NNNNN.png and .mp4
    fps=25,
    video=True,

    # Particle file format.  Grid fields sampled at the particles are taken to be
    # in code units; name any stored in other units, e.g. {"temp": "K"}.
    stored_units={},
    rdata_names=None,   # real columns of .prtclbin; None = x, y, z, vx, vy, vz
    idata_names=None,   # integer columns; None = gid, tag, lastmove, lastlevel
    stem=None,          # file name stem if a folder holds several particle sets
)


def plot_var(name: str) -> dict:
    """
    Settings of PLOT_VARS[name]; a name not in PLOT_VARS is taken as a
    quantity and plotted in code units with a linear viridis map.
    """
    defaults = dict(label=name, quantity=name, units="code",
                    cmap="viridis", norm=None, vmin=None, vmax=None)
    return {**defaults, **PLOT_VARS.get(name, {})}
