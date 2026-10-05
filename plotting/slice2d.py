"""
Plot 2D slices of one or more variables for every frame of a run.

Run from the repo root as a module:

    python -m plotting.slice2d                         # config.PLOT_ORDER for config.RUN
    python -m plotting.slice2d dens temp --axis y --frames 0:21:5
    python -m plotting.slice2d temp --run cbox_tabcool # another entry of config.SIMULATIONS
    python -m plotting.slice2d temp --vmin 1e4 --vmax 1e7 --cores 4
    python -m plotting.slice2d --athinput run/cbox.athinput --datafolder run/bin --outdir plots

Everything defaults to config.py: the run (SIMULATIONS, RUN; paths relative
to the roots in .env), the figure (PLOT) and each variable's quantity, units
and style (PLOT_VARS).  Flags override single values.  Images are written to
``<outdir>/<variable>/<basename>.<variable>.<axis>.<NNNNN>.<format>``.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

if "." in __package__:  # imported as vis_athenak.plotting
    from .. import config
    from ..simulation_data import SimulationData
    from ..utils.units import Units
else:  # run from the repo root
    import config
    from simulation_data import SimulationData
    from utils.units import Units

from .quantities import available, dimension, get  # noqa: E402
from .slices import draw_slice, take_slice  # noqa: E402


def parse_frames(tokens: list[str] | None):
    """['0', '5', '10:20:2'] -> set of frame numbers; None -> None (all)."""
    if not tokens:
        return None
    numbers = set()
    for tok in tokens:
        if ":" in tok:
            parts = [int(p) if p else None for p in tok.split(":")]
            start, stop = parts[0] or 0, parts[1]
            step = parts[2] if len(parts) > 2 and parts[2] else 1
            if stop is None:
                raise ValueError(f"frame range {tok!r} needs an end")
            numbers.update(range(start, stop, step))
        else:
            numbers.add(int(tok))
    return numbers


def unit_factor(params: dict, dimension: str, unit: str) -> float:
    """Code -> ``unit`` factor; 1 for "code" even if the run has no <units>."""
    return 1.0 if unit == "code" else Units.from_params(params).factor(dimension, unit)


def var_settings(name: str, opts: dict) -> dict:
    """config.plot_var(name) with the command-line overrides in ``opts`` applied."""
    var = config.plot_var(name)
    for key in ("cmap", "vmin", "vmax", "norm", "units"):
        if opts.get(key) is not None:
            var[key] = None if opts[key] == "linear" else opts[key]
    return var


def slice_variable(frame, var: dict, params: dict, opts: dict):
    """(x, y, plane) of ``var`` in its units, with x and y in opts["length_units"]."""
    length = unit_factor(params, "length", opts["length_units"])
    position = None if opts["position"] is None else opts["position"] / length
    data = get(frame, var["quantity"], params, units=var["units"])
    x, y, plane = take_slice(frame, data, axis=opts["axis"], position=position)
    return x * length, y * length, plane


def time_title(frame, params: dict, opts: dict) -> str:
    t = frame.time * unit_factor(params, "time", opts["time_units"])
    suffix = "" if opts["time_units"] == "code" else f" {opts['time_units']}"
    return f"t = {t:.2f}{suffix}"


def plot_frame(frame, variables, params, basename, opts) -> list[Path]:
    """Plot every variable of one frame; returns the files written."""
    written = []
    for name in variables:
        var = var_settings(name, opts)
        x, y, plane = slice_variable(frame, var, params, opts)

        fig, ax = plt.subplots(figsize=opts["figsize"], layout="constrained")
        draw_slice(ax, x, y, plane, cmap=var["cmap"], norm=var["norm"],
                   vmin=var["vmin"], vmax=var["vmax"],
                   axis=opts["axis"], length_units=opts["length_units"])
        ax.set_title(var["label"])
        fig.suptitle(time_title(frame, params, opts))

        outdir = Path(opts["outdir"]) / name
        outdir.mkdir(parents=True, exist_ok=True)
        path = outdir / f"{basename}.{name}.{opts['axis']}.{frame.number:05d}.{opts['format']}"
        fig.savefig(path, dpi=opts["dpi"], bbox_inches="tight")
        plt.close(fig)
        written.append(path)
    frame.unload()
    return written


def _plot_frame_star(job):
    return plot_frame(*job)


def main(argv=None):
    plot = config.PLOT
    parser = argparse.ArgumentParser(
        description="Plot 2D slices of AthenaK outputs, one image per frame and variable.",
    )
    parser.add_argument("variables", nargs="*", default=config.PLOT_ORDER,
                        help="keys of config.PLOT_VARS, or any quantity (plotted in code "
                             f"units); default: config.PLOT_ORDER = {config.PLOT_ORDER}")
    parser.add_argument("--run", default=config.RUN, choices=list(config.SIMULATIONS),
                        help=f"simulation in config.SIMULATIONS (default: {config.RUN})")
    parser.add_argument("--athinput", help="athinput file (default: from --run)")
    parser.add_argument("--datafolder", help="folder holding the outputs (default: from --run)")
    parser.add_argument("--outdir", help="where images are written (default: from --run)")
    parser.add_argument("--outputs", nargs="+", help="output ids to read (default: from --run)")
    parser.add_argument("--frames", nargs="+",
                        help="frame numbers and/or ranges like 0:21:5 (default: from --run)")
    parser.add_argument("--axis", choices=["x", "y", "z"], default=plot["axis"],
                        help=f"slice normal; z for 2D runs (default: {plot['axis']})")
    parser.add_argument("--position", type=float, default=plot["position"],
                        help="slice position along --axis in --length-units (default: midplane)")
    parser.add_argument("--length-units", default=plot["length_units"],
                        help=f"units of the axes (default: {plot['length_units']})")
    parser.add_argument("--time-units", default=plot["time_units"],
                        help=f"units of the time in titles (default: {plot['time_units']})")
    # The rest override PLOT_VARS for every variable plotted.
    parser.add_argument("--units", help="units of the variable (default: from PLOT_VARS)")
    parser.add_argument("--cmap", help="colormap (default: from PLOT_VARS)")
    parser.add_argument("--norm", choices=["log", "linear"],
                        help="colour scale (default: from PLOT_VARS)")
    parser.add_argument("--vmin", type=float, help="colour scale minimum (default: from PLOT_VARS)")
    parser.add_argument("--vmax", type=float, help="colour scale maximum (default: from PLOT_VARS)")
    parser.add_argument("--figsize", type=float, nargs=2, default=plot["fig_size_single"])
    parser.add_argument("--dpi", type=int, default=plot["dpi"])
    parser.add_argument("--format", default=plot["format"], help="image format")
    parser.add_argument("-c", "--cores", type=int, default=1,
                        help="frames plotted in parallel (default: 1)")
    args = parser.parse_args(argv)

    # Fill whatever was not given on the command line from config.SIMULATIONS.
    paths = (args.athinput, args.datafolder, args.outdir)
    if any(p is None for p in paths) or args.outputs is None or args.frames is None:
        try:
            run = config.resolve(args.run)
        except RuntimeError as err:
            if any(p is None for p in paths):
                parser.error(f"{err}, or pass --athinput, --datafolder and --outdir")
            run = config.SIMULATIONS[args.run]
        args.athinput = args.athinput or run["athinput"]
        args.datafolder = args.datafolder or run["data"]
        args.outdir = args.outdir or run["out"]
        args.outputs = args.outputs or run["outputs"]
        frames_default = run["frames"]
    else:
        frames_default = None

    sim = SimulationData(args.athinput, args.datafolder, outputs=args.outputs)
    wanted = parse_frames(args.frames)
    if wanted is None and frames_default is not None:
        wanted = set(frames_default)
    frames = [f for f in sim.frames if wanted is None or f.number in wanted]
    if not frames:
        parser.error(f"no frames selected; run has frames {sim.frame_numbers}")
    if not args.variables:
        parser.error("no variables given and config.PLOT_ORDER is empty")
    unknown = [v for v in args.variables
               if config.plot_var(v)["quantity"] not in available(frames[0])]
    if unknown:
        parser.error(f"unknown variables {unknown}; config.PLOT_VARS has "
                     f"{list(config.PLOT_VARS)}, quantities: {available(frames[0])}")

    opts = {k: getattr(args, k) for k in
            ("axis", "position", "length_units", "time_units", "units", "cmap", "norm",
             "vmin", "vmax", "figsize", "dpi", "format", "outdir")}
    # Fail on a bad unit before spawning any work.
    try:
        unit_factor(sim.params, "length", args.length_units)
        unit_factor(sim.params, "time", args.time_units)
        for v in args.variables:
            var = config.plot_var(v)
            get_units = args.units or var["units"]
            if get_units != "code":
                Units.from_params(sim.params).factor(dimension(var["quantity"]), get_units)
    except ValueError as err:
        parser.error(str(err))

    jobs = [(f, args.variables, sim.params, sim.basename, opts) for f in frames]
    print(f"{sim}\nplotting {args.variables} for {len(frames)} frames -> {args.outdir}")

    if args.cores > 1:
        with ProcessPoolExecutor(max_workers=args.cores) as pool:
            for frame, written in zip(frames, pool.map(_plot_frame_star, jobs)):
                print(f"  frame {frame.number:5d}: {len(written)} images")
    else:
        for job in jobs:
            written = plot_frame(*job)
            print(f"  frame {job[0].number:5d}: {len(written)} images")


if __name__ == "__main__":
    main()
