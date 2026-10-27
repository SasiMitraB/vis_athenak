"""
Particle movies: tracked particles with fading trails, coloured by a quantity.

    python main.py particles --info               # what the particle files hold
    python main.py particles                      # config.PARTICLES for config.RUN
    python main.py particles --run clump --n 20 --bins 2e4 2e5 --trail 100
    python main.py particles --tags 12 503 9981 --gas temp --frames 0:400:2
    python main.py particles --background 20000 --no-video

The same code serves any setup (a periodic 2D TRML box, a 3D cloud-crushing
run with outflow boundaries, ...): the domain, its periodicity, the
dimensionality and the units are read from the run's athinput, and particles
are matched across files by tag.

What it does:

1. Finds the run's particle files (``.prtclbin`` or AthenaK's ``.part.vtk``)
   in the ``particles`` folder of its config.SIMULATIONS entry, else under
   its data folder and that folder's parent.
2. Picks the particles to track in one file (``--select-at``): the tags
   given, or ``--n`` at random, split evenly over the bins of the colour
   quantity (``--bins``: e.g. one edge at 1e5 K gives half cold, half hot),
   optionally only inside a region.  ``--background`` more are drawn every
   frame as faint dots, without trails.
3. Collects their positions and colour quantity from every file, in
   parallel, and caches them in ``<outdir>/<name>.tracks.npz`` (rerun with
   other plot settings without reading the files again; ``--refresh`` to
   redo it).
4. Draws one image per file: optional gas slice underneath (``--gas``, from
   the grid output nearest in time), background dots, trails (broken where a
   particle crosses a periodic boundary or is missing), and the particles.
5. Makes ``<outdir>/<name>.<axis>.mp4``; with save_png off the images are
   deleted once in the video.

Colours come from config.PLOT_VARS when the colour quantity has an entry
there (label, units, cmap, norm, vmin, vmax), so particle and slice colours
match.  The colour quantity is any column of the particle files (``dens``,
``temp``, ``vx``, ...) or a quantity derived from them as for slices
(``temp`` from ``dens`` and ``eint``/``pres``, ``vmag``, ``t_cool``, ...).
Grid fields are taken to be stored in code units; if a field is stored in
other units say so in config.PARTICLES["stored_units"], e.g. {"temp": "K"}.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import colors  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

if "." in __package__:  # imported as vis_athenak.plotting
    from .. import config
    from ..simulation_data import SimulationData
    from ..simulation_data.particles import (
        DEFAULT_IDATA_NAMES, DEFAULT_RDATA_NAMES, ParticleData,
    )
    from ..utils.units import Units
else:  # run from the repo root
    import config
    from simulation_data import SimulationData
    from simulation_data.particles import (
        DEFAULT_IDATA_NAMES, DEFAULT_RDATA_NAMES, ParticleData,
    )
    from utils.units import Units

from .quantities import DERIVED, dimension, get  # noqa: E402
from .slice2d import (  # noqa: E402
    apply_fonts, frame_plane, n_workers, non_negative_int, parse_frames, slice_variable,
    time_title, unit_factor, var_settings,
)
from .slices import draw_slice  # noqa: E402

# Slice normal -> (horizontal, vertical, normal) particle coordinates.
PLANE_AXES = {"z": ("x", "y", "z"), "y": ("x", "z", "y"), "x": ("y", "z", "x")}
MESH_DIRECTION = {"x": "1", "y": "2", "z": "3"}
COLUMN_DIMENSIONS = {"x": "length", "y": "length", "z": "length",
                     "vx": "velocity", "vy": "velocity", "vz": "velocity"}
# Grid-variable names the derived quantities use, and the particle columns they map to.
ALIASES = {"velx": "vx", "vely": "vy", "velz": "vz"}

STYLES = {
    "dark": dict(base="dark_background", face="#0a0d14", edge="white", grid="#2a2f3a",
                 single="#4da3ff"),
    "light": dict(base="default", face="white", edge="black", grid="#dddddd",
                  single="#1f5fbf"),
}


# ── Quantities at the particles ──────────────────────────────────────────

def particle_dimension(name: str) -> str:
    """Dimension of a particle column or quantity, for unit conversion."""
    name = name.split("/")[-1]
    return COLUMN_DIMENSIONS.get(name) or dimension(ALIASES.get(name, name))


class ParticleView:
    """
    The particles at ``index`` of one file, in code units, shaped like a Frame
    (``fields``, ``[name]``) so ``plotting.quantities.get`` can derive
    quantities from the sampled grid fields.
    """

    def __init__(self, pf, index, params: dict, stored_units: dict | None = None):
        self.pf, self.index, self.params = pf, index, params
        self.stored_units = stored_units or {}
        names = pf.names
        self._fields = {n: n for n in names}
        for grid_name, column in ALIASES.items():
            if column in names and grid_name not in names:
                self._fields[grid_name] = column
        if "pres" in names and "eint" not in names:
            self._fields["eint"] = "pres"  # converted in __getitem__
        self._cache: dict[str, np.ndarray] = {}

    @property
    def fields(self) -> dict:
        return self._fields

    def _column(self, column: str) -> np.ndarray:
        if column not in self._cache:
            values = self.pf.read([column], self.index)[column].astype(float)
            unit = self.stored_units.get(column, "code")
            if unit != "code":
                values = values / Units.from_params(self.params).factor(
                    particle_dimension(column), unit)
            self._cache[column] = values
        return self._cache[column]

    def __getitem__(self, name: str) -> np.ndarray:
        column = self._fields[name]
        values = self._column(column)
        if name == "eint" and column == "pres":
            gamma = float((self.params.get("mhd") or self.params.get("hydro") or {})
                          .get("gamma", 5.0 / 3.0))
            values = values / (gamma - 1.0)
        return values


def particle_quantity(view: ParticleView, name: str, units: str = "code") -> np.ndarray:
    """Quantity ``name`` of the particles of ``view`` in ``units``."""
    if name in view.fields:
        data = view[name]
    elif name in DERIVED:
        data = np.asarray(get(view, name, view.params, units="code"), dtype=float)
    else:
        raise KeyError(f"{name!r} is neither a column of {view.pf.path.name} "
                       f"({view.pf.names}) nor a derived quantity ({list(DERIVED)})")
    if units != "code":
        data = data * Units.from_params(view.params).factor(particle_dimension(name), units)
    return data


def gather_file(pf, tags, color: dict | None, params: dict, stored_units: dict) -> dict:
    """Positions (code units) and colour (its units) of ``tags`` in one file."""
    index, which = pf.find(tags)
    order = np.argsort(index)
    index, which = index[order], which[order]
    out = {}
    for axis in ("x", "y", "z"):
        col = np.full(len(tags), np.nan)
        if axis in pf.names:
            col[which] = pf.read([axis], index)[axis]
        else:
            col[which] = 0.0
        out[axis] = col
    out["color"] = np.full(len(tags), np.nan)
    if color is not None and len(index):
        view = ParticleView(pf, index, params, stored_units)
        out["color"][which] = particle_quantity(view, color["quantity"], color["units"])
    return out


def _gather_star(job):
    return gather_file(*job)


# ── Domain ───────────────────────────────────────────────────────────────

def domain(params: dict) -> dict:
    """{axis: (min, max, periodic)} of the mesh, code units."""
    mesh = params.get("mesh", {})
    out = {}
    for axis, d in MESH_DIRECTION.items():
        lo, hi = float(mesh.get(f"x{d}min", 0.0)), float(mesh.get(f"x{d}max", 0.0))
        nx = int(mesh.get(f"nx{d}", 1))
        periodic = (mesh.get(f"ix{d}_bc") == "periodic" and mesh.get(f"ox{d}_bc") == "periodic"
                    and nx > 1)
        out[axis] = (lo, hi, periodic)
    return out


# ── Selection ────────────────────────────────────────────────────────────

def select_tags(pf, params: dict, opts: dict, color: dict | None) -> tuple[np.ndarray, str]:
    """Tags to track, picked in file ``pf``; and a line describing the choice."""
    rng = np.random.default_rng(opts["seed"])
    if opts["tags"]:
        tags = np.asarray(opts["tags"], dtype=np.int64)
        index, which = pf.find(tags)
        missing = sorted(set(tags.tolist()) - set(tags[which].tolist()))
        note = f" ({len(missing)} not in file {pf.number}: {missing[:10]})" if missing else ""
        return tags, f"{len(tags)} particles given by tag{note}"

    all_tags = pf.tags()
    candidates = np.arange(len(all_tags))
    region = opts.get("region") or {}
    if region:
        length = unit_factor(params, "length", opts["length_units"])
        positions = pf.read([a for a in region if a in pf.names])
        keep = np.ones(len(all_tags), dtype=bool)
        for axis, (lo, hi) in region.items():
            if axis not in positions:
                raise SystemExit(f"region: no {axis!r} column in {pf.path.name}")
            v = positions[axis] * length
            keep &= (v >= (-np.inf if lo is None else lo)) & (v <= (np.inf if hi is None else hi))
        candidates = candidates[keep]
        if not candidates.size:
            raise SystemExit(f"no particles inside region {region} in file {pf.number}")

    n = opts["n_track"]
    bins = opts.get("bins")
    if not bins:
        pick = rng.choice(candidates, size=min(n, candidates.size), replace=False)
        return np.sort(all_tags[pick]), f"{len(pick)} random particles"

    if color is None:
        print("note: no colour quantity, so the bins are ignored and the picks are random")
        pick = rng.choice(candidates, size=min(n, candidates.size), replace=False)
        return np.sort(all_tags[pick]), f"{len(pick)} random particles"
    values = particle_quantity(ParticleView(pf, candidates, params, opts["stored_units"]),
                               color["quantity"], color["units"])
    edges = [-np.inf, *sorted(bins), np.inf]
    shares = [len(a) for a in np.array_split(np.arange(n), len(edges) - 1)]
    chosen, parts = [], []
    for lo, hi, share in zip(edges[:-1], edges[1:], shares):
        inside = candidates[(values >= lo) & (values < hi)]
        take = rng.choice(inside, size=min(share, inside.size), replace=False)
        chosen.append(take)
        u = "" if color["units"] == "code" else f" {color['units']}"
        span = (f"< {hi:g}{u}" if lo == -np.inf else f">= {lo:g}{u}" if hi == np.inf
                else f"{lo:g}-{hi:g}{u}")
        parts.append(f"{take.size} with {color['quantity']} {span}"
                     + (f" (only {inside.size} exist)" if inside.size < share else ""))
    pick = np.concatenate(chosen)
    return all_tags[pick], f"{len(pick)} particles: " + ", ".join(parts)


def background_tags(pf, n: int, exclude, seed: int) -> np.ndarray:
    """``n`` random tags of file ``pf`` other than ``exclude``."""
    if n <= 0:
        return np.empty(0, dtype=np.int64)
    tags = pf.tags()
    tags = tags[~np.isin(tags, exclude)]
    rng = np.random.default_rng(seed + 1)
    return np.sort(rng.choice(tags, size=min(n, tags.size), replace=False))


# ── Trajectories (cached) ────────────────────────────────────────────────

def gather(files, tags, color, params, opts, cache: Path | None, workers: int) -> dict:
    """
    Positions and colour of ``tags`` in every file of ``files``: arrays of
    shape (n_files, n_tags), from ``cache`` if it holds the same request.
    """
    key = json.dumps({"color": color and [color["quantity"], color["units"]],
                      "stored_units": opts["stored_units"],
                      "rdata_names": list(opts["rdata_names"]),
                      "idata_names": list(opts["idata_names"])}, sort_keys=True)
    numbers = np.array([f.number for f in files])
    if cache is not None and cache.exists() and not opts["refresh"]:
        try:
            with np.load(cache) as old:
                if (str(old["key"]) == key and np.array_equal(old["tags"], tags)
                        and np.array_equal(old["numbers"], numbers)):
                    print(f"trajectories from cache {cache.name} (--refresh to reread)")
                    return {k: old[k] for k in ("x", "y", "z", "color")}
        except (OSError, KeyError, ValueError):
            pass

    print(f"reading {len(tags)} particles from {len(files)} files ...")
    jobs = [(pf, tags, color, params, opts["stored_units"]) for pf in files]
    out = {k: np.full((len(files), len(tags)), np.nan) for k in ("x", "y", "z", "color")}
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
            results = pool.map(_gather_star, jobs, chunksize=max(1, len(jobs) // (8 * workers)))
            for i, res in enumerate(results):
                for k in out:
                    out[k][i] = res[k]
    else:
        for i, job in enumerate(jobs):
            res = gather_file(*job)
            for k in out:
                out[k][i] = res[k]
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, key=key, tags=tags, numbers=numbers, **out)
    return out


# ── Drawing ──────────────────────────────────────────────────────────────

def colormap(name: str, cmap_range=None):
    """Colormap ``name``, cut to the fraction ``cmap_range`` = (lo, hi) of it if given
    (e.g. (0.12, 1) drops turbo's near-black end, invisible on a dark background)."""
    cmap = plt.get_cmap(name)
    if not cmap_range or tuple(cmap_range) == (0, 1):
        return cmap
    lo, hi = cmap_range
    return colors.ListedColormap(cmap(np.linspace(lo, hi, 256)), name=f"{name}_cut")


def make_norm(norm: str | None, vmin, vmax):
    if norm == "log":
        return colors.LogNorm(vmin=vmin, vmax=vmax)
    return colors.Normalize(vmin=vmin, vmax=vmax)


def trail_segments(h, v, c, periodic_h: float | None, periodic_v: float | None):
    """
    Segments of the trails: ``h, v, c`` are (n_steps, n_particles), oldest
    first.  Returns (segments (m, 2, 2), colour values (m,), age (m,) in
    [0, 1], 1 = newest).  Segments with a missing end, or crossing a periodic
    boundary (a jump of more than half the box), are left out.
    """
    steps = h.shape[0]
    if steps < 2:
        return np.empty((0, 2, 2)), np.empty(0), np.empty(0)
    p0 = np.stack([h[:-1], v[:-1]], axis=-1)
    p1 = np.stack([h[1:], v[1:]], axis=-1)
    ok = np.isfinite(p0).all(-1) & np.isfinite(p1).all(-1)
    for k, length in ((0, periodic_h), (1, periodic_v)):
        if length:
            with np.errstate(invalid="ignore"):
                ok &= ~(np.abs(p1[..., k] - p0[..., k]) > 0.5 * length)
    age = np.broadcast_to(((np.arange(steps - 1) + 1) / (steps - 1))[:, None], ok.shape)
    segments = np.stack([p0[ok], p1[ok]], axis=1)
    return segments, c[1:][ok], age[ok]


def draw_frame(job) -> Path:
    """Draw one image; ``job`` is built in main().  Returns the file written."""
    (i, number, t_title, h, v, c, ntrack, gas, opts) = job
    apply_fonts()
    style = STYLES[opts["style"]]
    cvar = opts["color_var"]
    norm = make_norm(cvar["norm"], cvar["vmin"], cvar["vmax"]) if cvar else None
    cmap = colormap(cvar["cmap"], opts["cmap_range"]) if cvar else None

    with plt.style.context(style["base"]):
        apply_fonts()
        fig, ax = plt.subplots(figsize=opts["figsize"], layout="constrained")
        fig.set_facecolor(style["face"])
        ax.set_facecolor(style["face"])

        if gas is not None:
            frame, params = gas
            gvar = opts["gas_var"]
            x, y, data = slice_variable(frame_plane(frame, params, opts), gvar, params, opts)
            mesh = draw_slice(ax, x, y, data, cmap=gvar["cmap"], norm=gvar["norm"],
                              vmin=gvar["vmin"], vmax=gvar["vmax"], colorbar=False,
                              axis=opts["axis"], length_units=opts["length_units"])
            mesh.set_alpha(opts["gas_alpha"])
            cb = fig.colorbar(mesh, ax=ax, shrink=0.8, pad=0.01)
            cb.set_label(f"gas: {gvar['label']}")
            cb.solids.set_alpha(1)
            frame.unload()

        hn, vn, cn = h[-1], v[-1], c[-1]
        # Background particles: faint dots, no trails.
        if hn.size > ntrack:
            hb, vb, cb_ = hn[ntrack:], vn[ntrack:], cn[ntrack:]
            ok = np.isfinite(hb) & np.isfinite(vb)
            kw = (dict(c=cb_[ok], cmap=cmap, norm=norm) if cvar
                  else dict(color=style["single"]))
            ax.scatter(hb[ok], vb[ok], s=opts["background_size"], alpha=opts["background_alpha"],
                       linewidths=0, rasterized=True, zorder=2, **kw)

        # Trails of the tracked particles.
        segs, cseg, age = trail_segments(h[:, :ntrack], v[:, :ntrack], c[:, :ntrack],
                                         opts["periodic_h"], opts["periodic_v"])
        if len(segs):
            if cvar:
                rgba = cmap(norm(np.ma.masked_invalid(cseg)))
            else:
                rgba = np.tile(colors.to_rgba(style["single"]), (len(segs), 1))
            rgba[:, 3] = opts["trail_alpha"] * age**1.5
            lc = LineCollection(segs, colors=rgba, linewidths=0.6 + 2.0 * age,
                                capstyle="round", zorder=3)
            ax.add_collection(lc)

        # The tracked particles, with a soft glow.
        ht, vt, ct = hn[:ntrack], vn[:ntrack], cn[:ntrack]
        ok = np.isfinite(ht) & np.isfinite(vt)
        if ok.any():
            kw = (dict(c=ct[ok], cmap=cmap, norm=norm) if cvar
                  else dict(color=style["single"]))
            ax.scatter(ht[ok], vt[ok], s=opts["marker_size"] * 2.5, alpha=0.25,
                       linewidths=0, zorder=4, **kw)
            ax.scatter(ht[ok], vt[ok], s=opts["marker_size"], edgecolors=style["edge"],
                       linewidths=1.0, zorder=5, **kw)

        if cvar:
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            fig.colorbar(sm, ax=ax, shrink=0.8, pad=0.01, label=cvar["label"])

        hl, vl = opts["xlim"], opts["ylim"]
        ax.set_xlim(*hl)
        ax.set_ylim(*vl)
        ax.set_aspect("equal")
        suffix = "" if opts["length_units"] == "code" else f" [{opts['length_units']}]"
        ax.set_xlabel(opts["h_axis"] + suffix)
        ax.set_ylabel(opts["v_axis"] + suffix)
        ax.grid(color=style["grid"], linewidth=0.5, alpha=0.6, zorder=0)
        title = f"{opts['title']}  |  frame {i}/{opts['n_frames'] - 1}  |  {t_title}"
        ax.set_title(title, fontsize=plt.rcParams["axes.titlesize"] * 0.8)

        path = Path(opts["outdir"]) / f"{opts['name']}.{opts['axis']}.{number:05d}.{opts['format']}"
        fig.savefig(path, dpi=opts["dpi"], facecolor=style["face"])
        plt.close(fig)
    return path


# ── Info ─────────────────────────────────────────────────────────────────

def print_info(pdata: ParticleData, params: dict) -> None:
    """Summary of the particle files, to check columns, units and positions."""
    first, last = pdata[0], pdata[-1]
    print(pdata)
    print(f"files {pdata.numbers[0]}..{pdata.numbers[-1]}  ({len(pdata)} files), "
          f"t = {first.time:g} .. {last.time:g} (code units), {first.n} particles in the first")
    print(f"columns: {first.names}")
    dom = domain(params)
    print(f"\n{'column':>12} {'min':>12} {'median':>12} {'max':>12}   (file {first.number})")
    for name in first.names:
        values = first[name]
        if values.size == 0:
            continue
        print(f"{name:>12} {np.nanmin(values):12.4g} {np.nanmedian(values):12.4g} "
              f"{np.nanmax(values):12.4g}")
        if name in dom and dom[name][1] > dom[name][0]:
            lo, hi, _ = dom[name]
            tol = 1e-6 * (hi - lo)
            if values.min() < lo - tol or values.max() > hi + tol:
                print(f"{'':>12} ^ outside the mesh [{lo:g}, {hi:g}]: the real columns may "
                      "not be x, y, z, vx, vy, vz in that order; set "
                      "PARTICLES['rdata_names']")
    if "temp" in first.names and "units" in params:
        t_unit = Units.from_params(params).temperature_cgs
        med = float(np.nanmedian(first["temp"]))
        print(f"\nstored temp median {med:.4g}: {med * t_unit:.4g} K if stored in code units."
              "  If it is already in K, set PARTICLES['stored_units'] = {'temp': 'K'}")
    tags = first.tags()
    print(f"\ntags {tags.min()}..{tags.max()}, {np.unique(tags).size} unique of {tags.size}")


# ── Command line ─────────────────────────────────────────────────────────

def _limits(text: str):
    lo, hi = text.split(":")
    return (float(lo) if lo else None, float(hi) if hi else None)


def auto_figsize(width: float, height: float, colorbars: int) -> tuple[float, float]:
    """Figure size that fits the view's aspect ratio plus colorbars."""
    aspect = width / height if height > 0 else 1.0
    if aspect <= 1:
        h = 8.0
        w = max(3.5, h * aspect)
    else:
        w = 10.0
        h = max(3.0, w / aspect + 1.0)
    return (w + 1.4 * colorbars, h + 0.6)


def main(argv=None):
    plot, P = config.PLOT, config.PARTICLES
    parser = argparse.ArgumentParser(
        description="Movie of tracked particles with fading trails (config.PARTICLES).")
    parser.add_argument("--run", default=config.RUN, choices=list(config.SIMULATIONS),
                        help=f"simulation in config.SIMULATIONS (default: {config.RUN})")
    parser.add_argument("--info", action="store_true",
                        help="print what the particle files hold, then stop")
    parser.add_argument("--particles", help="folder of the particle files "
                                            "(default: the run's 'particles' entry)")
    parser.add_argument("--stem", default=P.get("stem"),
                        help="particle file name stem, if several sets are found")
    parser.add_argument("--frames", nargs="+",
                        help="particle file numbers and/or ranges like 0:1000:2 "
                             "(default: PARTICLES['frames'], else all)")
    # What to track
    parser.add_argument("--tags", type=int, nargs="+", default=P.get("tags"),
                        help="track these tags (overrides --n / --bins)")
    parser.add_argument("--n", dest="n_track", type=int, default=P["n_track"],
                        help=f"number of particles to track (default: {P['n_track']})")
    parser.add_argument("--bins", type=float, nargs="*", default=P.get("bins"),
                        help="edges in the colour's units: --n is split evenly over the "
                             f"bins (default: {P.get('bins')}); none = random")
    parser.add_argument("--select-at", type=int, default=P.get("select_at"),
                        help="particle file number to pick them in (default: first frame)")
    parser.add_argument("--seed", type=int, default=P.get("seed", 0))
    parser.add_argument("--background", type=int, default=P.get("background", 0),
                        help="other particles drawn as faint dots "
                             f"(default: {P.get('background', 0)})")
    parser.add_argument("--trail", type=int, default=P.get("trail", 50),
                        help="trail length in frames, 0 = none, -1 = whole history "
                             f"(default: {P.get('trail', 50)})")
    # Colour
    parser.add_argument("--color", default=P.get("color"),
                        help="colour quantity: PLOT_VARS key, particle column or derived "
                             "quantity; 'none' for one colour "
                             f"(default: {P.get('color')})")
    parser.add_argument("--units", help="units of the colour (default: from PLOT_VARS)")
    parser.add_argument("--cmap", default=P.get("cmap"), help="colormap (default: PLOT_VARS)")
    parser.add_argument("--norm", choices=["log", "linear"], help="(default: PLOT_VARS)")
    parser.add_argument("--vmin", type=float, default=P.get("vmin"),
                        help="(default: PARTICLES, else PLOT_VARS, else the data range)")
    parser.add_argument("--vmax", type=float, default=P.get("vmax"),
                        help="(default: PARTICLES, else PLOT_VARS, else the data range)")
    # Gas underneath
    parser.add_argument("--gas", default=P.get("gas"),
                        help="grid variable drawn underneath, from the nearest grid output "
                             f"in time; 'none' for none (default: {P.get('gas')})")
    parser.add_argument("--gas-cmap", default=P.get("gas_cmap"),
                        help="colormap of the gas (default: PARTICLES, else PLOT_VARS)")
    parser.add_argument("--gas-alpha", type=float, default=P.get("gas_alpha", 0.6))
    # View
    parser.add_argument("--axis", choices=["x", "y", "z"], default=plot["axis"],
                        help=f"view along this axis; z for 2D runs (default: {plot['axis']})")
    parser.add_argument("--position", type=float, default=plot["position"],
                        help="gas slice position along --axis in --length-units")
    parser.add_argument("--slab", type=float, default=P.get("slab"),
                        help="3D: draw only background particles within this thickness "
                             "of --position (length units; default: all)")
    parser.add_argument("--xlim", type=_limits, default=P.get("xlim"),
                        help="horizontal range as lo:hi in length units (default: domain)")
    parser.add_argument("--ylim", type=_limits, default=P.get("ylim"),
                        help="vertical range as lo:hi in length units (default: domain)")
    parser.add_argument("--length-units", default=P.get("length_units") or plot["length_units"])
    parser.add_argument("--time-units", default=P.get("time_units") or plot["time_units"])
    parser.add_argument("--time-decimals", type=non_negative_int,
                        default=P.get("time_decimals", plot.get("time_decimals", 2)))
    parser.add_argument("--style", choices=list(STYLES), default=P.get("style", "dark"))
    parser.add_argument("--marker-size", type=float, default=P.get("marker_size", 120))
    parser.add_argument("--title", default=P.get("title"), help="(default: the run name)")
    parser.add_argument("--figsize", type=float, nargs=2, default=P.get("fig_size"),
                        help="(default: fitted to the view)")
    parser.add_argument("--dpi", type=int, default=P.get("dpi", plot["dpi"]))
    parser.add_argument("--format", default=plot["format"])
    parser.add_argument("--name", default=P.get("name", "particles"),
                        help="file name prefix of images, video and cache")
    parser.add_argument("--outdir", help="(default: <run's out>/particles)")
    parser.add_argument("--refresh", action="store_true",
                        help="reread the trajectories instead of using the cache")
    parser.add_argument("--video", action=argparse.BooleanOptionalAction,
                        default=P.get("video", True), help="make the mp4")
    parser.add_argument("--save-png", action=argparse.BooleanOptionalAction, default=None,
                        help="keep the images once the video is made "
                             f"(default: config.MAIN = {config.MAIN['save_png']})")
    parser.add_argument("--fps", type=float, default=P.get("fps", 25))
    parser.add_argument("--max-size", type=int,
                        default=config.COMBINED.get("video_max_size", 4096))
    parser.add_argument("-w", "--workers", "-c", "--cores", dest="workers", type=int,
                        default=n_workers(),
                        help=f"parallel worker processes (default: {n_workers()})")
    args = parser.parse_args(argv)

    # ── Run, particle files ──
    try:
        run = config.resolve(args.run)
    except RuntimeError as err:
        parser.error(str(err))
    params = _params(run["athinput"])
    basename = params.get("job", {}).get("basename")
    folders = ([Path(args.particles)] if args.particles
               else [run["particles"]] if run.get("particles")
               else [run["data"], run["data"].parent])
    rdata = tuple(P.get("rdata_names") or DEFAULT_RDATA_NAMES)
    idata = tuple(P.get("idata_names") or DEFAULT_IDATA_NAMES)
    try:
        pdata = ParticleData(folders, basename=basename, stem=args.stem,
                             rdata_names=rdata, idata_names=idata)
    except (FileNotFoundError, ValueError) as err:
        parser.error(f"{err}\n  set 'particles' in config.SIMULATIONS['{args.run}'] "
                     "or pass --particles")
    if args.info:
        print_info(pdata, params)
        return

    wanted = parse_frames(args.frames)
    if wanted is None and P.get("frames") is not None:
        wanted = set(P["frames"])
    files = [f for f in pdata if wanted is None or f.number in wanted]
    if not files:
        parser.error(f"no particle files selected; numbers {pdata.numbers[0]}.."
                     f"{pdata.numbers[-1]}")

    # ── Settings ──
    try:
        length = unit_factor(params, "length", args.length_units)
        unit_factor(params, "time", args.time_units)
    except ValueError as err:
        parser.error(str(err))
    color_name = None if str(args.color).lower() in ("none", "") else args.color
    color = None
    if color_name:
        color = var_settings(color_name, {"cmap": args.cmap, "norm": args.norm,
                                          "vmin": args.vmin, "vmax": args.vmax,
                                          "units": args.units})
        if color["label"] == color_name and color["units"] != "code":
            color["label"] = f"{color_name} [{color['units']}]"
        try:
            if color["units"] != "code":
                Units.from_params(params).factor(particle_dimension(color["quantity"]),
                                                 color["units"])
        except ValueError as err:
            parser.error(f"colour {color_name}: {err}")
    stored = dict(P.get("stored_units") or {})
    opts = dict(tags=args.tags, n_track=args.n_track, bins=args.bins, seed=args.seed,
                region=P.get("region"), length_units=args.length_units,
                stored_units=stored, rdata_names=rdata, idata_names=idata,
                refresh=args.refresh)

    # ── Selection ──
    by_number = {f.number: f for f in pdata}
    select_file = by_number.get(args.select_at, files[0]) if args.select_at is not None \
        else files[0]
    if args.select_at is not None and args.select_at not in by_number:
        parser.error(f"--select-at {args.select_at}: no such particle file")
    try:
        tracked, described = select_tags(select_file, params, opts, color)
    except KeyError as err:
        parser.error(str(err))
    extra = background_tags(select_file, args.background, tracked, args.seed)
    tags = np.concatenate([tracked, extra])
    print(f"{pdata}\ntracking {described} (picked in file {select_file.number})"
          + (f", plus {len(extra)} background particles" if len(extra) else ""))

    outdir = Path(args.outdir) if args.outdir else run["out"] / "particles"
    outdir.mkdir(parents=True, exist_ok=True)
    track = gather(files, tags, color, params, opts, outdir / f"{args.name}.tracks.npz",
                   args.workers)

    # ── Geometry ──
    hname, vname, nname = PLANE_AXES[args.axis]
    dom = domain(params)
    h = track[hname] * length
    v = track[vname] * length
    c = track["color"]
    if args.slab is not None and len(extra):
        centre = (args.position if args.position is not None
                  else 0.5 * (dom[nname][0] + dom[nname][1]) * length)
        far = np.abs(track[nname] * length - centre) > 0.5 * args.slab
        far[:, :len(tracked)] = False
        h, v = np.where(far, np.nan, h), np.where(far, np.nan, v)

    def view_range(axis, given):
        lo, hi = dom[axis][0] * length, dom[axis][1] * length
        if hi <= lo:  # no extent (e.g. x3 of a 2D run): fit the data
            lo, hi = np.nanmin(track[axis]) * length, np.nanmax(track[axis]) * length
        given = given or (None, None)
        return (lo if given[0] is None else given[0], hi if given[1] is None else given[1])

    xlim, ylim = view_range(hname, args.xlim), view_range(vname, args.ylim)

    # Colour range over the whole movie, unless configured.
    if color is not None:
        values = c[np.isfinite(c)]
        if color["norm"] == "log":
            values = values[values > 0]
        if values.size:
            color["vmin"] = float(values.min()) if color["vmin"] is None else color["vmin"]
            color["vmax"] = float(values.max()) if color["vmax"] is None else color["vmax"]

    # ── Gas underneath ──
    gas_name = None if str(args.gas).lower() in ("none", "") else args.gas
    gas_frames = [None] * len(files)
    gas_var = None
    if gas_name:
        sim = SimulationData(run["athinput"], run["data"], outputs=run["outputs"])
        gas_var = var_settings(gas_name, {"cmap": args.gas_cmap})
        if gas_var["vmin"] is None or gas_var["vmax"] is None:
            print(f"note: PLOT_VARS has no vmin/vmax for {gas_name}; its colours rescale "
                  "every frame")
        gtimes = sim.times
        for k, pf in enumerate(files):
            j = int(np.argmin(np.abs(gtimes - pf.time))) if np.isfinite(pf.time) else 0
            gas_frames[k] = (sim.frames[j], sim.params)
        gap = max(abs(g[0].time - pf.time) for g, pf in zip(gas_frames, files))
        print(f"gas: {gas_name} from {sim}; largest time offset to a particle frame "
              f"{gap:.3g} (code units)")

    figsize = tuple(args.figsize) if args.figsize else auto_figsize(
        xlim[1] - xlim[0], ylim[1] - ylim[0], (color is not None) + (gas_name is not None))
    plot_opts = dict(
        style=args.style, color_var=color, cmap_range=P.get("cmap_range"), gas_var=gas_var, gas_alpha=args.gas_alpha,
        axis=args.axis, position=args.position, length_units=args.length_units,
        time_units=args.time_units, time_decimals=args.time_decimals,
        marker_size=args.marker_size, background_size=P.get("background_size", 4),
        background_alpha=P.get("background_alpha", 0.5), trail_alpha=P.get("trail_alpha", 0.9),
        periodic_h=(dom[hname][1] - dom[hname][0]) * length if dom[hname][2] else None,
        periodic_v=(dom[vname][1] - dom[vname][0]) * length if dom[vname][2] else None,
        xlim=xlim, ylim=ylim, h_axis=hname, v_axis=vname,
        title=args.title or args.run, n_frames=len(files), figsize=figsize,
        dpi=args.dpi, format=args.format, name=args.name, outdir=str(outdir),
    )

    trail = len(files) if args.trail < 0 else args.trail
    jobs = []
    for i, pf in enumerate(files):
        i0 = max(0, i - trail)
        t_title = (time_title(pf, params, plot_opts) if np.isfinite(pf.time)
                   else f"file {pf.number}")
        # Only the trail window of the tracked particles; background at frame i only.
        hw, vw, cw = h[i0:i + 1].copy(), v[i0:i + 1].copy(), c[i0:i + 1].copy()
        hw[:-1, len(tracked):] = vw[:-1, len(tracked):] = np.nan
        jobs.append((i, pf.number, t_title, hw, vw, cw, len(tracked), gas_frames[i], plot_opts))

    print(f"drawing {len(jobs)} frames -> {outdir}")
    written = []
    if args.workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
            for k, path in enumerate(pool.map(draw_frame, jobs, chunksize=4)):
                written.append(path)
                if k % 50 == 0 or k == len(jobs) - 1:
                    print(f"  {k + 1}/{len(jobs)}: {path.name}")
    else:
        for k, job in enumerate(jobs):
            written.append(draw_frame(job))
            if k % 50 == 0 or k == len(jobs) - 1:
                print(f"  {k + 1}/{len(jobs)}: {written[-1].name}")

    if args.video:
        from .video import make_video
        out = make_video(written, outdir / f"{args.name}.{args.axis}.mp4", fps=args.fps,
                         max_size=args.max_size or None)
        print(f"video: {out}")
        if not (config.MAIN["save_png"] if args.save_png is None else args.save_png):
            for path in written:
                path.unlink(missing_ok=True)
            print(f"deleted the {len(written)} images (save_png = False)")


def _params(athinput: Path) -> dict:
    if "." in __package__:
        from ..simulation_data.athinput import parse_athinput
    else:
        from simulation_data.athinput import parse_athinput
    return parse_athinput(athinput)


if __name__ == "__main__":
    main()
