# Poking around the KH cooling test run in test_data/kh_cooling_64x128 with SimulationData.

import time

import numpy as np

from data_processing import athena_read
from simulation_data import Field, Frame, SimulationData, parse_athinput


# Change this to the folder path on your system
athinput = "test_data/kh_cooling_64x128/kh_cooling.athinput"
datafolder = "test_data/kh_cooling_64x128/bin"

# load the simulation (this doesn't read any data yet, just finds the files)
sim = SimulationData(athinput=athinput, datafolder=datafolder)
print(sim)

# how many frames do we have?
print("number of frames:", len(sim))
print("frame numbers:", sim.frame_numbers)

# You can have multiple different files for the same timestep. All those are stored together in the SimulationData class.
# which outputs got combined into each frame
print("outputs:", sim.outputs)

# what's in the athinput?
print("basename:", sim.basename)
print("The params Dictionary", type(sim.params))
print("Keys in the Params Dictionary correspond to the blocks", sim.params.keys())
# The contents of each block can be further accessed like a dictionary parsing.
print("mesh:", sim.params["mesh"]["nx1"], "x", sim.params["mesh"]["nx2"])
print("gamma:", sim.params["hydro"]["gamma"])

# The SimulationData class has list indexing.
print(sim[0])
print(sim[-1])
print(sim[2:5])

# times of all the frames (only reads the file headers, so it's quick)
print("times:", sim.times)
print("last frame is at t =", sim[-1].time)

# nothing has been loaded so far
print("anything loaded?", any(fr.is_loaded for fr in sim.frames))

# Loading just the last frame.
frame = sim[-1]
print(frame)            # not loaded yet
rho = frame["dens"]     # this is where the files get read
print(frame)            # now it's loaded

print("files for this frame:", frame.paths)
print("what's in it:", list(frame.keys()))

# density
print("dens shape:", rho.shape)     # (nx3, nx2, nx1)
print("dens type:", rho.dtype)
print("dens min/max:", rho.min(), rho.max())

# coordinates
x = frame["x1v"]   # cell centres in x
y = frame["x2v"]   # cell centres in y
print("x goes from", x[0], "to", x[-1], "with", len(x), "cells")
print("y goes from", y[0], "to", y[-1], "with", len(y), "cells")

# no magnetic field in this run
print("bcc1 in frame?", "bcc1" in frame)

# hydro_w and hydro_u are in the same frame, so rho*vx should equal mom1
diff = frame["dens"] * frame["velx"] - frame["mom1"]
print("max |rho*vx - mom1|:", np.abs(diff).max())

# asking for something that isn't there gives a KeyError listing what is
try:
    frame["temperature"]
except KeyError as err:
    print("KeyError:", err)

# throw away the data to free memory (it reloads if we touch it again)
frame.unload()
print(frame)

# only load hydro_w
sim_w = SimulationData(athinput, datafolder, outputs=["hydro_w"])
print(sim_w)
print("mom1 in frame?", "mom1" in sim_w[0])

# double precision instead of the default float32
sim_64 = SimulationData(athinput, datafolder, dtype=np.float64)
print("dtype:", sim_64[0]["dens"].dtype)

# you can also just parse the athinput yourself
params = parse_athinput(athinput)
print("tlim:", params["time"]["tlim"])

# or make a Frame from a single file
single = Frame(number=0, paths={"hydro_w": sim[0].paths["hydro_w"]})
print(single, "t =", single.time)
print("dens mean:", single["dens"].mean())


# ── Fields ──────────────────────────────────────────────────────────────

# Every frame has a dict of Field objects, one per variable. Building it only
# reads the file headers, so no data is loaded yet.
frame = sim[-1]
print(frame)
for key, field in frame.fields.items():
    print(f"  {key:14s} -> {field}")

# A Field knows its variable name, which output wrote it, and which file it's in
field = frame.fields["dens"]
print("name:", field.name, " output:", field.output, " path:", field.path)
print("is it a Field?", isinstance(field, Field))

# field.data reads the array the first time and caches it.
# frame["dens"] is the same thing as frame.fields["dens"].data
print("loaded before .data?", field.is_loaded)
print("same array?", field.data is frame["dens"])
print("loaded after .data?", field.is_loaded)

# Fields work with numpy directly, no need for .data
print("mean log10(dens):", np.log10(field).mean())

# drop just this field from memory (it gets re-read if used again)
field.unload()
print("loaded after unload?", field.is_loaded)

# dens is written by both hydro_w and hydro_u. Neither gets overwritten:
# the first output's is "dens", the other one is "hydro_u/dens".
print(frame.fields["dens"], frame.fields["hydro_u/dens"])
print("same values?", np.allclose(frame["dens"], frame["hydro_u/dens"]))
frame.unload()

# ── What gets read: .bin ────────────────────────────────────────────────

# A .bin file can only be read whole, so asking for dens loads every field
# of the hydro_w file. The hydro_u file isn't touched though.
frame = sim[-1]
frame["dens"]
print(frame)
print("loaded:", [k for k, f in frame.fields.items() if f.is_loaded])
frame.unload()

# ── What gets read: .athdf ──────────────────────────────────────────────

# The same run converted to .athdf
athinput_h5 = "test_data/kh_cooling_64x128_athdf/kh_cooling.athinput"
datafolder_h5 = "test_data/kh_cooling_64x128_athdf/athdf"
sim_h5 = SimulationData(athinput_h5, datafolder_h5)
print(sim_h5, "files:", sim_h5[0].paths)

# From .athdf only the field you ask for is read
frame = sim_h5[-1]
frame["dens"]
print(frame)
print("loaded:", [k for k, f in frame.fields.items() if f.is_loaded])

# The coordinates aren't read either until you ask for them
x = frame["x1v"]
print("x1v has", len(x), "cells")

# same numbers as the .bin run
print("max |dens_athdf - dens_bin|:", np.abs(frame["dens"] - sim[-1]["dens"]).max())

# If you need several fields, load them together: one read per file
# instead of one per field
frame.load(["velx", "vely", "mom1"])
print(frame)
frame.unload()

# ── Why per-field loading is faster ─────────────────────────────────────

# Reading dens from the first 200 frames, three ways.
n = 200

# 1. athena_read.athdf on the whole file: every variable, plus coordinates
t0 = time.perf_counter()
for fr in sim_h5.frames[:n]:
    athena_read.athdf(str(fr.paths["hydro_w"]))["dens"]
t_full = time.perf_counter() - t0

# 2. athena_read.athdf with quantities=["dens"]: one variable, but it still
#    rebuilds the coordinates on every call
t0 = time.perf_counter()
for fr in sim_h5.frames[:n]:
    athena_read.athdf(str(fr.paths["hydro_w"]), quantities=["dens"])["dens"]
t_quant = time.perf_counter() - t0

# 3. frame["dens"]: only dens, no coordinates (fresh SimulationData so
#    nothing is cached from above)
sim_h5 = SimulationData(athinput_h5, datafolder_h5)
t0 = time.perf_counter()
for fr in sim_h5:
    if fr.number >= n:
        break
    fr["dens"]
t_field = time.perf_counter() - t0

print(f"athena_read.athdf, all variables : {t_full:.3f} s")
print(f"athena_read.athdf, dens only     : {t_quant:.3f} s")
print(f"frame['dens']                    : {t_field:.3f} s")

# For .bin, the gain is from not reading the other outputs' files:
# dens only needs the hydro_w file, frame.data reads hydro_u as well.
t0 = time.perf_counter()
for fr in sim.frames[:n]:
    fr["dens"]
    fr.unload()
t_one = time.perf_counter() - t0

t0 = time.perf_counter()
for fr in sim.frames[:n]:
    fr.data["dens"]
    fr.unload()
t_all = time.perf_counter() - t0

print(f".bin frame['dens']    : {t_one:.3f} s")
print(f".bin frame.data       : {t_all:.3f} s")

# ── A time series from one field ────────────────────────────────────────

# Since only dens is read per frame, this is quick even over all 1001 frames.
# Looping over sim unloads each frame once we move on, so memory stays flat.
t0 = time.perf_counter()
dens_max = np.array([fr["dens"].max() for fr in sim_h5])
print(f"max dens over {len(sim_h5)} frames in {time.perf_counter() - t0:.2f} s")
print("max dens at t=0:", dens_max[0], " at t =", sim_h5.times[-1], ":", dens_max[-1])
