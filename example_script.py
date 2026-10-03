# Poking around the KH2D test run in test_data/kh2d with SimulationData.

import numpy as np

from simulation_data import Frame, SimulationData, parse_athinput


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
