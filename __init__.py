"""
vis_athenak: visualization and analysis tools for AthenaK simulation outputs.

Clone the repo into your project folder and import it as a package:

    from vis_athenak import SimulationData

    sim = SimulationData("path/to/athinput", "path/to/outputs")
"""

from .simulation_data import Frame, SimulationData, parse_athinput

__all__ = ["Frame", "SimulationData", "parse_athinput"]
