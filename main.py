"""
Entry point: plot 2D slices of the run set up in config.py.

    python main.py                           # PLOT_ORDER for config.RUN, every frame
    python main.py dens t_cool --frames 0:21:5 -c 4

What is plotted comes from config.py (RUN, SIMULATIONS, PLOT, PLOT_VARS,
PLOT_ORDER) and the root folders from .env.  Any arguments are passed on to
plotting/slice2d.py; ``python main.py --help`` lists them.
"""

if __package__:  # imported as vis_athenak.main
    from .plotting import slice2d
else:            # python main.py
    from plotting import slice2d


def main(argv=None):
    slice2d.main(argv)


if __name__ == "__main__":
    main()
