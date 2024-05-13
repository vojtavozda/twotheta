# %%

import sys
from matplotlib import pyplot
from pyFAI.utils.ellipse import fit_ellipse
import inspect
from matplotlib import patches
from numpy import rad2deg

def display(ptx, pty, ellipse=None):
    """A function to overlay a set of points and the calculated ellipse
    """
    fig = pyplot.figure()
    ax = fig.add_subplot(111)
    if ellipse is not None:
        error = False
        y0, x0, angle, wlong, wshort = ellipse
        if wshort == 0:
            error = True
            wshort = 0.0001
        if wlong == 0:
            error = True
            wlong = 0.0001
        patch = patches.Arc((x0, y0), width=wlong*2, height=wshort*2, angle=rad2deg(angle))
        if error:
            patch.set_color("red")
        else:
            patch.set_color("green")
        ax.add_patch(patch)

        bbox = patch.get_window_extent()
        ylim = min(y0 - wlong, pty.min()), max(y0 + wlong, pty.max())
        xlim = min(x0 - wlong, ptx.min()), max(x0 - wlong, ptx.max())
    else:
        ylim = pty.min(), pty.max()
        xlim = ptx.min(), ptx.max()
    ax.plot(ptx, pty, "ro", color="blue")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    pyplot.show()

from numpy import sin, cos, random, pi, linspace
arc = 0.8
npt = 100
R = linspace(0, arc * pi, npt)
ptx = 1.5 * cos(R) + 2 + random.normal(scale=0.05, size=npt)
pty = sin(R) + 1. + random.normal(scale=0.05, size=npt)

ellipse = fit_ellipse(pty, ptx)
print(ellipse)
display(ptx, pty, ellipse)

angles = linspace(0, pi / 2, 10)
pty = sin(angles) * 20 + 10
ptx = cos(angles) * 20 + 10
ellipse = fit_ellipse(pty, ptx)
print(ellipse)
display(ptx, pty, ellipse)

angles = linspace(0, pi * 2, 6, endpoint=False)
pty = sin(angles) * 10 + 50
ptx = cos(angles) * 20 + 100
ellipse = fit_ellipse(pty, ptx)
print(ellipse)
display(ptx, pty, ellipse)