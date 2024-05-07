# %%

from pyFAI.utils.ellipse import fit_ellipse
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches

def display(ptx, pty, ellipse=None):
    """A function to overlay a set of points and the calculated ellipse
    """
    fig = plt.figure()
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
        patch = patches.Arc((x0, y0), width=wlong*2, height=wshort*2, angle=np.rad2deg(angle))
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
    # ax.set_xlim(*xlim)
    # ax.set_ylim(*ylim)
    plt.gca().invert_yaxis()
    plt.show()

x = np.load('data/x_2.npy')
y = np.load('data/y_2.npy')
# conic = np.load('data/conic_104.npy')
# x = conic[0]
# y = conic[1]

ellipse = fit_ellipse(y,x)
print(ellipse)
display(x,y,ellipse)
