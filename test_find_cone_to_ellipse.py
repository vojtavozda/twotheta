
# %%
import os
import numpy as np
from numpy import sin, cos, tan, pi, sqrt
import elliptools as el
from matplotlib import pyplot as plt
import conetools
import importlib
from clrprint import printc
from genlib import plt_clrs
importlib.reload(conetools)
importlib.reload(el)


# change current directory to the one where this script is located
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# We know theta from the experiment
theta = 20/180*pi

ellipse = el.Ellipse(x0=1,y0=2,a=2,b=1,phi=0.5)
cone = ellipse.findCone(theta)


# Print the parameters
ellipse.print()
cone.print()

# -----[ Create the figure ]-----
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.set_proj_type('ortho',None) # persp,0.1

cone.plotMesh(ax,5)
ellipse.plot(ax,plotAxes=True)
ellipse.plotCone(ax)

xlim, ylim, zlim = ([-4,3],[-4,4],[0,4])
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_zlim(zlim)


ax.set_aspect('equal')
plt.tight_layout()

# ax.view_init(90,-90,0) # Top view
# ax.view_init(0,-90,0) # Side view
plt.show()

