
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
theta = 30/180*pi

# First, define an ellipse
ellipse = el.Ellipse(x0=1,y0=0.5,a=1,b=0.6,phi=0.3,theta=theta)
ellipse.print(f=5)

# Now we find a cone to that ellipse
cone = ellipse.findCone()
cone.setColor(1)
cone.print()

# Confirmation: Find ellipse to the cone
ellipse2 = cone.findEllipse()
ellipse2.print(f=5)

# -----[ Create the figure ]-----
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.set_proj_type('ortho',None) # persp,0.1

# Plot the ellipse
ellipse.plot(ax,plotAxes=True)

# Plot the cone.
# This functions call `Cone::getEllipse()` and plot cone properties according to
# found ellipse. So 'FindEllipseToCone' and 'FindConeToEllipse' should be
# consistent as this cone fits the ellipse.
cone.plotMesh(ax,2,plotDandelin=True)
cone.plotWireframe(ax,plotDandelin=True)

# ax.view_init(90,-90,0) # Top view
# ax.view_init(0,-90,0) # Side view
# plt.show()

ax.set_aspect('equal')
plt.tight_layout()
plt.show()