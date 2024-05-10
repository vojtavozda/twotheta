
# %%
import os
import numpy as np
from numpy import sin, cos, tan, pi, sqrt
import elliptools as el
from matplotlib import pyplot as plt
import importlib
from clrprint import printc
from genlib import plt_clrs
importlib.reload(el)


# change current directory to the one where this script is located
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# We know theta from the experiment
theta = 30/180*pi

# First, define an ellipse
ellipse = el.Ellipse(x0=0,y0=0,a=1,b=0.6,phi=0)

P = np.array([0.25,0.4])
ellipse.setData(np.array([P[0]]),np.array([P[1]]))
distP = ellipse.find_distance2(P)
print(distP)
print(ellipse.getSOS2())

# -----[ Create the figure ]-----
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.set_proj_type('ortho',None) # persp,0.1

# Plot the ellipse
ellipse.plot(ax,plotAxes=True)
plt.plot(P[0],P[1],'ro')


ax.view_init(90,-90,0)
ax.set_aspect('equal')
plt.tight_layout()
plt.show()