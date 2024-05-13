
# %%

import elliptools as ellt
import numpy as np
from numpy import sin, cos, tan, pi, sqrt
from genlib import plt_clrs
import importlib
import os
import matplotlib.pyplot as plt
importlib.reload(ellt)

def parametricCone(cone:ellt.Cone,t:float,phi:float):

    n = cone.n

    u = np.array([1,0,-n[0]/n[2]])
    u = u/np.linalg.norm(u)

    v = np.array([0,1,-n[1]/n[2]])
    v = v/np.linalg.norm(v)

    r = t*tan(cone.theta)

    return cone.apex+t*n + r*cos(phi)*u + r*sin(phi)*v


V = np.array([0,0,3])
n = np.array([0,0.6,-1])
n = n/np.linalg.norm(n)
theta = 0.2
cone = ellt.Cone(V,n,theta)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
cone.plotWireframe(ax)
cone.plotMesh(ax,3)
cone.getEllipse().plot(ax,plotAxes=True)

for t in np.linspace(1,3,2):
    for a in np.linspace(0,2*pi,20):
        P = parametricCone(cone,t,a)
        ax.scatter(P[0],P[1],P[2],color='b')

# Equalize aspect ratio
ax.set_aspect('equal')
plt.show()