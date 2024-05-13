
# %%

import os
import numpy as np
from numpy import sin, cos, tan, pi, sqrt
import elliptools as ellt
from matplotlib import pyplot as plt
import importlib
from clrprint import printc
from genlib import plt_clrs
from scipy import optimize
importlib.reload(ellt)


# change current directory to the one where this script is located
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ------------------------------------------------------------------------------
# Calculate two theta angles
energy = 5.932                  # [keV]
wavelength = 12.3985/energy     # [A]

d_012 = 3.4662      # hkl 012 [A]
d_104 = 2.5429      # hkl 104 [A]
d_110 = 2.3730      # hkl 110 [A]
d_113 = 2.0805      # hkl 113 [A]

# Two theta calculation follows from Bragg's equation `lambda = 2*d*sin(theta)`
two_theta_012 = 2*np.arcsin(wavelength/2/d_012)
two_theta_104 = 2*np.arcsin(wavelength/2/d_104)
two_theta_110 = 2*np.arcsin(wavelength/2/d_110)
two_theta_113 = 2*np.arcsin(wavelength/2/d_113)

data = np.load('data/data.npy')
data[data>30] = 30
data[data<0] = 0

conic_012 = np.load('data/conic_012.npy')
conic_104 = np.load('data/conic_104.npy')
conic_110 = np.load('data/conic_110.npy')
conic_113 = np.load('data/conic_113.npy')

el012 = ellt.Ellipse(xData=conic_012[0],yData=conic_012[1],color=plt_clrs[0],theta=two_theta_012)
el104 = ellt.Ellipse(xData=conic_104[0],yData=conic_104[1],color=plt_clrs[2],theta=two_theta_104)
el110 = ellt.Ellipse(xData=conic_110[0],yData=conic_110[1],color=plt_clrs[3],theta=two_theta_110)

el012.fit()
el012.findCone()

el104.fit()
el104.findCone()

el110.fit()
el110.findCone()

z0 = 1184.6707
phi   = -1.0414    /180*pi
theta = -32.5151     /180*pi
psi   = -0.0541     /180*pi
shift_x = -163.8418
shift_y = -1029.1911
P = np.array([-shift_x,-shift_y,z0])
R = ellt.rotate3D(P,-phi,-theta,-psi)
R = np.array([-144.28757645 ,-870.36951104 ,629.94872884])
print(f"R = {R}")
n = -R/np.linalg.norm(R)
print(n)
theta = two_theta_012
cone = ellt.Cone(R,n,two_theta_012)


fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.set_proj_type('ortho',None) # persp,0.1

plt.plot(conic_012[0],conic_012[1])
plt.plot(conic_104[0],conic_104[1])
plt.plot(conic_110[0],conic_110[1])

# cone.plotWireframe(ax)
# cone.getEllipse().plot(ax)
cone.theta = two_theta_104
cone.setColor(3)
cone.plotMesh(ax,1000)
cone.plotWireframe(ax)
cone.getEllipse().plot(ax,plotAxes=True)

ax.set_xlim(-2000,2000)
ax.set_ylim(-2000,2000)
ax.set_zlim(0,1000)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# top view
# ax.view_init(elev=90,azim=0)
ax.set_aspect('equal')
plt.show()
