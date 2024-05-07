
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

el012 = el.Ellipse(xData=conic_012[0],yData=conic_012[1],color=plt_clrs[0],theta=two_theta_012)
el104 = el.Ellipse(xData=conic_104[0],yData=conic_104[1],color=plt_clrs[1],theta=two_theta_104)
el110 = el.Ellipse(xData=conic_110[0],yData=conic_110[1],color=plt_clrs[2],theta=two_theta_110)

el012.fit()
el012.print()
el012.findCone()

n = el012.cone.n
assert np.linalg.norm(n) == 1, "Cone axis must be a unit vector!"
print(n)
phi = pi-np.arccos(n[0]/sqrt(n[0]**2+n[1]**2))
print(phi)
n2 = el.rotate3D(n,phi,0,0)
print(n2)
n3 = el.rotate3D(n2,0,el012.theta,0)
print(n3)
n4 = el.rotate3D(n3,phi,0,0)

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.set_proj_type('ortho',None) # persp,0.1

el012.plot(ax,plotAxes=True,plotData=True,plotCone=True)
V = el012.cone.apex
c = 1500
plt.plot([V[0],V[0]+n4[0]*c],[V[1],V[1]+n4[1]*c],[V[2],V[2]+n4[2]*c],color='r')

# %%
el104.fit()
el110.fit()
el012.print()

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.set_proj_type('ortho',None) # persp,0.1
# ax = fig.add_subplot(111)
# plt.imshow(data)

el012.plot(ax,plotAxes=True,plotData=True,plotCone=True)
# el104.plot(ax,plotAxes=True,plotData=True,plotCone=True)
# el110.plot(ax,plotAxes=True,plotData=True,plotCone=True)

# top view
ax.view_init(elev=90,azim=0)
plt.show()
