
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

ellipse = el.Ellipse(x0=0,y0=0,a=2,b=1.5,phi=0.3,theta=theta)
cone = ellipse.findCone()

# Print the parameters
ellipse.print()
cone.print()

# -----[ Create the figure ]-----
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.set_proj_type('ortho',None) # persp,0.1

ellipse.plot(ax,plotAxes=True)
ellipse.plotCone(ax)

# Here we find and plot points where cone intersects the semi-major axis
n = ellipse.cone.n
assert np.linalg.norm(n) == 1, "Cone axis must be a unit vector!"
plt.plot([0,n[0]],[0,n[1]],[0,n[2]],'k')
phi = pi-np.arccos(n[0]/sqrt(n[0]**2+n[1]**2))
print(phi)
n2 = el.rotate3D(n,phi,0,0)
# plt.plot([0,n2[0]],[0,n2[1]],[0,n2[2]],'r')
print(n2)
n3 = np.zeros(3)
n3[0] = n2[0]*cos(ellipse.theta)-n2[2]*sin(ellipse.theta)
n3[1] = n2[1]
n3[2] = n2[0]*sin(ellipse.theta)+n2[2]*cos(ellipse.theta)
# plt.plot([0,n3[0]],[0,n3[1]],[0,n3[2]],'g')
print(n3)
n4 = el.rotate3D(n3,-phi,0,0)
n5 = np.zeros(3)
n5[0] = n2[0]*cos(ellipse.theta)+n2[2]*sin(ellipse.theta)
n5[1] = n2[1]
n5[2] = -n2[0]*sin(ellipse.theta)+n2[2]*cos(ellipse.theta)
n6 = el.rotate3D(n5,-phi,0,0)
# plt.plot([0,n4[0]],[0,n4[1]],[0,n4[2]],'b')

V = ellipse.cone.apex
c = 4
plt.plot([V[0],V[0]+n4[0]*c],[V[1],V[1]+n4[1]*c],[V[2],V[2]+n4[2]*c],'r')
plt.plot([V[0],V[0]+n6[0]*c],[V[1],V[1]+n6[1]*c],[V[2],V[2]+n6[2]*c],'r')
plt.plot(V[0]-V[2]/n4[2]*n4[0],V[1]-V[2]/n4[2]*n4[1],0,'ro')
plt.plot(V[0]-V[2]/n6[2]*n6[0],V[1]-V[2]/n6[2]*n6[1],0,'ro')

v1 = np.array([(V[0]-V[2]/n6[2]*n6[0])-(V[0]-V[2]/n4[2]*n4[0]),
               (V[1]-V[2]/n6[2]*n6[1])-(V[1]-V[2]/n4[2]*n4[1]),0])
v1 = v1/np.linalg.norm(v1)
plt.plot([0,v1[0]],[0,v1[1]],[0,v1[2]],'k')
v = np.array([-v1[1],v1[0],0])
v = v/np.linalg.norm(v)
S = # center of ellipse 
n1 = np.array([V[0]-]) # Vector from apex to center of ellipse
u = np.cross(n1,v)
plt.plot([0,v[0]], [0,v[1]], [0,v[2]],'b')
plt.plot([V[0],V[0]+u[0]],[V[1],V[1]+u[1]],[V[2],V[2]+u[2]],'g')

# Vector `u` defines unit vector of rotation. Now rotate `n` around `u` by
# `theta` using https://en.wikipedia.org/wiki/Rotation_matrix#Axis_and_angle
c = cos(theta)
s = sin(theta)
C = 1-c
x = u[0]
y = u[1]
z = u[2]

Q = np.array((
        (x*x*C+c,x*y*C-z*s,x*z*C+y*s),
        (x*y*C+z*s,y*y*C+c,y*z*C-x*s),
        (x*z*C-y*s,y*z*C+x*s,z*z*C+c)
))

u2 = Q@n
plt.plot([V[0],V[0]+u2[0]],[V[1],V[1]+u2[1]],[V[2],V[2]+u2[2]],'g')


xlim, ylim, zlim = ([-2,2],[-2,2],[-1,2])
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_zlim(zlim)


ax.set_aspect('equal')
plt.tight_layout()

# ax.view_init(90,-90,0) # Top view
# ax.view_init(0,-90,0) # Side view
plt.show()

