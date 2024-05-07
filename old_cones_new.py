# %%

import os
import sys
import numpy as np
from numpy import pi, cos, sin, sqrt, tan
from numpy import linalg as LA
from clrprint import printc

import matplotlib
# WebAgg
# matplotlib.use('module://matplotlib_inline.backend_inline',force=False)
from matplotlib import pyplot as plt
import elliptools as el
from genlib import plt_clrs
import genlib as gl
from scipy import optimize

# %%

# change current directory to the one where this script is located
os.chdir(os.path.dirname(os.path.abspath(__file__)))

"""
Geometry description
====================
All ellipses (observed at detector) lie in xy-plane. All should have same
rotation `phi` but different center `S0=(cx,cy)` and different semi-major and
semi-minor axes `a` and `b`. Power diffraction defines several cones with
different angles `two_theta`. Sample, i.e. the cone apex is placed at point `V`.
Length of the cone axis measured from `V` to the detector is `z0`. Angle between
the cone axis and z-axis is `delta`. This can be also understood as tilt of the
detector to the cone axis. When `delta=0` then all ellipses are reduced to
circles. The cone axis intersects detector plane at point `S`. Distance from `S`
to `S0` is `s`. Direction of the cone axis is defined by unit vector `n`.

Fitting procedure
=================
Step 1: Select one ellipse and fit a cone to this ellipse
Step 2: Find best cone parameters (V,z0,phi)
Objective function calculates N cones, where N is number of ellipses, with
different `two_theta` but same apex `V`, `z0`, `delta` and `phi`. Appropriate
ellipses for these cones are compared to experimental data and sums of squares
are calculated. The function returns total sum of squares and tries to find cone
parameters (V,z0,delta,phi) so this total sum is minimized.


"""


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


def delta_diff(delta,a,z0,two_theta):

    if cos(two_theta+delta)==0 or cos(two_theta-delta)==0:
        return None

    return (a-z0*sin(two_theta)/2*(1/cos(two_theta+delta)+1/cos(two_theta-delta)))**2

def get_delta(a,z0,two_theta):
    """
    Solve 'a=(|SA|+|SB|)/2' for delta:
    """
    sTT = sin(two_theta)
    cTT = cos(two_theta)

    det = z0**2*sTT**2*cTT**2-4*a**2*(cTT**2-1)

    cos_deltaP = (z0*sTT*cTT + sqrt(det)) / (2*a)
    cos_deltaM = (z0*sTT*cTT - sqrt(det)) / (2*a)

    deltaP = np.arccos(cos_deltaP)
    deltaM = np.arccos(cos_deltaM)

    print(f"Found delta+ = {deltaP*180/pi:.0f}° and delta- = {deltaM*180/pi:.0f}°")
    return deltaP

def rotate_point(x,y,x0,y0,phi):
    """
    Rotate point (x,y) around center (x0,y0) by phi [radians].
    Returns new coordinates (x_new,y_new).
    """

    x -= x0
    y -= y0

    x_new = x*cos(phi) - y*sin(phi)
    y_new = x*sin(phi) + y*cos(phi)

    x_new += x0
    y_new += y0

    return x_new, y_new

# Define primary ellipse for which we find a cone -------
two_theta = 0.4
params = (0, 0, 2, 1.5, 0.5)

# -------------------------------------------------------
cx,cy,a,b,phi = params
ex,ey = el.get_ellipse_pts(params)

z0 = b/tan(two_theta)

delta = get_delta(a,z0,two_theta)


s = z0*sin(two_theta)/2*(1/cos(two_theta+delta)-1/(cos(two_theta-delta)))
V = np.array([0,0,0]).astype(float)
V[0] = -(cx+s+z0*sin(delta))
V[1] = cy
V[2] = z0*cos(delta)


# Rotate cone apex by phi around ellipse center
V[0],V[1] = rotate_point(V[0],V[1],cx,cy,phi)
# Calculate direction of cone axis
n = np.array([-sin(delta),0,cos(delta)])
# Rotate this vector around origin (it is a vector)
n[0],n[1] = rotate_point(n[0],n[1],0,0,phi)
print("|n| = ",LA.norm(n))

# Find coordinates of point where cone axis intersects plane z=0
S = np.array([V[0]-V[2]*n[0]/n[2],V[1]-V[2]*n[1]/n[2],0])

# Calculate coordinates of rotated major axis
Am = [cx-a,cy,0]
Am[0],Am[1] = rotate_point(Am[0],Am[1],cx,cy,phi)
Ap = [cx+a,cy,0]
Ap[0],Ap[1] = rotate_point(Ap[0],Ap[1],cx,cy,phi)

# Calculate coordinates of rotated minor axis
Bm = [cx,cy-b,0]
Bm[0],Bm[1] = rotate_point(Bm[0],Bm[1],cx,cy,phi)
Bp = [cx,cy+b,0]
Bp[0],Bp[1] = rotate_point(Bp[0],Bp[1],cx,cy,phi)

# Calculate cone circle (https://math.stackexchange.com/a/73242)
# `a` is vector perpendicular to `n` pointing towards `Ap` (from a.n=0)
a = np.array([Ap[0],Ap[1],(-Ap[0]*n[0]-Ap[0]*n[1])/n[2]])
a = a/LA.norm(a)
# `b` is perpendicular to `a` and `n`
b = np.cross(a,n)

t = np.linspace(0,2*np.pi,50)   # Parameter of the circle
c = 0                           # Multiplier (move circle along cone axis)
r = (z0+c)*tan(two_theta)       # Radius of the circle
cone_x = S[0]-n[0]*c + r*cos(t)*a[0] + r*sin(t)*b[0]
cone_y = S[1]-n[1]*c + r*cos(t)*a[1] + r*sin(t)*b[1]
cone_z = S[2]-n[2]*c + r*cos(t)*a[2] + r*sin(t)*b[2]


# ------------------------------------------------------------------------------
# Plot all the data
lw1 = 2
lw2 = 1
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.set_proj_type('ortho',None) # persp,0.1


# Plot cone --------------------------------------------------------------------
ax.plot(V[0],V[1],V[2],'.',color='k',markersize=10)     # V  - cone apex
ax.plot(S[0],S[1],S[2],'.',color='k',markersize=10)     # S  - cone axis
ax.plot([V[0],S[0]],[V[1],S[1]],[V[2],S[2]],c='k',ls='--')


# Plot primary ellipse ---------------------------------------------------------
ax.plot(ex,ey,c=plt_clrs[0],lw=lw1)
ax.plot(cx,cy,0,'.',color=plt_clrs[0],markersize=10)    # S0 - ellipse center
t = np.array([-5,5])

ax.plot([Am[0],Ap[0]],[Am[1],Ap[1]],[Am[2],Ap[2]],c=plt_clrs[0],ls='--',lw=lw2)
ax.plot([Bm[0],Bp[0]],[Bm[1],Bp[1]],[Bm[2],Bp[2]],c=plt_clrs[0],ls='--',lw=lw2)

# ax.plot(cone_x,cone_y,cone_z,c=plt_clrs[2],lw=lw2)

ax.plot([V[0],Ap[0]],[V[1],Ap[1]],[V[2],Ap[2]],c=plt_clrs[2],lw=lw2)
ax.plot([V[0],Am[0]],[V[1],Am[1]],[V[2],Am[2]],c=plt_clrs[2],lw=lw2)
ax.plot([V[0],Bp[0]],[V[1],Bp[1]],[V[2],Bp[2]],c=plt_clrs[2],lw=lw2)
ax.plot([V[0],Bm[0]],[V[1],Bm[1]],[V[2],Bm[2]],c=plt_clrs[2],lw=lw2)


ax.set_xlim([np.min(ex),np.max(ex)])
ax.set_ylim([np.min(ey),np.max(ey)])
ax.set_zlim([0,V[2]])

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_aspect('equal')
# ax.view_init(90, -90) # 3D view
# ax.view_init(0, -90) # Side view
# ax.view_init(0, 0) # Front view


plt.show()

# %%
