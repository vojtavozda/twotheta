# %%

import os
import sys
import numpy as np
from numpy import pi, cos, sin, sqrt, tan
from numpy import linalg as LA

import matplotlib
# WebAgg
# matplotlib.use('module://matplotlib_inline.backend_inline',force=False)
from matplotlib import pyplot as plt
import elliptools as ellt
from genlib import plt_clrs
import genlib as gl
from scipy import optimize

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

Tips
----

"""

two_theta = pi/8
delta = pi/8

# Define cone
z_max = 1
Z0 = np.array([0,0,0.7*z_max])
a = np.linspace(0,2*np.pi,20)
r = np.linspace(0,1,10)
T, R = np.meshgrid(a, r)
cone_X = R * cos(T) * tan(two_theta) * z_max
cone_Y = R * sin(T) * tan(two_theta) * z_max
cone_Z = R * z_max

# Define plane
n = np.array([sin(delta),0,cos(delta)])

plane_X, plane_Y = np.meshgrid(
    [np.min(cone_X)*1.5,np.max(cone_X)*1.5],
    [np.min(cone_Y)*1.5,np.max(cone_Y)*1.5])
plane_Z = (Z0@n - n[0]*plane_X - n[1]*plane_Y) / n[2]


fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')

ax.plot_surface(cone_X, cone_Y, cone_Z,alpha=0.5,antialiased=True,color=plt_clrs[0])
ax.plot_wireframe(cone_X, cone_Y, cone_Z,color=plt_clrs[0],linewidth=0.5)
ax.plot_surface(plane_X, plane_Y, plane_Z, alpha=0.3,antialiased=True,color='k')
ax.plot_wireframe(plane_X, plane_Y, plane_Z,color='k',linewidth=0.5)

# Plot z-axis
ax.plot([0,0],[0,0],[Z0[2],z_max],c='k',ls=':')
ax.plot([0,0],[0,0],[0,Z0[2]],c='k',ls='-')
ax.plot(Z0[0],Z0[1],Z0[2],'.',color='k',markersize=10)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_aspect('equal')

plt.show()

# %%

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

# ------------------------------------------------------------------------------
# Calculated parameters using pyFAI are:
# distance: 0.0010321342264403 [m]
# PONI1:    0.0002829606114021 [m]
# PONI2:    0.0001775045680058 [m]
# Rot1:     0.0173670646845016 [rad]
# Rot2:     0.6732451185851951 [rad]
# Rot3:    -0.0004043452479840 [rad]


# %%

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
two_theta = 35/180*pi
two_theta = two_theta_012
params = (1,2,4,3,1*pi/4)
# params = (1,2,3,4,3*pi/4)

params = (283, 658, 486, 292, 0.27) # SVD fit of ellipse 0 ()
# params = (281, 761, 571, 400, 0.34) # SVD fit of ellipse 1 ()
# params = (498, 588, 781, 407, 0.41) # SVD fit of ellipse 2 ()
# params = (490, 759, 937, 580, 0.47) # SVD fit of ellipse 3 ()
# params = (518, 336, 651, 241, 0.32) # SVD fit of ellipse 4 ()
# params = (-14, 2010, 2000, 1712, pi/2+0.3) # SOS fit of ellipse 4 ()
# params = (516, 392, 732, 311, 0.32) # SVD fit of ellipse 5 ()
# params = (914,  38,  89,   5, 0.39) # SVD fit of ellipse 6 ()
# params = (955,  26,  55,   2, 0.40) # SVD fit of ellipse 7 ()
# -------------------------------------------------------
cx,cy,a,b,phi = params
ex,ey = ellt.get_ellipse_pts(params)

z0 = b/tan(two_theta)

delta = get_delta(a,z0,two_theta)

# res = optimize.minimize(delta_diff,1.5,args=(a,z0,two_theta))
# print("Found delta:",res.x*180/pi)
# delta = res.x[0]-90
# delta_arr = np.linspace(0,2*pi,100)
# a_arr = z0*sin(two_theta)/2*(1/cos(two_theta+delta_arr)+1/cos(two_theta-delta_arr))
# plt.plot(delta_arr*180/pi,a_arr)
# plt.plot([0,360],[a,a],ls='--',c='k')
# plt.ylim(-20,20)

s = z0*sin(two_theta)/2*(1/cos(two_theta+delta)-1/(cos(two_theta-delta)))
V = np.array([0,0,0]).astype(float)
V[0] = cx+s+z0*sin(delta)
V[1] = cy
V[2] = z0*cos(delta)

# Rotate cone apex by phi around ellipse center
V[0],V[1] = rotate_point(V[0],V[1],cx,cy,phi)
# Calculate direction of cone axis
n = np.array([sin(delta),0,cos(delta)])
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
# Find new ellipse for secondary cone with different two_theta
# All these equations are taken from above but solve for new ellipse params
two_theta2 = 40/180*pi
two_theta2 = two_theta_104
b2 = z0*tan(two_theta2)
s2 = z0*sin(two_theta2)/2*(1/cos(two_theta2+delta)-1/(cos(two_theta2-delta)))
V2 = V.copy()
V2[0],V2[1] = rotate_point(V2[0],V2[1],cx,cy,-phi)
cx2 = V2[0]-s2-z0*sin(delta)
cy2 = V2[1]
cx2,cy2 = rotate_point(cx2,cy2,cx,cy,phi)
a2 = z0*sin(two_theta2)/2*(1/cos(two_theta2+delta)+1/cos(two_theta2-delta))
ex2,ey2 = ellt.get_ellipse_pts((cx2,cy2,a2,b2,phi))

# Calculate coordinates of rotated major axis
Am2 = [cx2-a2,cy2,0]
Am2[0],Am2[1] = rotate_point(Am2[0],Am2[1],cx2,cy2,phi)
Ap2 = [cx2+a2,cy2,0]
Ap2[0],Ap2[1] = rotate_point(Ap2[0],Ap2[1],cx2,cy2,phi)

# Calculate coordinates of rotated minor axis
Bm2 = [cx2,cy2-b2,0]
Bm2[0],Bm2[1] = rotate_point(Bm2[0],Bm2[1],cx2,cy2,phi)
Bp2 = [cx2,cy2+b2,0]
Bp2[0],Bp2[1] = rotate_point(Bp2[0],Bp2[1],cx2,cy2,phi)

# Calculate secondary cone circle
a2 = np.array([Ap2[0],Ap2[1],(-Ap2[0]*n[0]-Ap2[0]*n[1])/n[2]])
a2 = a2/LA.norm(a2)
b2 = np.cross(a2,n)

t = np.linspace(0,2*np.pi,50)   # Parameter of the circle
r = (z0+c)*tan(two_theta2)       # Radius of the circle
cone2_x = S[0]-n[0]*c + r*cos(t)*a2[0] + r*sin(t)*b2[0]
cone2_y = S[1]-n[1]*c + r*cos(t)*a2[1] + r*sin(t)*b2[1]
cone2_z = S[2]-n[2]*c + r*cos(t)*a2[2] + r*sin(t)*b2[2]

# ------------------------------------------------------------------------------
# Plot all the data
lw1 = 2
lw2 = 1
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.set_proj_type('ortho',None) # persp,0.1

# data = np.load("data.npy")
# data[data<0] = 0
# data[data>30] = 30
# xx,yy = np.meshgrid(np.arange(1024),np.arange(512))
# ax.contourf(xx, yy, data, 100, zdir='z', offset=-0.1, cmap="plasma")

# Plot fitted ellipses from Jungfrau detector ----------------------------------
for i in range(8):
    x = np.load(os.path.join('data',f'x_{i}.npy'))
    y = np.load(os.path.join('data',f'y_{i}.npy'))
    cart = ellt.fit_ellipse(x,y)
    params = ellt.cart_to_pol(cart)
    xel,yel = ellt.get_ellipse_pts(params)
    ax.plot(xel,yel,np.zeros(len(xel)),ls='--',c='k',lw=0.5)

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

ax.plot(cone_x,cone_y,cone_z,c=plt_clrs[2],lw=lw2)

ax.plot([V[0],Ap[0]],[V[1],Ap[1]],[V[2],Ap[2]],c=plt_clrs[2],lw=lw2)
ax.plot([V[0],Am[0]],[V[1],Am[1]],[V[2],Am[2]],c=plt_clrs[2],lw=lw2)
ax.plot([V[0],Bp[0]],[V[1],Bp[1]],[V[2],Bp[2]],c=plt_clrs[2],lw=lw2)
ax.plot([V[0],Bm[0]],[V[1],Bm[1]],[V[2],Bm[2]],c=plt_clrs[2],lw=lw2)

# Plot secondary ellipse -------------------------------------------------------
ax.plot(ex2,ey2,c=plt_clrs[3],lw=lw1)
ax.plot(cx2,cy2,0,'.',color=plt_clrs[3],markersize=10)    # S0 - ellipse center
ax.plot(cone2_x,cone2_y,cone2_z,c=plt_clrs[1],lw=lw2)
ax.plot([Am2[0],Ap2[0]],[Am2[1],Ap2[1]],[Am2[2],Ap2[2]],c=plt_clrs[3],ls='--',lw=lw2)
ax.plot([Bm2[0],Bp2[0]],[Bm2[1],Bp2[1]],[Bm2[2],Bp2[2]],c=plt_clrs[3],ls='--',lw=lw2)
ax.plot([V[0],Ap2[0]],[V[1],Ap2[1]],[V[2],Ap2[2]],c=plt_clrs[1],lw=lw2)
ax.plot([V[0],Am2[0]],[V[1],Am2[1]],[V[2],Am2[2]],c=plt_clrs[1],lw=lw2)
ax.plot([V[0],Bp2[0]],[V[1],Bp2[1]],[V[2],Bp2[2]],c=plt_clrs[1],lw=lw2)
ax.plot([V[0],Bm2[0]],[V[1],Bm2[1]],[V[2],Bm2[2]],c=plt_clrs[1],lw=lw2)

ax.set_xlim([np.min(ex),np.max(ex)])
ax.set_ylim([np.min(ey),np.max(ey)])
ax.set_zlim([0,V[2]])

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_aspect('equal')
ax.view_init(90, -90)


plt.show()

# %%
