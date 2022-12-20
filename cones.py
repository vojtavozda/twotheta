# %%

import numpy as np
from numpy import pi, cos, sin, sqrt, tan

import matplotlib
# WebAgg
# matplotlib.use('module://matplotlib_inline.backend_inline',force=False)
from matplotlib import pyplot as plt
import ellipse as el
from genlib import plt_clrs

# two_theta = pi/8
# delta = pi/8

# # Define cone
# z_max = 1
# Z0 = np.array([0,0,0.7*z_max])
# a = np.linspace(0,2*np.pi,20)
# r = np.linspace(0,1,10)
# T, R = np.meshgrid(a, r)
# cone_X = R * cos(T) * tan(two_theta) * z_max
# cone_Y = R * sin(T) * tan(two_theta) * z_max
# cone_Z = R * z_max

# # Define plane
# n = np.array([sin(delta),0,cos(delta)])

# plane_X, plane_Y = np.meshgrid(
#     [np.min(cone_X)*1.5,np.max(cone_X)*1.5],
#     [np.min(cone_Y)*1.5,np.max(cone_Y)*1.5])
# plane_Z = (Z0@n - n[0]*plane_X - n[1]*plane_Y) / n[2]


# fig = plt.figure()
# ax = fig.add_subplot(111,projection='3d')

# ax.plot_surface(cone_X, cone_Y, cone_Z,alpha=0.5,antialiased=True,color=plt_clrs[0])
# ax.plot_wireframe(cone_X, cone_Y, cone_Z,color=plt_clrs[0],linewidth=0.5)
# ax.plot_surface(plane_X, plane_Y, plane_Z, alpha=0.3,antialiased=True,color='k')
# ax.plot_wireframe(plane_X, plane_Y, plane_Z,color='k',linewidth=0.5)

# # Plot z-axis
# ax.plot([0,0],[0,0],[Z0[2],z_max],c='k',ls=':')
# ax.plot([0,0],[0,0],[0,Z0[2]],c='k',ls='-')
# ax.plot(Z0[0],Z0[1],Z0[2],'.',color='k',markersize=10)

# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# ax.set_aspect('equal')

plt.show()
# %%

# TODO: Rotate ellipse (rotate all points)

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

    # print(f"delta+ = {deltaP*180/pi:.0f}, delta- = {deltaM*180/pi:.0f}")
    return deltaP

def rotate_point(x,y,x0,y0,phi):

    x -= x0
    y -= y0

    x_new = x*cos(phi) - y*sin(phi)
    y_new = x*sin(phi) + y*cos(phi)

    x_new += x0
    y_new += y0

    return x_new, y_new

two_theta = pi/8

params = (1,2,4,3,0*pi/4)
cx,cy,a,b,phi = params
ex,ey = el.get_ellipse_pts(params)

z0 = b/tan(two_theta)

delta = get_delta(a,z0,two_theta)

s = z0*sin(two_theta)/2*(1/cos(two_theta+delta)-1/(cos(two_theta-delta)))
V = np.array([0,0,0]).astype(float)
V[0] = cx+s+z0*sin(delta)
V[1] = cy
V[2] = z0*cos(delta)
n = np.array([sin(delta),0,cos(delta)])

# Rotate V:
# V[0],V[1] = rotate_point(V[0],V[1],cx,cy,phi)
# n[0],n[1] = rotate_point(n[0],n[1],cx,cy,phi)

# S should be calculated from n_x,n_y
S = np.array([V[0]-V[2]*tan(delta),cy,0])

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')

ax.plot(ex,ey)
ax.plot(V[0],V[1],V[2],'.',color='k',markersize=10)
ax.plot(cx,cy,0,'.',color='k',markersize=10)
ax.plot(S[0],S[1],S[2],'.',color='k',markersize=10)
t = np.array([-5,5])

ax.plot([V[0],S[0]],[V[1],S[1]],[V[2],S[2]],c='k',ls='--')
ax.plot([cx-a,cx+a],[cy,cy],[0,0],c=plt_clrs[1],ls='--')
ax.plot([cx,cx],[cy-b,cy+b],[0,0],c=plt_clrs[2],ls='--')
ax.plot([V[0],V[0]+V[2]*tan(two_theta-delta)],[V[1],cy],[V[2],0],c=plt_clrs[1])
ax.plot([V[0],V[0]-V[2]*tan(two_theta+delta)],[V[1],cy],[V[2],0],c=plt_clrs[1])
ax.plot([V[0],cx],[V[1],cy-z0*tan(two_theta)],[V[2],0],c=plt_clrs[2])
ax.plot([V[0],cx],[V[1],cy+z0*tan(two_theta)],[V[2],0],c=plt_clrs[2])


ax.set_xlim([np.min(ex),np.max(ex)])
ax.set_ylim([np.min(ey),np.max(ey)])
ax.set_zlim([0,V[2]])

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_aspect('equal')

plt.show()

print(f"delta = {delta*180/pi:.0f}°")