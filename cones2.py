# %%

import numpy as np
from numpy import sin, cos, pi, tan
from numpy.linalg import norm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import ellipse as el
from genlib import plt_clrs

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
# Load points of found ellipses from detector
x_012 = np.load(f'x_0.npy')
y_012 = np.load(f'y_0.npy')
el_012 = np.array((x_012,y_012,np.zeros_like(x_012)))

x_104 = np.load(f'x_2.npy')
y_104 = np.load(f'y_2.npy')
el_104 = np.array((x_104,y_104,np.zeros_like(x_104)))

x_110 = np.load(f'x_4.npy')
y_110 = np.load(f'y_4.npy')
el_110 = np.array((x_110,y_110,np.zeros_like(x_110)))

x_113 = np.load(f'x_6.npy')
y_113 = np.load(f'y_6.npy')
el_113 = np.array((x_113,y_113,np.zeros_like(x_113)))

def rotate3D(data:np.ndarray,phi:float,theta:float,psi:float):
    """
    Rotate vector in 3D around origin. Angles are Euler angles:
    phi:    rotate around the `z` axis
    theta:  rotate around the `x'` axis
    psi:    rotate around the `z''` axis

    For more details check:
    https://mathworld.wolfram.com/EulerAngles.html
    """

    R_phi = np.array((
        ( cos(phi), sin(phi), 0),
        (-sin(phi), cos(phi), 0),
        (        0,        0, 1)
    ))
    R_theta = np.array((
        (1,           0,          0),
        (0,  cos(theta), sin(theta)),
        (0, -sin(theta), cos(theta))
    ))
    R_psi = np.array((
        ( cos(psi), sin(psi), 0),
        (-sin(psi), cos(psi), 0),
        (        0,        0, 1)
    ))

    R = R_psi@(R_theta@R_phi)

    return R@data

def move3D(data:np.ndarray,vector:np.ndarray):
    
    if len(data.shape)==1:
        return data + vector

    for i in range(data.shape[1]):
        data[:,i] += vector
    return data

def cone(angle,length):
    """ Get coordinates of a cone with apex at (0,0,0) and axis = z """
    a = np.linspace(0,2*np.pi,20)
    r = np.linspace(0,1,10)
    T, R = np.meshgrid(a, r)
    X = R * cos(T) * tan(angle) * length
    Y = R * sin(T) * tan(angle) * length
    Z = R * length
    return X,Y,Z

def sphere(x0,y0,z0,radius):
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    X = x0+cos(u)*sin(v)*radius
    Y = y0+sin(u)*sin(v)*radius
    Z = z0+cos(v)*radius
    return X,Y,Z

def get_ellipse_parameters(z0,n,two_theta):

    # First focus from closer Dandeline sphere
    z_D1 = n[2]*z0/(n[2]+sin(two_theta))
    r_D1 = z_D1*sin(two_theta)
    f1 = np.array([0,0,z_D1])+r_D1*n

    # Second focus from further Dandeline sphere
    z_D2 = -n[2]*z0/(-n[2]+sin(two_theta))
    r_D2= z_D2*sin(two_theta)
    f2 = np.array([0,0,z_D2])-r_D2*n

    # Calculate parameters of an ellipse at plane and cone intersection
    b = z0*tan(two_theta)       # semi-minor axis
    c = (f2+f1)/2                   # ellipse center
    f = (norm(f2-f1))/2             # distance of foci from the center
    a = np.sqrt(f**2+b**2)          # semi-major axis
    phi = np.angle(n[0]+n[1]*1j)    # angle in xy projection

    return c[0],c[1],a,b,phi

z0 = 1074
phi   = -19.68    /180*pi
theta = -19.6 /180*pi
psi   = 14.93   /180*pi
shift_x = -167.8
shift_y = -1015.8

base_x = rotate3D((1,0,0),phi,theta,psi)
base_y = rotate3D((0,1,0),phi,theta,psi)
base_z = rotate3D((0,0,1),phi,theta,psi)    # Also plane's normal vector
n = base_z

detector = np.array((
    (0,0,0),
    (1024,0,0),
    (1024,512,0),
    (0,512,0)
)).T

# Rotate dectector and move to desired position within rotated plane
el_012   = move3D(rotate3D(el_012,phi,theta,psi),shift_x*base_x+shift_y*base_y+np.array((0,0,z0)))
el_104   = move3D(rotate3D(el_104,phi,theta,psi),shift_x*base_x+shift_y*base_y+np.array((0,0,z0)))
el_110   = move3D(rotate3D(el_110,phi,theta,psi),shift_x*base_x+shift_y*base_y+np.array((0,0,z0)))
el_113   = move3D(rotate3D(el_113,phi,theta,psi),shift_x*base_x+shift_y*base_y+np.array((0,0,z0)))
detector = move3D(rotate3D(detector,phi,theta,psi),shift_x*base_x+shift_y*base_y+np.array((0,0,z0)))

params_012 = get_ellipse_parameters(z0,n,two_theta_012)
el012_x, el012_y = el.get_ellipse_pts(params_012)
el012_z = (n[2]*z0-n[0]*el012_x-n[1]*el012_y)/n[2]
print("Sum of squares 012 =",el.get_sum_of_squares(el_012[0,:],el_012[1,:],params_012))

params_104 = get_ellipse_parameters(z0,n,two_theta_104)
el104_x, el104_y = el.get_ellipse_pts(params_104)
el104_z = (n[2]*z0-n[0]*el104_x-n[1]*el104_y)/n[2]
print("Sum of squares 104 =",el.get_sum_of_squares(el_104[0,:],el_104[1,:],params_104))

params_110 = get_ellipse_parameters(z0,n,two_theta_110)
el110_x, el110_y = el.get_ellipse_pts(params_110)
el110_z = (n[2]*z0-n[0]*el110_x-n[1]*el110_y)/n[2]
print("Sum of squares 110 =",el.get_sum_of_squares(el_110[0,:],el_110[1,:],params_110))

params_113 = get_ellipse_parameters(z0,n,two_theta_113)
el113_x, el113_y = el.get_ellipse_pts(params_113)
el113_z = (n[2]*z0-n[0]*el113_x-n[1]*el113_y)/n[2]
print("Sum of squares 113 =",el.get_sum_of_squares(el_113[0,:],el_113[1,:],params_113))

plane_x = np.array([-600,600,600,-600,-600])
plane_y = np.array([-300,-300,300,300,-300])
plane_z = (n[2]*z0-n[0]*plane_x-n[1]*plane_y)/n[2]

# ------------------------------------------------------------------------------
# Plot all the data

# Define a cone
cone_X, cone_Y, cone_Z = cone(two_theta_113,z0+100)
# sphere1_X, sphere1_Y, sphere1_Z = sphere(0,0,z_D1,r_D1)
# sphere2_X, sphere2_Y, sphere2_Z = sphere(0,0,z_D2,r_D2)

def plot_data(ax:plt.Axes):

    # Plot detector with data
    ax.plot(el_012[0,:],el_012[1,:],el_012[2,:],'-')
    ax.plot(el_104[0,:],el_104[1,:],el_104[2,:],'-')
    ax.plot(el_110[0,:],el_110[1,:],el_110[2,:],'-')
    ax.plot(el_113[0,:],el_113[1,:],el_113[2,:],'-')
    ax.add_collection3d(Poly3DCollection([detector.T],
        alpha=0.2,facecolor='k',linewidths=1,edgecolor='k'))

    ax.add_collection3d(Poly3DCollection([np.array((plane_x,plane_y,plane_z)).T],
        alpha=0.1,facecolor='k',linewidths=1,edgecolor='k'))

    c = 100
    ax.plot([0,base_z[0]*c],[0,base_z[1]*c],[z0,z0+base_z[2]*c],ls='--',c='r')

    # Plot cone
    ax.plot(0,0,z0,'.',color='k',markersize=5)              # point z0
    ax.plot(0,0,0,'.',color='k',markersize=10)              # cone apex
    ax.plot([0,0],[0,0],[0,np.max(cone_Z)],ls=':',c='k')       # cone axis
    ax.plot_surface(cone_X, cone_Y, cone_Z,alpha=0.2,antialiased=True,color=plt_clrs[0])
    ax.plot_wireframe(cone_X, cone_Y, cone_Z,color=plt_clrs[0],linewidth=0.1)

    # Plot Dandelin spheres
    # ax.plot_wireframe(sphere1_X,sphere1_Y,sphere1_Z,linewidth=0.2)
    # ax.plot(f1[0],f1[1],f1[2],'.',markersize=5,c='k')

    # ax.plot_wireframe(sphere2_X,sphere2_Y,sphere2_Z,linewidth=0.2)
    # ax.plot(f2[0],f2[1],f2[2],'.',markersize=5,c='k')

    # Plot ellipse
    ax.plot(el012_x,el012_y,el012_z,c='k')
    ax.plot(el104_x,el104_y,el104_z,c='k')
    ax.plot(el110_x,el110_y,el110_z,c='k')
    ax.plot(el113_x,el113_y,el113_z,c='k')

    ax.set_xlim(np.min((np.min(detector[0,:]),np.min(el012_x),np.min(cone_X))),
                np.max((np.max(detector[0,:]),np.max(el012_x),np.max(cone_X))))
    ax.set_ylim(np.min((np.min(detector[1,:]),np.min(el012_y),np.min(cone_Y))),
                np.max((np.max(detector[1,:]),np.max(el012_y),np.max(cone_Y))))
    ax.set_zlim(0,np.max((np.max(cone_Z),np.max(el113_z))))
    ax.set_aspect('equal')

# ----- Plot single 3D figure -----
fig = plt.figure()
axs = [plt.Axes]
axs[0] = fig.add_subplot(111,projection='3d')
axs[0].view_init(90,-90,0)
axs[0].set_proj_type('ortho',None)

# ----- Plot 3D with projections in subplots -----
# fig = plt.figure(figsize=(8,6))

# gs = fig.add_gridspec(5,3)
# axs = [plt.Axes]*4
# axs[0] = plt.subplot(gs[:3,:], projection='3d')
# axs[1] = plt.subplot(gs[3:,0], projection='3d')
# axs[2] = plt.subplot(gs[3:,1], projection='3d')
# axs[3] = plt.subplot(gs[3:,2], projection='3d')

# plot_data(axs[1])
# plot_data(axs[2])
# plot_data(axs[3])

# axs[1].view_init(90, -90)
# axs[2].view_init( 0, -90)
# axs[3].view_init( 0,   0)


# for i in range(1,4):
#     axs[i].set_proj_type('ortho',None)
#     axs[i].set_xticklabels([])
#     axs[i].set_yticklabels([])
#     axs[i].set_zticklabels([])

plot_data(axs[0])
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')
axs[0].set_zlabel('z')

plt.show()
