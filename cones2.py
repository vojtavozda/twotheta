# %%
import os
import sys
import numpy as np
from numpy import sin, cos, pi, tan, sqrt
from numpy.linalg import norm, inv
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import importlib

import ellipse as el
from genlib import plt_clrs

importlib.reload(el)

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
x_012 = np.load(os.path.join('data','x_0.npy'))
y_012 = np.load(os.path.join('data','y_0.npy'))
el_012 = np.array((x_012,y_012,np.zeros_like(x_012)))

x_104 = np.load(os.path.join('data','x_2.npy'))
y_104 = np.load(os.path.join('data','y_2.npy'))
el_104 = np.array((x_104,y_104,np.zeros_like(x_104)))

x_110 = np.load(os.path.join('data','x_4.npy'))
y_110 = np.load(os.path.join('data','y_4.npy'))
el_110 = np.array((x_110,y_110,np.zeros_like(x_110)))

x_113 = np.load(os.path.join('data','x_6.npy'))
y_113 = np.load(os.path.join('data','y_6.npy'))
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

def move(data:np.ndarray,vector:np.ndarray):
    
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


# ==============================================================================

z0 = 1184.6707
phi   = -1.0414    /180*pi
theta = -32.5151     /180*pi
psi   = -0.0541     /180*pi
shift_x = -163.8418
shift_y = -1029.1911

# z0 = 1242.1062
# phi = 0.3633/180*pi
# theta = -35/180*pi
# psi = 0.0128/180*pi
# shift_x = -161.8735
# shift_y = -1061.7845

# z0,phi,theta,psi,shift_x,shift_y = [1199.9753,0.2374,-0.5179,-0.0000,-149.9912,-1000.0338]

base_x = rotate3D((1,0,0),phi,theta,psi)
base_y = rotate3D((0,1,0),phi,theta,psi)
base_z = rotate3D((0,0,1),phi,theta,psi)    # Also plane's normal vector
n = base_z

detector = np.array((
    (0,0,0,),
    (1024,0,0),
    (1024,512,0),
    (0,512,0),
    (0,0,0)
)).T

# Rotate dectector and move to desired position within rotated plane
el_012   = move(rotate3D(el_012,phi,theta,psi),shift_x*base_x+shift_y*base_y+np.array((0,0,z0)))
el_104   = move(rotate3D(el_104,phi,theta,psi),shift_x*base_x+shift_y*base_y+np.array((0,0,z0)))
el_110   = move(rotate3D(el_110,phi,theta,psi),shift_x*base_x+shift_y*base_y+np.array((0,0,z0)))
el_113   = move(rotate3D(el_113,phi,theta,psi),shift_x*base_x+shift_y*base_y+np.array((0,0,z0)))
detector = move(rotate3D(detector,phi,theta,psi),shift_x*base_x+shift_y*base_y+np.array((0,0,z0)))

params_012 = el.get_ellipse_from_cone(z0,n,two_theta_012)
el012_x, el012_y = el.get_ellipse_pts(params_012)
el012_z = (n[2]*z0-n[0]*el012_x-n[1]*el012_y)/n[2]
print("Sum of squares 012 =",el.get_sum_of_squares(el_012[0,:],el_012[1,:],params_012))

params_104 = el.get_ellipse_from_cone(z0,n,two_theta_104)
el104_x, el104_y = el.get_ellipse_pts(params_104)
el104_z = (n[2]*z0-n[0]*el104_x-n[1]*el104_y)/n[2]
print("Sum of squares 104 =",el.get_sum_of_squares(el_104[0,:],el_104[1,:],params_104))

params_110 = el.get_ellipse_from_cone(z0,n,two_theta_110)
el110_x, el110_y = el.get_ellipse_pts(params_110)
el110_z = (n[2]*z0-n[0]*el110_x-n[1]*el110_y)/n[2]
print("Sum of squares 110 =",el.get_sum_of_squares(el_110[0,:],el_110[1,:],params_110))

params_113 = el.get_ellipse_from_cone(z0,n,two_theta_113)
el113_x, el113_y = el.get_ellipse_pts(params_113)
el113_z = (n[2]*z0-n[0]*el113_x-n[1]*el113_y)/n[2]
print("Sum of squares 113 =",el.get_sum_of_squares(el_113[0,:],el_113[1,:],params_113))

plane_x = np.array([-600,600,600,-600,-600])
plane_y = np.array([-300,-300,300,300,-300])
plane_z = (n[2]*z0-n[0]*plane_x-n[1]*plane_y)/n[2]


# ------------------------------------------------------------------------------
# Plot all the data

# Define a cone
cone_X, cone_Y, cone_Z = cone(two_theta_113,z0+500)
sphere1_X, sphere1_Y, sphere1_Z = sphere(0,0,params_113['z_D1'],params_113['r_D1'])
sphere2_X, sphere2_Y, sphere2_Z = sphere(0,0,params_113['z_D2'],params_113['r_D2'])

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
    ax.plot_wireframe(sphere1_X,sphere1_Y,sphere1_Z,linewidth=0.2)
    ax.plot(params_113['f1'][0],params_113['f1'][1],params_113['f1'][2],'.',markersize=5,c='k')

    # ax.plot_wireframe(sphere2_X,sphere2_Y,sphere2_Z,linewidth=0.2)
    ax.plot(params_113['f2'][0],params_113['f2'][1],params_113['f2'][2],'.',markersize=5,c='k')

    # Plot ellipse
    ax.plot(el012_x,el012_y,el012_z,c='k')
    ax.plot(el104_x,el104_y,el104_z,c='k')
    ax.plot(el110_x,el110_y,el110_z,c='k')
    ax.plot(el113_x,el113_y,el113_z,c='k')
    # ax.plot(params_113['P1'][0],params_113['P1'][1],params_113['P1'][2],'.',color='k',markersize=10)              # cone apex

    # Plot projection to xy plane
    z = -z0/2

    ax.plot(0,0,z,'.',color='k',markersize=10)
    ax.plot(detector[0,:],detector[1,:],np.zeros_like(detector[2,:])+z,c='k')
    ax.plot(el_012[0,:],el_012[1,:],np.zeros_like(el_012[2,:])+z,'-')
    ax.plot(el_104[0,:],el_104[1,:],np.zeros_like(el_104[2,:])+z,'-')
    ax.plot(el_110[0,:],el_110[1,:],np.zeros_like(el_110[2,:])+z,'-')
    ax.plot(el_113[0,:],el_113[1,:],np.zeros_like(el_113[2,:])+z,'-')
    ax.plot(el012_x,el012_y,np.zeros_like(el012_x)+z,c='k')
    ax.plot(el104_x,el104_y,np.zeros_like(el104_x)+z,c='k')
    ax.plot(el110_x,el110_y,np.zeros_like(el110_x)+z,c='k')
    ax.plot(el113_x,el113_y,np.zeros_like(el113_x)+z,c='k')

    
    ax.set_xlim(np.min(cone_X),np.max(cone_X))
    ax.set_ylim(np.min(cone_Y),np.max(cone_Y))
    ax.set_zlim(0,np.max(cone_Z))
    
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

plt.tight_layout()
plt.show()


importlib.reload(el)

proj_012_x = (base_y[0]*el012_y-base_y[1]*el012_x)/(base_y[0]*base_x[1]-base_y[1]*base_x[0])
proj_012_y = (el012_x-proj_012_x*base_x[0])/base_y[0]

x_012 = np.load(os.path.join('data','x_0.npy'))
y_012 = np.load(os.path.join('data','y_0.npy'))

x_104 = np.load(os.path.join('data','x_2.npy'))
y_104 = np.load(os.path.join('data','y_2.npy'))

x_110 = np.load(os.path.join('data','x_4.npy'))
y_110 = np.load(os.path.join('data','y_4.npy'))

x_113 = np.load(os.path.join('data','x_6.npy'))
y_113 = np.load(os.path.join('data','y_6.npy'))

el_012 = np.array((x_012,y_012))
el_104 = np.array((x_104,y_104))
el_110 = np.array((x_110,y_110))
el_113 = np.array((x_113,y_113))

data = np.load(os.path.join('data','data.npy'))
data[data<0] = 0
data[data>30] = 30
data[:,0:5] = 0
data[:,255:257] = 0
data[:,511:513] = 0
data[:,767:769] = 0
data[:,1019:] = 0
data[0:4,:] = 0
data[255:257,:] = 0
data[504:,:] = 0

fig = plt.figure()
axs:plt.Axes = plt.gca()
plt.imshow(data,origin='lower',vmin=0,vmax=5)
plt.plot(el_012[0,:],el_012[1,:])
plt.plot(el_104[0,:],el_104[1,:])
plt.plot(el_110[0,:],el_110[1,:])
plt.plot(el_113[0,:],el_113[1,:])

two_thetas = np.linspace(30,57,1000)
intensities = np.zeros_like(two_thetas)

for i,two_theta in enumerate(two_thetas):

    """
    Goal: Find parameters of an ellipse projected from xy plane to tilted
    detector plane.
    """
    # Get parameters of an ellipse in xy plane
    params = el.get_ellipse_from_cone(z0,n,two_theta/180*pi)

    a = params['a']
    b = params['b']

    # Center of the ellipse
    point_c = np.array([params['x0'],params['y0']])
    # Point where semi-major axis touches the ellipse
    point_a = np.array([point_c[0]+a*cos(params['phi']),
                        point_c[1]+a*sin(params['phi'])])
    # Point where semi-minor axis touches the ellipse
    point_b = np.array([point_c[0]-b*sin(params['phi']),
                        point_c[1]+b*cos(params['phi'])])
    # Calculate vectors from center to point a and point b
    vec_a = point_a-point_c
    vec_b = point_b-point_c

    # Transform matrix from xy plane into tilted detector plane
    M = inv(np.array([base_x[0:2],base_y[0:2]]).T)
    # Shift vector to origin
    shift_vec = np.array([-shift_x,-shift_y])

    # Now project the center of the ellipse to the plane and move to the origin
    proj_point_c = move(M@point_c,shift_vec)
    # Do the same with vectors
    proj_vec_a = M@vec_a
    proj_vec_b = M@vec_b
    # Calculate semi-major and semi-minor axes of the projected ellipse
    proj_a = norm(proj_vec_a)
    proj_b = norm(proj_vec_b)
    # Angle of projected ellipse is not conserved
    proj_phi = np.angle(proj_vec_a[0]+proj_vec_a[1]*1j)

    # Join parameters of the projected ellipse
    proj_params = (proj_point_c[0],proj_point_c[1],proj_a,proj_b,proj_phi)
    el_proj = np.array(el.get_ellipse_pts(proj_params,npts=500))
    if i%np.ceil(len(two_thetas)/30)==0:
        plt.plot(el_proj[0,:],el_proj[1,:],ls='--',c='w',lw=0.5)

    # Same points can be obtained by simple projecting original ellipse:
    # ...el_orig = np.array(el.get_ellipse_pts(params,npts=200))
    # ...el_proj = move(M@el_orig,shift_vec)
    # But in this case we have just points and not parameters
    
    if np.isnan(proj_params).any() or np.isinf(proj_params).any():
        print("Warining: NaN or Inf in proj_params")
        intensities[i] = 0
    else:
        intensities[i] = el.mask_sum(data,proj_params)

plt.xlim(0,1024)
plt.ylim(0,512)
axs.set_aspect('equal')
plt.show()

fig = plt.figure()
plt.plot(two_thetas,intensities)
plt.show()
# %%

plt.plot(np.sum(data,axis=0))
plt.show()
plt.plot(np.sum(data,axis=1))
plt.show()