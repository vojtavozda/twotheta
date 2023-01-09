# %%

import numpy as np
from numpy import sin, cos, pi
from numpy.linalg import norm
from matplotlib import pyplot as plt


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

def rotate3D(point:np.ndarray,phi:float,theta:float,psi:float):
    """
    Rotate vector in 3D around origin. Angles are Euler angles:
    phi:    rotate around the `z` axis
    theta:  rotate around the `x'` axis
    psi:    rotate around the `z''` axis
    """
    R_psi = np.array((
        (1,        0,        0),
        (0, cos(psi),-sin(psi)),
        (0, sin(psi), cos(psi))
    ))
    R_theta = np.array((
        ( cos(theta), 0, sin(theta)),
        (          0, 1,          0),
        (-sin(theta), 0, cos(theta))
    ))
    R_psi = np.array((
        (cos(psi),-sin(psi), 0),
        (sin(psi), cos(psi), 0),
        (       0,        0, 1)
    ))


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


    # R = R_phi@(R_theta@R_psi)
    R = R_psi@(R_theta@R_phi)

    return R@point

z0 = 0
phi = pi/2
theta = 0
psi = 0
shift_x = 0
shift_y = 0

base_x = rotate3D((1,0,0),phi,theta,psi)
base_y = rotate3D((0,1,0),phi,theta,psi)


# ------------------------------------------------------------------------------
# Plot all the data

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.set_proj_type('ortho',None) # persp,0.1

ax.plot(el_012[0,:],el_012[1,:],el_012[2,:],'.')

el_012 = rotate3D(el_012,phi,theta,psi)

ax.plot(el_012[0,:],el_012[1,:],el_012[2,:],'.')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_aspect('equal')
ax.view_init(90, -90)

plt.show()
