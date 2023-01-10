
# %%

import numpy as np
from numpy import sin, cos, pi, tan
from numpy.linalg import norm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy import optimize

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

# ------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------
def objective(params):
    
    global el_012, el_104, el_110, el_113

    z0,phi,theta,psi,shift_x,shift_y = params

    # Find base vectors of rotated detector
    base_x = rotate3D((1,0,0),phi,theta,psi)
    base_y = rotate3D((0,1,0),phi,theta,psi)
    n = rotate3D((0,0,1),phi,theta,psi)         # Also plane's normal vector

    # Rotate dectector and move to desired position within rotated plane
    el_012_rot = move3D(rotate3D(el_012,phi,theta,psi),shift_x*base_x+shift_y*base_y+np.array((0,0,z0)))
    el_104_rot = move3D(rotate3D(el_104,phi,theta,psi),shift_x*base_x+shift_y*base_y+np.array((0,0,z0)))
    el_110_rot = move3D(rotate3D(el_110,phi,theta,psi),shift_x*base_x+shift_y*base_y+np.array((0,0,z0)))
    el_113_rot = move3D(rotate3D(el_113,phi,theta,psi),shift_x*base_x+shift_y*base_y+np.array((0,0,z0)))

    # Calculate parameters of ellipses at cones and plane intersection
    params_012 = get_ellipse_parameters(z0,n,two_theta_012)
    params_104 = get_ellipse_parameters(z0,n,two_theta_104)
    params_110 = get_ellipse_parameters(z0,n,two_theta_110)
    params_113 = get_ellipse_parameters(z0,n,two_theta_113)
    
    # Compare found ellipses with experimental data
    sos = 0
    sos += el.get_sum_of_squares(el_012_rot[0,:],el_012_rot[1,:],params_012)
    sos += el.get_sum_of_squares(el_104_rot[0,:],el_104_rot[1,:],params_104)
    sos += el.get_sum_of_squares(el_110_rot[0,:],el_110_rot[1,:],params_110)
    sos += el.get_sum_of_squares(el_113_rot[0,:],el_113_rot[1,:],params_113)
    
    # print(f"z0={z0:.0f}, phi={phi*180/pi:.0f}, theta={theta*180/pi:.0f}, psi={psi*180/pi:.0f}, (x0,y0)=({shift_x:.0f},{shift_y:.0f}) | sos={sos:.0f}")
    
    return sos



# ==============================================================================
# parameters: (z0,phi,theta,psi,shift_x,shift_y)
ansatz = [1200,0,-0.112,0,-150,-1000]
# ansatz = [1200,0,0.0,0,0,0]
bounds = ((100,5000),(-pi,pi),(-pi,pi),(-pi,pi),(-10000,10000),(-10000,10000))
res = optimize.minimize(objective,ansatz,bounds=bounds)
z0,phi,theta,psi,shift_x,shift_y = res.x
print(f"SOS = {res.fun:.2f}")
print("Calculated parameters:")
print(f"z0 = {z0:.2f}")
print(f"phi = {phi*180/pi:.2f}°")
print(f"theta = {theta*180/pi:.2f}°")
print(f"psi = {psi*180/pi:.2f}°")
print(f"x = {shift_x:.2f}")
print(f"y = {shift_y:.2f}")