
# %%

# Main idea:
# ----------
# Sample is places at the origin (0,0,0) and beam is directed along the z-axis.
# Diffraction creates cones with apexes at the origin.
# We are looking for the position and orientation of the detector so found
# conics (ellipses) on the detector match intersections of cones.

import numpy as np
from numpy import sin, cos, pi, tan
from numpy.linalg import norm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy import optimize
import importlib

import elliptools as ellt
from genlib import plt_clrs

importlib.reload(ellt)

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
x_012 = np.load('data/x_0.npy')
y_012 = np.load('data/y_0.npy')
el_012 = np.array((x_012,y_012,np.zeros_like(x_012)))

x_104 = np.load('data/x_2.npy')
y_104 = np.load('data/y_2.npy')
el_104 = np.array((x_104,y_104,np.zeros_like(x_104)))

x_110 = np.load('data/x_4.npy')
y_110 = np.load('data/y_4.npy')
el_110 = np.array((x_110,y_110,np.zeros_like(x_110)))

x_113 = np.load('data/x_6.npy')
y_113 = np.load('data/y_6.npy')
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


# ------------------------------------------------------------------------------
def objective(params):
    
    global el_012, el_104, el_110, el_113

    z0,phi,theta,psi,shift_x,shift_y = params

    # print("try params:",params)

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
    p012 = ellt.get_ellipse_from_cone(z0,n,two_theta_012)
    p104 = ellt.get_ellipse_from_cone(z0,n,two_theta_104)
    p110 = ellt.get_ellipse_from_cone(z0,n,two_theta_110)
    p113 = ellt.get_ellipse_from_cone(z0,n,two_theta_113)
    
    # print("params_012",p012)
    fitEl012 = ellt.Ellipse(x0=p012['x0'],y0=p012['y0'],a=p012['a'],b=p012['b'],phi=p012['phi'],xData=el_012_rot[0,:],yData=el_012_rot[1,:],theta=two_theta_012)
    fitEl104 = ellt.Ellipse(x0=p104['x0'],y0=p104['y0'],a=p104['a'],b=p104['b'],phi=p104['phi'],xData=el_104_rot[0,:],yData=el_104_rot[1,:],theta=two_theta_104)
    fitEl110 = ellt.Ellipse(x0=p110['x0'],y0=p110['y0'],a=p110['a'],b=p110['b'],phi=p110['phi'],xData=el_110_rot[0,:],yData=el_110_rot[1,:],theta=two_theta_110)

    sos = 0
    sos += fitEl012.getSOS()
    sos += fitEl104.getSOS()
    sos += fitEl110.getSOS()

    # Compare found ellipses with experimental data
    # sos = 0
    # sos += ellt.get_sum_of_squares(el_012_rot[0,:],el_012_rot[1,:],p012)
    # sos += ellt.get_sum_of_squares(el_104_rot[0,:],el_104_rot[1,:],p104)
    # sos += ellt.get_sum_of_squares(el_110_rot[0,:],el_110_rot[1,:],p110)
    # sos += el.get_sum_of_squares(el_113_rot[0,:],el_113_rot[1,:],p113)
    
    # print(f"z0={z0:.0f}, phi={phi*180/pi:.0f}, theta={theta*180/pi:.0f}, psi={psi*180/pi:.0f}, (x0,y0)=({shift_x:.0f},{shift_y:.0f}) | sos={sos:.0f}")
    
    return sos



# ==============================================================================
# parameters: (z0,phi,theta,psi,shift_x,shift_y)
ansatz = [1200,0,-25/180*pi,0,-150,-1000]
# ansatz = [1184,-1/180*pi,-32/180*pi,0,-160,-1030]
# ansatz = [1100,0,-38.0,0,0,-500]
bounds = ((100,5000),(-pi,pi),(-35/180*pi,35/180*pi),(-pi,pi),(-10000,10000),(-10000,10000))

res = optimize.minimize(objective,ansatz,bounds=bounds,tol=1e-10)
z0,phi,theta,psi,shift_x,shift_y = res.x

print(f"SOS = {res.fun:.3f}")
print(f"Interactions = {res.nfev}")
print(f"\n[Calculated parameters]")
print(f"z0 = {z0:.4f}")
print(f"phi = {phi*180/pi:.4f}°")
print(f"theta = {theta*180/pi:.4f}°")
print(f"psi = {psi*180/pi:.4f}°")
print(f"x = {shift_x:.4f}")
print(f"y = {shift_y:.4f}")

print(f"z0,phi,theta,psi,shift_x,shift_y = [{z0:.4f},{phi:.4f},{theta:.4f},{psi:.4f},{shift_x:.4f},{shift_y:.4f}]")