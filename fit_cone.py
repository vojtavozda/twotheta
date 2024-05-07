# %%

import numpy as np
from numpy import cos, sin, tan, pi
from scipy import optimize
from matplotlib import pyplot as plt
import elliptools as el

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

two_theta0 = two_theta_012
two_theta2 = two_theta_104

# Load data
data0_x = np.load('data/x_0.npy')
data0_y = np.load('data/y_0.npy')
data2_x = np.load('data/x_2.npy')
data2_y = np.load('data/y_2.npy')

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

def objective(params):

    sum_of_squares = 0

    Vx,Vy,z0,phi,delta = params

    # Calculate coordinates of the first ellipse
    s = z0*sin(two_theta0)/2*(1/cos(two_theta0+delta)-1/(cos(two_theta0-delta)))
    cx = Vx - s - z0*sin(delta)
    cy = Vy
    Vz = z0*cos(delta)

    # Rotate cone apex by phi around ellipse center
    Vx,Vy = rotate_point(Vx,Vy,cx,cy,phi)

    a = z0*sin(two_theta0)/2*(1/cos(two_theta0+delta)+1/cos(two_theta0-delta))
    b = z0*tan(two_theta0)

    sum_of_squares += el.get_sum_of_squares(data0_x,data0_y,(cx,cy,a,b,phi))

    # ---------------

    # Calculate coordinates of the first ellipse
    s2 = z0*sin(two_theta2)/2*(1/cos(two_theta2+delta)-1/(cos(two_theta2-delta)))
    # Vx,Vy = rotate_point(Vx,Vy,cx,cy,-phi)
    cx2 = Vx - s2 - z0*sin(delta)
    cy2 = Vy
    cx2,cy2 = rotate_point(cx2,cy2,cx,cy,phi)

    a = z0*sin(two_theta2)/2*(1/cos(two_theta2+delta)+1/cos(two_theta2-delta))
    b = z0*tan(two_theta2)

    sum_of_squares += el.get_sum_of_squares(data2_x,data2_y,(cx2,cy2,a,b,phi))

    # ex,ey = el.get_ellipse_pts((cx,cy,a,b,phi))

    return sum_of_squares

ansatz = [760,790,417,0.27,0.62]
bounds = ((None,None),(None,None),(1,None),(0,2*pi),(0,2*pi))
res = optimize.minimize(objective,ansatz,bounds=bounds)

# ------------------------------------------------------------------------------

Vx,Vy,z0,phi,delta = res.x

print(res.x)

# Vx,Vy,z0,phi,delta = (757,789,417,0.27,0.626)
# Vx,Vy,z0,phi,delta = (774,658,417,0.27,0.626)

# Calculate coordinates of the first ellipse
s = z0*sin(two_theta0)/2*(1/cos(two_theta0+delta)-1/(cos(two_theta0-delta)))
cx = Vx - s - z0*sin(delta)
cy = Vy

a = z0*sin(two_theta0)/2*(1/cos(two_theta0+delta)+1/cos(two_theta0-delta))
b = z0*tan(two_theta0)

ex0,ey0 = el.get_ellipse_pts((cx,cy,1*a,b,1*phi))

print(cx,cy,a,b,phi)

# ---------------

# Calculate coordinates of the first ellipse
s2 = z0*sin(two_theta2)/2*(1/cos(two_theta2+delta)-1/(cos(two_theta2-delta)))
# Vx,Vy = rotate_point(Vx,Vy,cx,cy,-phi)
cx2 = Vx - s2 - z0*sin(delta)
cy2 = Vy
cx2,cy2 = rotate_point(cx2,cy2,cx,cy,phi)

a = z0*sin(two_theta2)/2*(1/cos(two_theta2+delta)+1/cos(two_theta2-delta))
b = z0*tan(two_theta2)

ex2,ey2 = el.get_ellipse_pts((cx2,cy2,a,b,phi))

# -----------------

plt.scatter(data0_x,data0_y)
plt.scatter(data2_x,data2_y)
plt.plot(ex0,ey0)
plt.plot(ex2,ey2)
ax = plt.gca()
ax.set_aspect('equal')
plt.show()

