
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
theta = 30/180*pi

ellipse = el.Ellipse(x0=1,y0=0.5,a=1,b=0.6,phi=0.3,theta=theta)
cone = ellipse.findCone(theta) # ! This is wrong

# Print the parameters
ellipse.print()
cone.print()

# -----[ Create the figure ]-----
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.set_proj_type('ortho',None) # persp,0.1

# cone.plotMesh(ax,2)
ellipse.plot(ax,plotAxes=True)
# ellipse.plotCone(ax)

# xlim, ylim, zlim = ([-4,3],[-4,4],[0,4])
# ax.set_xlim(xlim)
# ax.set_ylim(ylim)
# ax.set_zlim(zlim)

# ax.view_init(90,-90,0) # Top view
# ax.view_init(0,-90,0) # Side view
# plt.show()

# cone.apex = np.array([1.15378,0.00000,0.62354])
# new_n = np.array([-0.69282,0.00000,-0.72111])

C = ellipse.c
A = ellipse.a
G = cos(2*ellipse.theta)
c = 2*ellipse.a
b = C-sqrt((G-1)*(C**2*(G+1)-2*A**2))/(G-1)
print(f"new b1={b}")
b = C-sqrt((G-1)*(C**2*(G+1)-2*A**2))/(G-1)
print(f"new b2={b}")
a = b-2*ellipse.c
alpha = np.arccos((b**2+c**2-a**2)/(2*b*c))
print(f"alpha={alpha*180/pi}")

phi = ellipse.phi
W = np.array((0,0,0))*0.1
W[2] = b*sin(alpha)
delta = W[2]/tan(alpha)-ellipse.a
W[0] = ellipse.x0 + delta*cos(phi)
W[1] = ellipse.y0 + delta*sin(phi)

plt.plot(W[0],W[1],W[2],'go')
wa2 = np.array(((ellipse.x0+ellipse.a*cos(phi))-W[0],
                (ellipse.y0+ellipse.a*sin(phi))-W[1],
                -W[2]))
wa2 = wa2/np.linalg.norm(wa2)
k = 1
plt.plot([W[0],W[0]+wa2[0]*k],[W[1],W[1]+wa2[1]*k],[W[2],W[2]+wa2[2]*k],'g')
ws = np.zeros(3)
ws[0] = wa2[0]*cos(ellipse.theta)+wa2[2]*sin(ellipse.theta)
ws[1] = wa2[1]
ws[2] = -wa2[0]*sin(ellipse.theta)+wa2[2]*cos(ellipse.theta)

wa1 = np.zeros(3)
wa1[0] = ws[0]*cos(ellipse.theta)+ws[2]*sin(ellipse.theta)
wa1[1] = ws[1]
wa1[2] = -ws[0]*sin(ellipse.theta)+ws[2]*cos(ellipse.theta)
k = 2.3
plt.plot([W[0],W[0]+wa1[0]*k],[W[1],W[1]+wa1[1]*k],[W[2],W[2]+wa1[2]*k],'g')


wa2_2 = el.rotate3D(wa2,phi,0,0)
wa2_3 = np.zeros(3)
wa2_3[0] = wa2_2[0]*cos(2*ellipse.theta)+wa2_2[2]*sin(2*ellipse.theta)
wa2_3[1] = wa2_2[1]
wa2_3[2] = -wa2_2[0]*sin(2*ellipse.theta)+wa2_2[2]*cos(2*ellipse.theta)
wa2_4 = el.rotate3D(wa2_3,-phi,0,0)
k = 2.3
plt.plot([W[0],W[0]+wa2_4[0]*k],[W[1],W[1]+wa2_4[1]*k],[W[2],W[2]+wa2_4[2]*k],'r')


cone = el.Cone(apex=W,axis=ws,theta=theta)
cone.plotMesh(ax,2)
cone.print(5)


ax.set_aspect('equal')
plt.tight_layout()
plt.show()