
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

ellipse = el.Ellipse(x0=0,y0=0,a=1,b=0.6,phi=0.0,theta=theta)
cone = ellipse.findCone(theta)


# Print the parameters
ellipse.print()
cone.print()

# -----[ Create the figure ]-----
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.set_proj_type('ortho',None) # persp,0.1

# cone.plotMesh(ax,2)
ellipse.plot(ax,plotAxes=True)
ellipse.plotCone(ax)

# xlim, ylim, zlim = ([-4,3],[-4,4],[0,4])
# ax.set_xlim(xlim)
# ax.set_ylim(ylim)
# ax.set_zlim(zlim)



# ax.view_init(90,-90,0) # Top view
# ax.view_init(0,-90,0) # Side view
# plt.show()



V = cone.apex
cone.print()
cone.plotMesh(ax,2)
print(f"|V-|={sqrt((V[0]+ellipse.a)**2+V[2]**2)}")
print(f"|V+|={sqrt((V[0]-ellipse.a)**2+V[2]**2)}")

# A = ellipse.a
# C = ellipse.c
# G = cos(2*ellipse.theta)
# a = np.abs((-sqrt((G-1)*(C**2*(G+1)-2*A**2))+C*(-G)+C)/(G-1))
# b = np.abs(C-sqrt((G-1)*(C**2*(G+1)-2*A**2))/(G-1))
# print(a,b)
# a = (sqrt((G-1)*(C**2*(G+1)-2*A**2))+C*(-G)+C)/(G-1)
# b = sqrt((G-1)*(C**2*(G+1)-2*A**2))/(G-1)+C
# print(a,b)
# c = 2*ellipse.a


# W = np.array((0,0,0))*0.1
# beta = np.arccos((a**2+c**2-b**2)/(2*a*c))
# W[2] = a*tan(beta)
# W[0] = -A + sqrt(a**2-W[2]**2)
# plt.plot(W[0],W[1],W[2],'ro')

# # beta = np.arccos(((2*A)**2+c**2-b**2)/(2*2*A*c))
# # sin(beta) = V[2]
# wa1 = np.array((-W[0]+A,0,-W[2]))
# wa1 = wa1/np.linalg.norm(wa1)
# plt.plot([W[0],W[0]+wa1[0]],[W[1],W[1]+wa1[1]],[W[2],W[2]+wa1[2]],'r')
# ws = np.zeros(3)
# ws[0] = wa1[0]*cos(ellipse.theta)+wa1[2]*sin(ellipse.theta)
# ws[1] = wa1[1]
# ws[2] = -wa1[0]*sin(ellipse.theta)+wa1[2]*cos(ellipse.theta)
# wa2 = np.zeros(3)
# wa2[0] = ws[0]*cos(ellipse.theta)+ws[2]*sin(ellipse.theta)
# wa2[1] = ws[1]
# wa2[2] = -ws[0]*sin(ellipse.theta)+ws[2]*cos(ellipse.theta)
# plt.plot([W[0],W[0]+ws[0]],[W[1],W[1]+ws[1]],[W[2],W[2]+ws[2]],'r')
# k = 2.3
# plt.plot([W[0],W[0]+wa2[0]*k],[W[1],W[1]+wa2[1]*k],[W[2],W[2]+wa2[2]*k],'r')

# cone = el.Cone(apex=W,axis=ws,theta=theta)
# cone.plotMesh(ax,2)


C = sqrt(ellipse.a**2-ellipse.b**2)
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

W = np.array((0,0,0))*0.1
W[2] = b*sin(alpha)
W[0] = W[2]/tan(alpha)-ellipse.a
plt.plot(W[0],W[1],W[2],'go')
wa2 = np.array((ellipse.a-W[0],0,-W[2]))
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

cone = el.Cone(apex=W,axis=ws,theta=theta)
cone.plotMesh(ax,2)
cone.print(5)


ax.set_aspect('equal')
plt.tight_layout()
plt.show()