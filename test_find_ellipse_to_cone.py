
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

# ellipse = el.Ellipse(x0=0,y0=0,a=2,b=1.5,phi=0.3,theta=theta)
ellipse = el.Ellipse(x0=0,y0=0,a=1,b=0.6,phi=0,theta=theta)
cone = ellipse.findCone()

cone.apex = np.array([1.15378,0.00000,0.62354])
new_n = np.array([-0.69282,0.00000,-0.72111])
new_n = new_n/np.linalg.norm(new_n)
cone.n = new_n

# Print the parameters
ellipse.print()
sM1 = np.array([ellipse.x0+ellipse.a*cos(ellipse.phi),ellipse.y0+ellipse.a*sin(ellipse.phi)])
sM2 = np.array([ellipse.x0-ellipse.a*cos(ellipse.phi),ellipse.y0-ellipse.a*sin(ellipse.phi)])
printc(f"c={ellipse.c:.2f}, sM1=[{sM1[0]},{sM1[1]}], sM2[{sM2[0]},{sM2[1]}]",fc='b')
cone.print()

# -----[ Create the figure ]-----
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.set_proj_type('ortho',None) # persp,0.1

ellipse.plot(ax,plotAxes=True)
ellipse.plotCone(ax)

# 1) cos(G) = (a^2+b^2-c^2)/(2*a*b)
# 2) s=((a+b+c)/2)
# 3) r=sqrt(s*(s-a)*(s-b)*(s-c))/s
# 4) cos(B) = (a^2+c^2-b^2)/(2*a*c)
# 5) r = l*tg(B/2)
# 6) tan(B/2) = sqrt(s*(s-a)*(s-b)*(s-c))/(l*s)
# -> put (2) into (6) to get:
# 6a) tan(B/2) = sqrt(((a+b+c)/2)*(((a+b+c)/2)-a)*(((a+b+c)/2)-b)*(((a+b+c)/2)-c))/(l*((a+b+c)/2))
# -> with alternate form:
# 6b) tan(B/2)=sqrt(-a^4+2a^2(b^2+c^2)-(b^2-c^2)^2)/(2l(a+b+c))


# + solve (1) for c:
# -> (7) c=sqrt(a^2-2*a*b*cos(G)+b^2)
# + put (7) into (4) and solve for b:
# -> (8a) b=a*sin(B)*csc(B-G)
# -> (8b) b=a*sin(B)*csc(B+G)

# C = A*(b-2*A)/c
# cos(G) = ((2*A)^2+b^2-c^2)/(2*2*A*b)


# TODO: Ends of semi-major axis can be found by rotating the cone axis by theta.
# This can be done by rotation matrix Q (as defined below). Just use cross
# product of cone axis and vector in xy plane rotated by phi.

# Here we find and plot points where cone intersects the semi-major axis
n = np.copy(ellipse.cone.n)
assert np.linalg.norm(n) == 1, "Cone axis must be a unit vector!"
# plt.plot([0,n[0]],[0,n[1]],[0,n[2]],'k')
phi = pi-np.arccos(n[0]/sqrt(n[0]**2+n[1]**2))
print(f"Found phi: {phi:.2f} ({phi*180/pi:.2f}°)")
n2 = el.rotate3D(n,phi,0,0)
# plt.plot([0,n2[0]],[0,n2[1]],[0,n2[2]],'r')
# print(n2)
n3 = np.zeros(3)
n3[0] = n2[0]*cos(ellipse.theta)-n2[2]*sin(ellipse.theta)
n3[1] = n2[1]
n3[2] = n2[0]*sin(ellipse.theta)+n2[2]*cos(ellipse.theta)
# plt.plot([0,n3[0]],[0,n3[1]],[0,n3[2]],'g')
# print(n3)
n4 = el.rotate3D(n3,-phi,0,0)
n5 = np.zeros(3)
n5[0] = n2[0]*cos(ellipse.theta)+n2[2]*sin(ellipse.theta)
n5[1] = n2[1]
n5[2] = -n2[0]*sin(ellipse.theta)+n2[2]*cos(ellipse.theta)
n6 = el.rotate3D(n5,-phi,0,0)
# plt.plot([0,n4[0]],[0,n4[1]],[0,n4[2]],'b')


V = ellipse.cone.apex
c = 1

plt.plot([V[0],V[0]+n4[0]*c],[V[1],V[1]+n4[1]*c],[V[2],V[2]+n4[2]*c],'r')
plt.plot([V[0],V[0]+n6[0]*c],[V[1],V[1]+n6[1]*c],[V[2],V[2]+n6[2]*c],'r')
# Semi-major axis end points
sM1 = np.array([V[0]-V[2]/n4[2]*n4[0],V[1]-V[2]/n4[2]*n4[1]])
sM2 = np.array([V[0]-V[2]/n6[2]*n6[0],V[1]-V[2]/n6[2]*n6[1]])
print(f"Found sM1: [{sM1[0]:.2f},{sM1[1]:.2f}], sM2: [{sM2[0]:.2f},{sM2[1]:.2f}]")
plt.plot(sM1[0],sM1[1],0,'ro')
plt.plot(sM2[0],sM2[1],0,'ro')
a = sqrt((sM1[0]-sM2[0])**2+(sM1[1]-sM2[1])**2)/2
print(f"Found a: {a:.2f}")


# center of ellipse 
x0 = ((V[0]-V[2]/n4[2]*n4[0])+(V[0]-V[2]/n6[2]*n6[0]))/2
y0 = ((V[1]-V[2]/n4[2]*n4[1])+(V[1]-V[2]/n6[2]*n6[1]))/2
plt.plot(x0,y0,0,'ro') # Checked


def getIncribedCircleRadius(a,b,c):
    s = (a+b+c)/2
    return sqrt((s-a)*(s-b)*(s-c)/s)

# Find radius of Dandelin sphere
tA = 2*a                                                # c
tB = sqrt((V[0]-sM1[0])**2+(V[1]-sM1[1])**2+V[2]**2)    # b
tC = sqrt((V[0]-sM2[0])**2+(V[1]-sM2[1])**2+V[2]**2)    # a
print(f"tA: {tA:.2f}, tB: {tB:.2f}, tC: {tC:.2f}")
r = getIncribedCircleRadius(tA,tB,tC)
print(f"Radius of Dandelin sphere: {r:.2f}")

print((tB-tC)/tA)
print((tB-tC)/2)

# Find c and b (via angle beta)
beta = np.arccos((tA**2+tC**2-tB**2)/(2*tA*tC))
l = r/tan(beta/2)
c = l-a
print(f"Found c: {c:.2f}")
b = sqrt(a**2-c**2)
print(f"Found b: {b:.2f}")

# Plot Dandelin circle
t = np.linspace(0,2*np.pi,50)   # Parameter of the circle
danX = x0 + c*cos(phi) + r*cos(t)*cos(phi)
danY = y0 + c*sin(phi) + r*cos(t)*sin(phi)
danZ = r + r*sin(t)
ax.plot(danX,danY,danZ,color=plt_clrs[1])
danX = x0 + c*cos(phi) - r*cos(t)*sin(phi)
danY = y0 + c*sin(phi) + r*cos(t)*cos(phi)
danZ = r + r*sin(t)
ax.plot(danX,danY,danZ,color=plt_clrs[1])
plt.plot(x0+c*cos(phi),y0+c*sin(phi),0,'ro',markersize=3)


sm1 = np.array([x0+b*sin(phi),y0-b*cos(phi),0])
sm2 = np.array([x0-b*sin(phi),y0+b*cos(phi),0])
plt.plot(sm1[0],sm1[1],sm1[2],'ro',markersize=5)
plt.plot(sm2[0],sm2[1],sm2[2],'ro',markersize=5)
ellipse2 = el.Ellipse(x0=x0,y0=y0,a=a,b=b,phi=phi,theta=theta,color=plt_clrs[1])
ellipse2.plot(ax,plotAxes=True)   





# ax.view_init(0,-90,0) # Side view



ax.set_aspect('equal')
plt.show()

raise SystemExit

# Following should find points at the ends of the semi-minor axis
# But it doesn't work because I rotate either wrong vector or correct vector by
# wrong angle.
# New idea: We know distance `d1`` from apex to the point where cone axis
# intersects xy plane. Distance `d2` from apex to the end of semi-minor axis can
# be calculated (hopefully) as cos(theta) = d1/d2. Now we have to find a point
# at semi-minor axis whose distance from apex is `d2`.

# Find coordinates of point where cone axis intersects plane z=0
S = np.array([V[0]-V[2]*n[0]/n[2],
              V[1]-V[2]*n[1]/n[2],0])

# Distance from the apex to the point where cone axis intersects xy plane
dS = sqrt((V[0]-S[0])**2+(V[1]-S[1])**2+V[2]**2)
plt.plot(S[0],S[1],0,'go')
# plt.quiver(V[0],V[1],V[2],n[0],n[1],n[2],color='b',length=dS)
print(f"Distance from apex to the point where cone axis intersects xy plane: {dS:.2f}")
dC = sqrt((V[0]-x0)**2+(V[1]-y0)**2+V[2]**2)
print(f"Distance from apex to center of ellipse: {dC:.2f}")
# Distance from the apex to the semi-minor axis end point
dB = dS/cos(theta)
print(f"Distance from apex to semi-minor axis end point: {dB:.2f}")

# Vector pointing along the semi-major axis
v1 = np.array([(V[0]-V[2]/n6[2]*n6[0])-(V[0]-V[2]/n4[2]*n4[0]),
               (V[1]-V[2]/n6[2]*n6[1])-(V[1]-V[2]/n4[2]*n4[1]),0])
v1 = v1/np.linalg.norm(v1) # Checked
# plt.plot([0,v1[0]],[0,v1[1]],[0,v1[2]],'k')
# Vector pointing along the semi-minor axis
v = np.array([-v1[1],v1[0],0])
v = v/np.linalg.norm(v) # Checked
plt.quiver(0,0,0,v[0],v[1],v[2],color='g',length=0.2)

# Find a point on the semi-minor axis which has distance `dB` from apex:
# Line coefficients:
if v[0] == 0:
    x = np.array([0,0])
    y = np.array([y0+sqrt(dB**2-dS**2),y0-sqrt(dB**2-dS**2)])
else:
    la = v[1]/v[0]
    lb = y0-la*x0
    # Quadratic equation coefficients:
    qa = 1+la**2
    qb = -2*V[0]-2*V[1]*la+2*la*lb
    qc = V[0]**2+V[1]**2+V[2]**2+lb**2-2*V[1]*lb-dB**2
    # Solve the quadratic equation
    if qb**2-4*qa*qc < 0:
        print("No real roots!")
    x = np.roots([qa,qb,qc])
    y = la*x+lb
plt.plot(x[0],y[0],0,'go')
plt.plot(x[1],y[1],0,'go')
plt.show()

# %%


# Vector from apex to center of ellipse
n1 = np.array([x0-V[0],y0-V[1],-V[2]])
n1 = n1/np.linalg.norm(n1) #Checked
# c = 5
# plt.plot([V[0],V[0]+n1[0]*c],[V[1],V[1]+n1[1]*c],[V[2],V[2]+n1[2]*c],'b')
u = np.cross(n1,v)
# plt.plot([0,v[0]], [0,v[1]], [0,v[2]],'b')
plt.plot([V[0],V[0]+u[0]],[V[1],V[1]+u[1]],[V[2],V[2]+u[2]],'r')

# # Vector `u` defines unit vector of rotation. Now rotate `n` around `u` by
# # `theta` using https://en.wikipedia.org/wiki/Rotation_matrix#Axis_and_angle
c = cos(theta)
s = sin(theta)
C = 1-c
x = u[0]
y = u[1]
z = u[2]

Q = np.array((
        (x*x*C+c,x*y*C-z*s,x*z*C+y*s),
        (x*y*C+z*s,y*y*C+c,y*z*C-x*s),
        (x*z*C-y*s,y*z*C+x*s,z*z*C+c)
))

u2 = Q@n1
c = 5
plt.plot([V[0],V[0]+u2[0]*c],[V[1],V[1]+u2[1]*c],[V[2],V[2]+u2[2]*c],'r')


xlim, ylim, zlim = ([-2,2],[-2,2],[-1,2])
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_zlim(zlim)


ax.set_aspect('equal')
plt.tight_layout()

# ax.view_init(90,-90,0) # Top view
# ax.view_init(0,-90,0) # Side view
plt.show()

