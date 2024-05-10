
# %%
import os
import numpy as np
from numpy import sin, cos, tan, pi, sqrt
import elliptools as ellt
from matplotlib import pyplot as plt
import importlib
from clrprint import printc
from genlib import plt_clrs
from scipy import optimize
importlib.reload(ellt)


# change current directory to the one where this script is located
os.chdir(os.path.dirname(os.path.abspath(__file__)))

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

data = np.load('data/data.npy')
data[data>30] = 30
data[data<0] = 0

conic_012 = np.load('data/conic_012.npy')
conic_104 = np.load('data/conic_104.npy')
conic_110 = np.load('data/conic_110.npy')
conic_113 = np.load('data/conic_113.npy')

el012 = ellt.Ellipse(xData=conic_012[0],yData=conic_012[1],color=plt_clrs[0],theta=two_theta_012)
el104 = ellt.Ellipse(xData=conic_104[0],yData=conic_104[1],color=plt_clrs[2],theta=two_theta_104)
el110 = ellt.Ellipse(xData=conic_110[0],yData=conic_110[1],color=plt_clrs[3],theta=two_theta_110)

el012.fit()
el012.findCone()

el104.fit()
el104.findCone()

el110.fit()
el110.findCone()


fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.set_proj_type('ortho',None) # persp,0.1

el012.plot(ax,plotAxes=True,plotData=True,dataColor=9)
el012.cone.plotWireframe(ax)

el104.plot(ax,plotAxes=True,plotData=True,dataColor=8)
el104.cone.plotWireframe(ax)

el110.plot(ax,plotAxes=True,plotData=True,dataColor=1)
el110.cone.plotWireframe(ax)

# top view
ax.view_init(elev=90,azim=0)
plt.show()


# %%
importlib.reload(ellt)

# First, define a test ellipse
theta = 0.5
ellipse = ellt.Ellipse(x0=0.5,y0=0,a=1,b=0.7,phi=0.1,theta=theta)
ellipse.print(f=5)
printc("[Exact cone] ",fw='b',fc='g',end='')
ellipse.findCone().print()

x,y = ellipse.getPoints(tmin=0,tmax=2*pi/2)
ellipse.setData(x,y)
print(ellipse.getSOS())

counter = 0

def objectiveSingle(params):
    global counter

    V = np.array(params[:3])
    n = np.array(params[3:])
    n = n/np.linalg.norm(n)
    # print(f"Apex:[{V[0]:.2f},{V[1]:.2f},{V[2]:.2f}] | n:[{n[0]:.2f},{n[1]:.2f},{n[2]:.2f}]",end="")
    cone = ellt.Cone(V,n,theta)
    # cone.print()
    el = cone.getEllipse()
    el.setData(x,y)

    # fig = plt.figure()
    # ax = fig.add_subplot(111,projection='3d')
    # ax.set_proj_type('ortho',None)
    # el.plot(ax,plotAxes=True,plotData=True,dataColor=1,color=3)
    # ellipse.plot(ax,plotAxes=True)
    # cone.plotWireframe(ax)


    sos = el.getSOS()
    # print(f" --> SOS:{sos}")

    # counter+=1
    # if counter>5:
    #     raise SystemExit
    return sos

def unitVectorConstraint(params):
    n = np.array(params[3:])
    # Return the difference between the magnitude and 1
    return np.abs(np.linalg.norm(n) - 1)

ansatz = [1.4,0,1,-0.6,0,-1]
bounds = ((-5,5),(-5,5),(0,5),(-1,1),(-1,1),(-1,0))
constraints = [{'type': 'eq', 'fun': unitVectorConstraint}]


V = np.array(ansatz[:3])
n = np.array(ansatz[3:])
n = n/np.linalg.norm(n)
cone = ellt.Cone(V,n,theta,color=3)
el = cone.getEllipse()
cone.print()

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.set_proj_type('ortho',None)
cone.plotWireframe(ax)
el.plot(ax,plotAxes=True,color=3)
ellipse.plot(ax,plotAxes=True,plotData=True)
ellipse.cone.plotWireframe(ax)

ax.set_aspect('equal')
plt.show()

res = optimize.minimize(objectiveSingle,ansatz,bounds=bounds,constraints=constraints)


V = np.array(res.x[:3])
n = np.array(res.x[3:])
n = n/np.linalg.norm(n)
cone = ellt.Cone(V,n,theta,color=3)
el = cone.getEllipse()
cone.print()

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.set_proj_type('ortho',None)
cone.plotWireframe(ax)
el.plot(ax,plotAxes=True,color=3)
ellipse.plot(ax,plotAxes=True,plotData=True)
ellipse.cone.plotWireframe(ax)

ax.set_aspect('equal')
plt.show()

# %%
importlib.reload(ellt)


def objectiveMulti(params):

    V = np.array(params[:3])
    n = np.array(params[3:])
    n = n/np.linalg.norm(n)

    # print(f"Apex:[{V[0]:.2f},{V[1]:.2f},{V[2]:.2f}] | n:[{n[0]:.2f},{n[1]:.2f},{n[2]:.2f}]",end="")

    sos = 0
    fitEl012 = ellt.Cone(V,n,two_theta_012).getEllipse()
    fitEl012.setData(el012.xData,el012.yData)
    sos += fitEl012.getSOS()

    fitEl104 = ellt.Cone(V,n,two_theta_104).getEllipse()
    fitEl104.setData(el104.xData,el104.yData)
    sos += fitEl104.getSOS()

    fitEl110 = ellt.Cone(V,n,two_theta_110).getEllipse()
    fitEl110.setData(el110.xData,el110.yData)
    sos += fitEl110.getSOS()

    # print(f" --> SOS:{sos}")

    # counter+=1
    # if counter>5:
    #     raise SystemExit
    return sos

def unitVectorConstraint(params):
    n = np.array(params[3:])
    # Return the difference between the magnitude and 1
    return np.abs(np.linalg.norm(n) - 1)

cone012 = el012.getCone()
printc(f"[ANSATZ 012] ",fw='b',end='')
cone012.print()

cone104 = el104.getCone()
printc(f"[ANSATZ 104] ",fw='b',end='')
cone104.print()

cone110 = el110.getCone()
printc(f"[ANSATZ 110] ",fw='b',end='')
cone110.print()

V = np.array([np.mean([cone012.apex[0],cone104.apex[0],cone110.apex[0]]),
              np.mean([cone012.apex[1],cone104.apex[1],cone110.apex[1]]),
              np.mean([cone012.apex[2],cone104.apex[2],cone110.apex[2]])])
n = np.array([np.mean([cone012.n[0],cone104.n[0],cone110.n[0]]),
              np.mean([cone012.n[1],cone104.n[1],cone110.n[1]]),
              np.mean([cone012.n[2],cone104.n[2],cone110.n[2]])])
n = n/np.linalg.norm(n)
mCone = ellt.Cone(V,n,theta)
# mCone = cone

ansatz = [mCone.apex[0],mCone.apex[1],mCone.apex[2],mCone.n[0],mCone.n[1],mCone.n[2]]

# ----------------- [ Plot ansatz] -----------------
V = np.array(ansatz[:3])
n = np.array(ansatz[3:])
n = n/np.linalg.norm(n)

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.set_proj_type('ortho',None)
el012.plotData(ax)
el104.plotData(ax)
el110.plotData(ax)

cone012 = ellt.Cone(V,n,two_theta_012,color=el012.clr)
cone012.plotWireframe(ax)
cone012.getEllipse().plot(ax,plotAxes=True)

cone104 = ellt.Cone(V,n,two_theta_104,color=el104.clr)
cone104.plotWireframe(ax)
cone104.getEllipse().plot(ax,plotAxes=True)

cone110 = ellt.Cone(V,n,two_theta_110,color=el110.clr)
cone110.plotWireframe(ax)
cone110.getEllipse().plot(ax,plotAxes=True)

ax.set_aspect('equal')
plt.show()

# ----------------- [ Optimize ] -----------------
methods = ['Nelder-Mead','Powell','CG','BFGS','Newton-CG','L-BFGS-B','TNC','COBYLA','SLSQP','trust-const','dogleg','trust-ncg','trust-exact','trust-krylov']
bounds = ((100,2000),(500,2000),(100,1000),(-0.5,0),(-1,0),(-1,0))
# constraints = [{'type': 'eq', 'fun': unitVectorConstraint}]
# constraints = []

res = optimize.minimize(objectiveMulti,ansatz,
                        method = methods[1],
                        bounds=bounds)


V = np.array(res.x[:3])
n = np.array(res.x[3:])
n = n/np.linalg.norm(n)
cone = ellt.Cone(V,n,theta,color=3)
el = cone.getEllipse()
printc("[RESULT] ",fw='b',end='')
cone.print()

# ----------------- [ Plot result ] -----------------
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.set_proj_type('ortho',None)
el012.plotData(ax)
el104.plotData(ax)
el110.plotData(ax)

cone012 = ellt.Cone(V,n,two_theta_012,color=el012.clr)
cone012.plotWireframe(ax)
cone012.getEllipse().plot(ax,plotAxes=True)

cone104 = ellt.Cone(V,n,two_theta_104,color=el104.clr)
cone104.plotWireframe(ax)
cone104.getEllipse().plot(ax,plotAxes=True)

cone110 = ellt.Cone(V,n,two_theta_110,color=el110.clr)
cone110.plotWireframe(ax)
cone110.getEllipse().plot(ax,plotAxes=True)

plt.xlim(0,2000)
plt.ylim(0,2000)

ax.set_aspect('equal')
plt.show()