# %% 


import numpy as np
from numpy import pi, sin, cos
from matplotlib import pyplot as plt
from matplotlib import patches
import ellipse as el
import importlib
from scipy import optimize
import genlib as gl

# %% Fit two ellipses at once ==================================================

importlib.reload(el)

def objective2(p:list,x1,y1,x2,y2) -> float:

    x0 = p[0]
    y0 = p[1]
    phi = p[2]
    r = p[3]
    a1 = p[4]
    b1 = a1*r
    a2 = p[5]
    b2 = a2*r

    sos = 0
    sos += el.get_sum_of_squares(x1,y1,[x0,y0,a1,b1,phi])
    sos += el.get_sum_of_squares(x2,y2,[x0,y0,a2,b2,phi])

    return sos



x1 = np.load('x_2.npy')
y1 = np.load('y_2.npy')
x2 = np.load('x_2.npy')
y2 = np.load('y_2.npy')

# x1,y1 = el.get_ellipse([100,100,90,45,0])
# x2,y2 = el.get_ellipse([100,100,60,30,0])

ansatz = [500,900,3,1,700,1000]
bounds = ((-1000,1000),(000,1000),(0,2*pi),(0.1,100),(1,1000),(1,1000))
res = optimize.minimize(objective2,ansatz,args=(x1,y1,x2,y2),bounds=bounds)
print("Fit parameters:",res.x)
print("Final SOS:",res.fun)
x = res.x

x1f,y1f = el.get_ellipse([x[0],x[1],x[4],x[4]*x[3],x[2]])
x2f,y2f = el.get_ellipse([x[0],x[1],x[5],x[5]*x[3],x[2]])

plt.plot(x1,y1,'.',markersize=5,ls='',c='b')
plt.plot(x2,y2,'.',markersize=5,ls='',c='b')
plt.plot(x1f,y1f)
plt.plot(x2f,y2f)
plt.show()

x0 = x[0]
y0 = x[1]
axr = x[3]
phi = x[2]

data = np.load("data.npy")
data[data<0] = 0
data[data>30] = 30
plt.imshow(data,origin='lower',vmin=0,vmax=15)
plt.plot(x1f,y1f,c='r')
plt.plot(x2f,y2f,c='r')
for r in np.arange(100,2000,100):
    plt.gca().add_patch(patches.Ellipse((x0,y0),2*r,2*r*axr,angle=phi*180/pi,color='red',fill=False,ls='--'))
plt.xlim([0,data.shape[1]])
plt.ylim([0,data.shape[0]])
plt.show()

radii = np.arange(100,2000,2)
ellipse_sums = np.zeros(len(radii))
for i,r in enumerate(radii):
    ellipse_sums[i] = el.mask_sum(data,x0,y0,r,axr,phi,w=1)
radii = radii[ellipse_sums>0]
ellipse_sums = ellipse_sums[ellipse_sums>0]
plt.plot(radii,ellipse_sums)
plt.show()




# %% Fit one ellipse ===========================================================

importlib.reload(el)

def objective1(p:list,x:np.ndarray,y:np.ndarray) -> float:

    sos = el.get_sum_of_squares(x,y,p)
    return sos

x,y = el.get_ellipse([100,100,90,40,0])

x = np.load('x_5.npy')
y = np.load('y_5.npy')

ansatz = [500,730,1000,610,0.4]
bounds = ((0,1000),(400,1000),(1,1000),(1,1000),(0,2*pi))
res = optimize.minimize(objective1,ansatz,args=(x,y),bounds=bounds)
fit = res.x
print("Fit parameters:",res.x)

xf,yf = el.get_ellipse(res.x)
print("Final SOS:",el.get_sum_of_squares(x,y,res.x),'=',res.fun)
plt.plot(x,y,'.',markersize=10,ls='')
plt.plot(xf,yf)
plt.gca().axis('equal')
plt.show()

data = np.load("data.npy")
data[data<0] = 0
data[data>30] = 30
plt.imshow(data,origin='lower',vmin=0,vmax=15)
plt.plot(xf,yf,c='r')
for r in np.arange(100,2000,100):
    plt.gca().add_patch(patches.Ellipse((fit[0],fit[1]),2*r,2*r*fit[3]/fit[2],angle=fit[4]*180/pi,color='red',fill=False,ls='--'))
plt.xlim([0,data.shape[1]])
plt.ylim([0,data.shape[0]])
plt.show()

# fit0 = res.x
# fit2 = res.x
# fit4 = res.x
fit5 = res.x

# data = np.load('data.npy')
# data[data>30] = 30
# data[data<0] = 0

# %%


data = np.load("data.npy")
data[data<0] = 0
data[data>30] = 30


plt.imshow(data,origin='lower',vmin=0,vmax=15)
N = 100
radii = np.linspace(700,1000,N,endpoint=False)
fitA = fit0
fitB = fit5
# x0  = np.linspace(fitA[0],fitB[0],N)
# y0  = np.linspace(fitA[1],fitB[1],N)
# a   = np.linspace(fitA[2],fitB[2],N)
# b   = np.linspace(fitA[3],fitB[3],N)
# phi = np.linspace(fitA[4],fitB[4],N)
x0  = np.hstack([np.linspace(fit0[0],fit2[0],N),np.linspace(fit2[0],fit5[0],int(N/2))])
y0  = np.hstack([np.linspace(fit0[1],fit2[1],N),np.linspace(fit2[1],fit5[1],int(N/2))])
a   = np.hstack([np.linspace(fit0[2],fit2[2],N),np.linspace(fit2[2],fit5[2],int(N/2))])
b   = np.hstack([np.linspace(fit0[3],fit2[3],N),np.linspace(fit2[3],fit5[3],int(N/2))])
phi = np.hstack([np.linspace(fit0[4],fit2[4],N),np.linspace(fit2[4],fit5[4],int(N/2))])


ellipse_sums = np.zeros(len(x0))
for i in range(len(x0)):
    if i%10==0:
        plt.gca().add_patch(patches.Ellipse((fitA[0],fitA[1]),2*fitA[2],2*fitA[3],angle=fitA[4]*180/pi,color='red',fill=False,ls='--'))
        plt.gca().add_patch(patches.Ellipse((fitB[0],fitB[1]),2*fitB[2],2*fitB[3],angle=fitB[4]*180/pi,color='orange',fill=False,ls='--'))
        plt.gca().add_patch(patches.Ellipse((x0[i],y0[i]),2*a[i],2*b[i],angle=phi[i]*180/pi,color='green',fill=False))
    ellipse_sums[i] = el.mask_sum(data,x0[i],y0[i],a[i],b[i]/a[i],phi[i],w=1)
plt.show()

plt.plot(ellipse_sums)
plt.show()


# %% Fit using SVD - plot all ellipses =========================================
N = 8

data = np.load("data.npy")
data[data<0] = 0
data[data>30] = 30

plt.imshow(data,origin='lower',vmin=0,vmax=15)
for i in range(N):
    
    print(f"===[ Ellipse {i} ]===")
    

    # Load data
    x = np.load(f'x_{i}.npy')
    y = np.load(f'y_{i}.npy')
    plt.plot(x,y,'.',markersize=10,markeredgecolor=gl.plt_clrs[i])
    
    # Fit ellipse using SVD
    cart = el.fit_ellipse(x,y)
    params = el.cart_to_pol(cart)
    x0, y0, a, b, phi = params
    print(f'SVD: x0={x0:3.0f}, y0={y0:3.0f}, a={a:3.0f}, b={b:3.0f}, phi={phi:1.2f}')
    xel,yel = el.get_ellipse_pts(params)
    plt.plot(xel,yel,color=gl.plt_clrs[i])
    
    # Fit ellipse using scipy.optimize
    # ansatz = params
    # bounds = ((-2000,2000),(-2000,2000),(1,2000),(1,2000),(0,2*pi))
    # res = optimize.minimize(el.fit_ellipse_sos,ansatz,args=(x,y),bounds=bounds)
    # x0, y0, a, b, phi = res.x
    # print(f'SOS: x0={x0:3.0f}, y0={y0:3.0f}, a={a:3.0f}, b={b:3.0f}, phi={phi:1.2f}')
    # xel,yel = el.get_ellipse_pts(res.x)
    # plt.plot(xel,yel,color=gl.plt_clrs[i],ls='--')
    
plt.xlabel('x')
plt.ylabel('y')
plt.show()
