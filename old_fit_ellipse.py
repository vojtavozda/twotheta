# %%

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
import cv2
from scipy import optimize

def EllipseSum(data,cx,cy,r,axr,angle) -> float:
    """
    Create circular mask with cx,cy,r and calculate sum of `data` below the
    mask.
    """
    
    mask = data.copy()*0
    cv2.ellipse(mask,(int(cx),int(cy)),(int(r*axr),int(r)),angle,0,360,1,2)
    overlap = mask*data
    if np.sum(mask) == 0:
        return 0
    else:
        return np.sum(overlap)/np.sum(mask)

def EllipseStd(data:np.ndarray,cx,cy,r,axr,angle) -> float:

    mask = data.copy()*0
    cv2.ellipse(mask,(int(cx),int(cy)),(int(r*axr),int(r)),angle,0,360,1,2)
    overlap = (mask*data).flatten()
    vals = overlap[overlap>0]
    if np.sum(vals)==0:
        return 0
    else:
        return np.std(vals)*np.std(vals)


def objective(p,data) -> float:
    cs = EllipseSum(data,p[0],p[1],p[2],p[3],p[4])
    print(f"sum = {cs:.0f}, cx = {p[0]:.0f}, cy = {p[1]:.0f}, r = {p[2]:.0f}, axr = {p[3]:.2f}, angle = {p[4]:.0f}")
    return 100 - cs

def objectiveStd(p,data) -> float:
    std_sum = 0
    for r in range(10,3000,10):
        std_sum += EllipseStd(data,p[0],p[1],r,p[2],p[3])
    print(f"std_sum = {std_sum:.0f}, cx = {p[0]:.0f}, cy = {p[1]:.0f}, axr = {p[2]:.2f}, angle = {p[3]:.0f}")
    return std_sum

# Create fictional experimental data
empty = np.zeros([512,512])
data = empty.copy()
R = [60,61,62,63,64,65,66,67,68]
V = [1,2,4,8,16,8,4,2,1]
axr = 1.3
angle = 20
cv2.ellipse(data,(255,255),(int(190*axr),190),angle,0,360,50,10)
cv2.ellipse(data,(255,255),(int(210*axr),210),angle,0,360,50,1)
cv2.ellipse(data,(255,255),(int(350*axr),350),angle,0,360,50,10)
for r,v in zip(R,V):
    temp = empty.copy()
    cv2.ellipse(temp,(255,255),(int((r    )*axr),r    ),angle,0,360,v  ,10)
    cv2.ellipse(temp,(255,255),(int((r+50 )*axr),r+50 ),angle,0,360,v*2, 5)
    cv2.ellipse(temp,(255,255),(int((r+80 )*axr),r+80 ),angle,0,360,v/2,20)
    cv2.ellipse(temp,(255,255),(int((r+200)*axr),r+200),angle,0,360,v/2, 1)
    cv2.ellipse(temp,(255,255),(int((r+230)*axr),r+230),angle,0,360,v  , 1)
    cv2.ellipse(temp,(255,255),(int((r+260)*axr),r+260),angle,0,360,v*2, 5)
    data += temp

data[data<0] = 0
data += (np.random.random(data.shape)-0.5)*50

data = np.load("data/data.npy")
data[data<0] = 0
data[data>30] = 30


fig,ax = plt.subplots()
ax.imshow(data,vmin=0,vmax=20)
angle = 0.409*180/np.pi
axr = 780/407
cx = 498
cy = 588
radii = np.arange(100,1500,1)
ellipse_sums = radii*0.0
ellipse_stds = radii*0.0
for i,r in enumerate(radii):
    # ellipse_sums[i] = EllipseSum(data,cx,cy,r,axr,angle)
    # ellipse_stds[i] = EllipseStd(data,cx,cy,r,axr,angle)
    if (r%100==0):
        ax.add_patch(patches.Ellipse((cx,cy),2*r*axr,2*r,angle=angle,color='red',fill=False))
plt.show()

# %%

data = np.load("data/data.npy")
data[data<0] = 0
data[data>30] = 30

x0 = 159
y0 = 759
axr = 1.5
a = 613
b = a/r
phi = 3.188

plt.imshow(data)
for r in np.arange(100,2000,100):
    plt.gca().add_patch(patches.Ellipse((x0,y0),2*r*axr,2*r,angle=phi,color='red',fill=False))
plt.show()

radii = np.arange(100,2000,2)
ellipse_sums = np.zeros(len(radii))
for i,r in enumerate(radii):
    ellipse_sums[i] = EllipseSum(data,x0,y0,r,axr,phi)
radii = radii[ellipse_sums>0]
ellipse_sums = ellipse_sums[ellipse_sums>0]
plt.plot(radii,ellipse_sums)
plt.show()


# %%

ansatz = [200,1000,1.2,10]
bounds = ((None,None),(None,None),(1,10),(0,360))
res = optimize.minimize(objectiveStd,ansatz,args=data,tol=1,bounds=bounds,
    method = 'Nelder-Mead')

fig,ax = plt.subplots()
ax.imshow(data,vmin=0,vmax=20)
ax.add_patch(patches.Ellipse((ansatz[0],ansatz[1]),100*ansatz[2],100,angle=ansatz[3],color='red',fill=False,ls=':',lw=2))
for r in np.arange(0,3000,200):
    ax.add_patch(patches.Ellipse((res.x[0],res.x[1]),r*res.x[2],r,angle=res.x[3],color='red',fill=False,ls='--',lw=2))
plt.show()


# %%

ansatz = [230,250,70]
bounds = ((0,None),(0,None),(0,None))

res = optimize.minimize(objective,
ansatz,args=(data),
bounds=bounds,
method = 'Nelder-Mead'
)

cx = res.x[0]
cy = res.x[1]
r = res.x[2]
fig,ax = plt.subplots()
ax.imshow(data,vmin=0,vmax=90)
ax.add_patch(patches.Circle((ansatz[0],ansatz[1]),ansatz[2],color='red',fill=False,ls=':',lw=2))
ax.add_patch(patches.Circle((cx,cy),r,color='red',fill=False,ls='--',lw=2))
plt.show()



# %%


meanJF3 = np.load("data.npy")
plt.imshow(meanJF3,vmin=0,vmax=30)
plt.show()
