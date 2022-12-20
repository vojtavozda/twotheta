# %%

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
import cv2
from scipy import optimize

def CircleSum(data,cx,cy,r) -> float:
    """
    Create circular mask with cx,cy,r and calculate sum of `data` below the
    mask.
    """
    
    mask = data.copy()*0
    cv2.circle(mask,(int(cx),int(cy)),int(r),1,5)
    overlap = mask*data
    if np.sum(mask) == 0:
        return 0
    else:
        return np.sum(overlap)/np.sum(mask)

def CircleStd(data:np.ndarray,cx,cy,r) -> float:

    mask = data.copy()*0
    cv2.circle(mask,(int(cx),int(cy)),int(r),1,5)
    overlap = (mask*data).flatten()
    vals = overlap[overlap>0]
    if np.sum(vals)==0:
        return 0
    else:
        return np.std(vals)


def objective(p,data) -> float:
    cs = CircleSum(data,p[0],p[1],p[2])
    print(f"sum = {cs:.0f}, cx = {p[0]:.0f}, cy = {p[1]:.0f}, r = {p[2]:.0f}")
    return 100 - cs

def objectiveStd(p,data) -> float:
    std_sum = 0
    for r in range(400):
        std_sum += CircleStd(data,p[0],p[1],r)
    print(f"std_sum = {std_sum:.0f}, cx = {p[0]:.0f}, cy = {p[1]:.0f}")        
    return std_sum


# Create fictional experimental data
empty = np.zeros([512,512])
data = empty.copy()
R = [60,61,62,63,64,65,66,67,68]
V = [1,2,4,8,16,8,4,2,1]
cv2.circle(data,(255,255),190,50,10)
cv2.circle(data,(255,255),210,50,1)
cv2.circle(data,(255,255),350,50,10)
for r,v in zip(R,V):
    temp = empty.copy()
    cv2.circle(temp,(255,255),r,v,10)
    cv2.circle(temp,(255,255),r+50,v*2,5)
    cv2.circle(temp,(255,255),r+80,v/2,20)
    cv2.circle(temp,(255,255),r+200,v/2,1)
    cv2.circle(temp,(255,255),r+230,v,1)
    cv2.circle(temp,(255,255),r+260,v*2,5)
    data += temp

data += (np.random.random((512,512))-0.5)*50
data[data<0] = 0

# fig,ax = plt.subplots()
# ax.imshow(data,vmin=0,vmax=90)

# radii = np.arange(0,500,1)
# circle_sums = np.zeros(500)
# circle_stds = np.zeros(500)
# for i,r in enumerate(radii):
#     cx = 255
#     cy = 255
#     circle_sums[i] = CircleSum(data,cx,cy,r)
#     circle_stds[i] = CircleStd(data,cx,cy,r)
#     if (i%20==0):
#         ax.add_patch(patches.Circle((cx,cy),r,color='red',fill=False))
# plt.show()


# plt.plot(radii,circle_sums)
# plt.show()

# plt.plot(radii,circle_stds)
# plt.show()
# print(np.sum(circle_stds))

# cxs = np.arange(200,300,1)
# circle_stds = cxs*0
# for i,cx in enumerate(cxs):
#     cy = 255
#     for r in range(0,400):
#         circle_stds[i] += CircleStd(data,cx,cy,r)

# plt.plot(cxs,circle_stds)
# plt.show()

# %%

ansatz = [300,300]
res = optimize.minimize(objectiveStd,ansatz,args=data,tol=1,
method = 'Nelder-Mead')
 
fig,ax = plt.subplots()
ax.imshow(data,vmin=0,vmax=90)
ax.add_patch(patches.Circle((ansatz[0],ansatz[1]),50,color='red',fill=False,ls=':',lw=2))
ax.add_patch(patches.Circle((res.x[0],res.x[1]),50,color='red',fill=False,ls='--',lw=2))
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
