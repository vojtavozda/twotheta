# %%

import os
import numpy as np
from matplotlib import pyplot as plt
import elliptools as ellt
import importlib

# change current directory to the one where this script is located
os.chdir(os.path.dirname(os.path.abspath(__file__)))

clrs = plt.rcParams['axes.prop_cycle'].by_key()['color']
data = np.load('data/data.npy')
data[data>30] = 30
data[data<0] = 0
plt.imshow(data)

# Data were exported using `find_points.py` script
conic_012 = np.load('data/conic_012.npy')
conic_104 = np.load('data/conic_104.npy')
conic_110 = np.load('data/conic_110.npy')
conic_113 = np.load('data/conic_113.npy')

conic_012_prb = np.load('data/conic_012_prb.npy')
conic_104_prb = np.load('data/conic_104_prb.npy')
conic_110_prb = np.load('data/conic_110_prb.npy')
conic_113_prb = np.load('data/conic_113_prb.npy')

plt.plot(conic_012[0],conic_012[1],color=clrs[1],linewidth=2)
plt.plot(conic_104[0],conic_104[1],color=clrs[1],linewidth=2)
plt.plot(conic_110[0],conic_110[1],color=clrs[1],linewidth=2)
plt.plot(conic_113[0],conic_113[1],color=clrs[1],linewidth=2)

plt.plot(conic_104_prb[0],conic_104_prb[1],color=clrs[3],linewidth=2)
plt.plot(conic_012_prb[0],conic_012_prb[1],color=clrs[3],linewidth=2)
plt.plot(conic_110_prb[0],conic_110_prb[1],color=clrs[3],linewidth=2)
plt.plot(conic_113_prb[0],conic_113_prb[1],color=clrs[3],linewidth=2)




coeffs_012_cart = ellt.fit_ellipse(conic_012[0], conic_012[1])
coeffs_104_cart = ellt.fit_ellipse(conic_104[0], conic_104[1])
coeffs_110_cart = ellt.fit_ellipse(conic_110[0], conic_110[1])
coeffs_113_cart = ellt.fit_ellipse(conic_113[0], conic_113[1])

x012,y012 = ellt.get_ellipse_pts(ellt.cart_to_pol(coeffs_012_cart))
x104,y104 = ellt.get_ellipse_pts(ellt.cart_to_pol(coeffs_104_cart))
x110,y110 = ellt.get_ellipse_pts(ellt.cart_to_pol(coeffs_110_cart))
x113,y113 = ellt.get_ellipse_pts(ellt.cart_to_pol(coeffs_113_cart))

plt.plot(x012,y012,color=clrs[1],linestyle=':')
plt.plot(x104,y104,color=clrs[1],linestyle=':')
plt.plot(x110,y110,color=clrs[1],linestyle=':')
plt.plot(x113,y113,color=clrs[1],linestyle=':')


coeffs_012_prb_cart = ellt.fit_ellipse(conic_012_prb[0], conic_012_prb[1])
coeffs_104_prb_cart = ellt.fit_ellipse(conic_104_prb[0], conic_104_prb[1])
coeffs_110_prb_cart = ellt.fit_ellipse(conic_110_prb[0], conic_110_prb[1])
coeffs_113_prb_cart = ellt.fit_ellipse(conic_113_prb[0], conic_113_prb[1])

x012_prb,y012_prb = ellt.get_ellipse_pts(ellt.cart_to_pol(coeffs_012_prb_cart))
x104_prb,y104_prb = ellt.get_ellipse_pts(ellt.cart_to_pol(coeffs_104_prb_cart))
x110_prb,y110_prb = ellt.get_ellipse_pts(ellt.cart_to_pol(coeffs_110_prb_cart))
x113_prb,y113_prb = ellt.get_ellipse_pts(ellt.cart_to_pol(coeffs_113_prb_cart))

plt.plot(x012_prb,y012_prb,color=clrs[3],linestyle=':')
plt.plot(x104_prb,y104_prb,color=clrs[3],linestyle=':')
plt.plot(x110_prb,y110_prb,color=clrs[3],linestyle=':')
plt.plot(x113_prb,y113_prb,color=clrs[3],linestyle=':')

plt.show()

# %%
importlib.reload(ellt)

cone_apex = np.array([0,0,0])
cone_axis = np.array([0,0,1])
distance = 1
two_theta = 0.01

params = ellt.get_ellipse_from_general_cone(cone_apex,cone_axis,distance,two_theta)

print(params)

# %%

