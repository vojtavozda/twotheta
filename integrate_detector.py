# %%

import numpy as np
from numpy import sin, cos, pi, tan, sqrt
from numpy.linalg import norm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import importlib

import ellipse as el
from genlib import plt_clrs

importlib.reload(el)

# ------------------------------------------------------------------------------

data = np.load("data.npy")
data[data<0] = 0
data[data>30] = 30

print(data.shape)


# %%
