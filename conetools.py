
# %%
import os
import sys
import numpy as np
from numpy import sin, cos, pi, tan, sqrt
from numpy.linalg import norm, inv
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import importlib

import elliptools as el
from genlib import plt_clrs

importlib.reload(el)

clrs = plt.rcParams['axes.prop_cycle'].by_key()['color']

