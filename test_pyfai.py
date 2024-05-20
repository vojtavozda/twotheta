# %%

# Notes:
# ======
# 1) How to obtain a poni file:
# -----------------------------
# See video tutorial at:
# https://pyfai.readthedocs.io/en/v2023.1/usage/cookbook/calib-gui/index.html
# - install packages: `pip install pyfai fabio pyqt5 PyOpenGL`
# - open pyFAI GUI using `pyFAI-calib2`
# - set energy to 5.932 keV
# - calibrant is Al2O3
# - define custom detector (512x1024px, pixel 1um)
# - load some image with rings for geometry calibration (*.npy)
# - define mask (used mainly for peak picking)
# - inside peak picking, correct ring number must be selected (old data have
#   rings 1, 2, 3, 5 - 4 is missing!)
# - fit the geometry and save the poni file
#
# 2) Perform 1D and 2D integration:
# ---------------------------------
# - load data and poni file (-> AzimuthalIntegrator)
# - use methods of AzimuthalIntegrator to perform 1D and 2D integration


import os
import sys
from matplotlib import pyplot as plt
from pyFAI.utils.ellipse import fit_ellipse
import inspect
from matplotlib import patches
from numpy import rad2deg
import pyFAI
import fabio

os.chdir(os.path.dirname(os.path.abspath(__file__)))

filepath = os.path.join("p2838","JF3_run135.npy")
data = fabio.open(filepath).data
data[data<0] = 0
data[data>20] = 20

ponipath = os.path.join("p2838","fit.poni")
ai = pyFAI.load(ponipath)

res1D = ai.integrate1d_ng(data,1024,unit="2th_deg")
res2D = ai.integrate2d_ng(data,1024,360,unit="2th_deg")

plt.imshow(data)
plt.show()
plt.imshow(res2D[0],extent=[res2D[1].min(),res2D[1].max(),res2D[2].min(),res2D[2].max()])
# set axpect NOT to be equal
plt.gca().set_aspect('auto')

plt.show()
plt.plot(res1D[0],res1D[1])
plt.show()
# %%
