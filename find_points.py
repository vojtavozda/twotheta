# %%

import numpy as np
from matplotlib import pyplot as plt


data:np.ndarray = np.load("data.npy")
data[data>30] = 30
data[data<0] = 0

w = 10
y0 = 30

xdata = np.arange(0,data.shape[1])
ydata = xdata*0

for x in range(data.shape[1]):
    y1 = y0-w/2
    y2 = y0+w/2
    if y1<0:
        y2 -= y1
        y1 = 0
    elif y2>data.shape[0]:
        y1 -= y2-data.shape[0]
        y2 = data.shape[0]
    ysel = np.arange(y1,y2).astype(int)
    data_ = data[ysel,x]
    counter = 1
    for i in np.arange(0,0):
        if x+i > 0 and x+i < data.shape[1]:
            data_ += data[ysel,x+i]
            counter += 1
    data_ = data_/counter
    y0 = ysel[np.argmax(data_)]
    # y0 = ysel[np.argmax(data[ysel,x])]
    if x == 888: y0 = 0
    ydata[x] = y0

x0 = np.load('x_0.npy')
y0 = np.load('y_0.npy')
x1 = np.load('x_1.npy')
y1 = np.load('y_1.npy')
x2 = np.load('x_2.npy')
y2 = np.load('y_2.npy')
x3 = np.load('x_3.npy')
y3 = np.load('y_3.npy')
x4 = np.load('x_4.npy')
y4 = np.load('y_4.npy')
x5 = np.load('x_5.npy')
y5 = np.load('y_5.npy')
x6 = np.load('x_6.npy')
y6 = np.load('y_6.npy')
x7 = np.load('x_7.npy')
y7 = np.load('y_7.npy')

plt.imshow(data)
plt.plot(x0,y0,color='g',marker='.',ls='',markersize=1)
plt.plot(x1,y1,color='g',marker='.',ls='',markersize=1)
plt.plot(x2,y2,color='g',marker='.',ls='',markersize=1)
plt.plot(x3,y3,color='g',marker='.',ls='',markersize=1)
plt.plot(x4,y4,color='g',marker='.',ls='',markersize=1)
plt.plot(x5,y5,color='g',marker='.',ls='',markersize=1)
plt.plot(x6,y6,color='g',marker='.',ls='',markersize=1)
plt.plot(x7,y7,color='g',marker='.',ls='',markersize=1)
fx = 810
lx = 1024
# plt.plot(xdata[fx:lx],ydata[fx:lx],color='r',marker='.',ls='',markersize=1)
plt.show()




