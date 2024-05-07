"""
Script for extracting points from 2D Jungfrau data.
"""

# %%
import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
import extra_data


def findLine(data:np.ndarray,y0:int,direction:int=1) -> np.ndarray:

    x_range = 10
    y_range = 15

    xdata = np.arange(0,data.shape[1])
    ydata = xdata*0

    if direction < 0: xdata = np.flip(xdata)

    # Itereate over the columns of the data
    for x0 in xdata:

        # 1: Select y-range
        y1 = int(y0-y_range/2)
        y2 = int(y0+y_range/2)
        if y1<0:
            y2 -= y1
            y1 = 0
        elif y2>data.shape[0]:
            y1 -= y2-data.shape[0]
            y2 = data.shape[0]

        ysel = np.arange(y1,y2).astype(int)

        # Method 1: Maximum of 1 column
        # data_ = data[ysel,x0]

        x1 = x0-x_range/2
        x2 = x0+x_range/2
        x1 = int(max(0,x1))
        x2 = int(min(data.shape[1],x2))

        # Method 2: Mean of x_range
        # data_ = data[ysel,x1:x2].mean(axis=1)

        # Method 3: Gaussian weighted mean of x_range
        gw = 10
        weights = np.exp(-((np.arange(x1,x2)-x0)**2)/gw)
        data_G = data[ysel,x0]*0
        for i in range(x1,x2):
            data_G += data[ysel,i]*weights[i-x1]
        data_ = data_G/weights.sum()

        counter = 1
        for i in np.arange(0,0):
            if x0+i > 0 and x0+i < data.shape[1]:
                data_ += data[ysel,x0+i]
                counter += 1
        data_ = data_/counter
        y0 = ysel[np.argmax(data_)]
        peaks,props = find_peaks(data_,distance=y_range,width=[0,30])
        small_prominence = False
        if len(peaks) > 0:
            # Prominence is calculated relative to signal
            # If prominence is smaller than 5% of signal, it is considered small
            small_prominence = props['prominences'][0] < (np.max(data_)-np.min(data_))/20
        if ((len(peaks) == 0) or (y0!=peaks[0]+y1 and small_prominence)):
            # If there is no peak found or the peak is not at maximum and the
            # prominence is small, break the loop (conic out of detector)

            # plt.plot(ysel,data_)
            # plt.plot(peaks+y1,data_[peaks],'x')
            # plt.show()
            # print(props)
            if direction > 0:
                xdata = xdata[:x0]
                ydata = ydata[:x0]
            else:
                xdata = xdata[:len(xdata)-x0-1]
                ydata = ydata[x0+1:]
            break
        ydata[x0] = peaks[0]+y1
        # ydata[x0] = y0

    conic = np.vstack((xdata,ydata))
    if direction < 0: conic[0,:] = np.flip(conic[0,:]) 
    return conic



data_path = "/home/vovo/FZU/experimenty/240920_HED_#6869/twotheta/p2838/JF3_run[135]_train[all]_bkg.npy"
data:np.ndarray = np.load(data_path)
# data:np.ndarray = np.load(os.path.join('p2838','JF3_run[135]_train[all].npy'))
export_dir = "/home/vovo/FZU/experimenty/240920_HED_#6869/twotheta/data"

data[data>30] = 30
data[data<0] = 0

# Create mask for data of ones
mask = data*0+1
mask[255:257,:] = 0

# Apply mask
data = data*mask


peaks2, _ = find_peaks(data[:,data.shape[1]-1], distance=5,prominence=2,width=[0,30])
# plt.plot(data[:,0])
# plt.plot(data[:,data.shape[1]-1])
# plt.plot(peaks1, data[peaks1,0], "x")
# plt.plot(peaks2, data[peaks2,data.shape[1]-1], "x")
# plt.show()

# First, look at first column of data, identify peaks and find conics in forward
# direction 
conics_forward = []
peaks1, _ = find_peaks(data[:,0], distance=5,prominence=2,width=[0,30])
for peak in peaks1:
    conic = findLine(data, peak)
    conics_forward.append(conic)
    print(f"Length of conic: {len(conic[0])}")
print(f"Found {len(conics_forward)} conics in forward direction.")

# Second, look at last column of data, identify peaks and find conics in
# backward direction
conics_backward = []
for peak in peaks2:
    conic = findLine(data, peak,-1)
    conics_backward.append(conic)
    print(f"Length of conic: {len(conic[0])}")
print(f"Found {len(conics_backward)} conics in backward direction.")

# Remove conics that are same or very similar with tolerance
conics = []
for conic in conics_forward+conics_backward:
    if len(conics) == 0:
        conics.append(conic)
    else:
        for conic_ in conics:
            # Append if shapes are different
            if conic.shape != conic_.shape:
                conics.append(conic)
                break
            # Break if conic is similar to any other conic
            if np.allclose(conic,conic_,atol=5):
                break
        else:
            conics.append(conic)
            continue
print(f"{len(conics)} conics identified as unique.")

plt.imshow(data)
for i,conic in enumerate(conics):
    plt.plot(conic[0],conic[1],color='r')
    np.save(os.path.join(export_dir,f"conic_{i}.npy"),conic)
plt.show()



