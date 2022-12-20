"""
General library
===============

Mixture of general python functions
"""

import os
from typing import Tuple
from posixpath import dirname
import numpy as np
import csv
import shutil
import cv2

from matplotlib import pyplot as plt

# List of colors for plotting (adapted from matlab)
plt_clrs = ['#0072BD','#D95319','#EDB120','#7E2F8E','#77AC30','#4DBEEE','#A2142F']
plt_clrs = plt.rcParams['axes.prop_cycle'].by_key()['color']

def plot_plt_colors():

    fig = plt.figure()
    for i,plt_clr in enumerate(plt_clrs):
        plt.plot([0,1],[i,i],c=plt_clr,lw=10)
    plt.show()

def adjust_lightness(color, amount=1):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

def query_yes_no(question:str,default='yes') -> bool:
    """ Ask question with Yes/No choice to answer. Return appropriate bool. """

    valid = {'yes':True,'y':True,'ye':True,'no':False,'n':False}

    if default is None:
        prompt = ' [y/n] '
    elif default == 'yes':
        prompt = ' [Y/n] '
    elif default == 'no':
        prompt = ' [y/N] '
    else:
        raise ValueError(f"Invalid default answer: '{default}'!")

    while True:
        choice = input(question + prompt).lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            print("Please respond with 'yes'/'y' or 'no'/'n'.")


def sec2str(seconds:int,add_sign=False) -> str:

    """ Convert seconds to string 'HH:MM:SS' """

    if np.isnan(seconds): return np.nan
    
    if seconds >= 0:
        if add_sign:
            sign = '+'
        else:
            sign = ''
    else:
        sign = '-'
    seconds = abs(seconds)

    m, s = divmod(seconds,60)
    h, m = divmod(m,60)
    return sign + '%02d:%02d:%02d'%(h,m,s)

def str2sec(time_str:str) -> int:
    
    """ Convert string '+/-HH:MM:SS' to seconds """

    if time_str == '': return np.nan

    try:
        time_str = time_str.replace('+','')
        sign = 1 if '-' not in time_str else -1
        time_str = time_str.replace('-','')
        h,m,s = time_str.split(':')
        return sign*(int(h)*3600 + int(m)*60 + int(s))
    except:
        return np.nan


def append_csv_row(ofile:str,row:list) -> None:
    """ Append row to csv file (open --> write --> close) """
    csvfile = open(ofile,'a')
    csvWriter = csv.writer(csvfile,delimiter=',')
    csvWriter.writerow(row)
    csvfile.close()

def dice(X, Y) -> float:
    """ Calculate dice coeficient (F1-score). """

    intersection = np.sum(np.multiply(X, Y))
    
    union = np.sum(X) + np.sum(Y)

    return np.mean((2*intersection + 0.1)/(union + 0.1))


def flatten_list(nested_list:list) -> list:
    """ Convert list of nested lists into simple flatten list. 
    Example:
    [0,[1,2,[3,4],5],6,7] -> [0,1,2,3,4,5,6,7]
    """
    flat_list = []
    # Iterate through the outer list
    for element in nested_list:
        if type(element) is list:
            # If the element is of type list, iterate through the sublist
            for item in element:
                if type(item) is list:
                    # If item is list as well, call recursion
                    flat_list = flat_list+flatten_list(item)
                else:
                    # If not, append item
                    flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list

def isfloat(value:str) -> bool:
    """ Returns true if value is convertable to float, otherwise false. """
    try:
        float(value)
        return True
    except ValueError:
        return False

def numberOfLinesInFile(fname:str) -> int:

    with open(fname) as f:
        i = 0
        for i, l in enumerate(f):
            pass
    return i + 1

def mkdir(dir_path,overwrite=True,num=0,rm=False,f=False):
    """
    Create directory `dir_path` without complaining whether it exists or not.
    When overwrite=False, `_X` is appended to the end of dir path.
    `X` is smallest positive integer for which given dir does not exist.
    If needed, set `rm` to True so dir is deleted first if exists.
    f: Force - use rm without asking
    """

    if rm and os.path.exists(dir_path) and os.path.isdir(dir_path):
        cont = False
        if f:
            cont = True
        else:
            cont = query_yes_no(clrprint.yellow(f"WARNING: Existing dir '{dir_path}' is going to be completely removed! Agree?"))
        if cont:
            shutil.rmtree(dir_path)
        else:
            overwrite = query_yes_no(clrprint.yellow("Do you want to merge dirs?"))
            if not overwrite:
                print(clrprint.yellow("New dir with serie number append was created."))

    end = ''
    if num>0: end = '_'+str(num)
    dir_path_num = dir_path + end
    if not os.path.isdir(dir_path_num):
        os.makedirs(dir_path_num)
        return dir_path_num
    elif overwrite:
        return dir_path_num
    else:
        return mkdir(dir_path,overwrite,num+1)


def getPaths(dir:str,ext:str=None) -> list:
    """ Returns list of sorted filenames with provided extension in given
    directory.
    Args:
        dir: Directory with files
        ext: Extension of search files (with or without dot). Extension is
        automatically changed to the most common extension among files in
        provided directory if not specified.
    Returns:
        Sorted list of filenames
    """
    paths = []
    if not os.path.isdir(dir):
        if not os.path.isfile(dir):
            print(clrprint.red(f"'{dir}' is not file nor directory!"))
            return
        else:
            return [dir]

    if ext is None:
        # Get the most common extension
        exts = np.array([os.path.splitext(filename)[1] for filename in sorted(os.listdir(dir))])
        exts = exts[exts!='']
        u, counts = np.unique(exts,return_counts=True)
        ext = u[np.argmax(counts)]
    
    ext = ext.replace('.','')

    for filename in sorted(os.listdir(dir)):
        if filename.endswith(ext):
            paths.append(os.path.join(dir,filename))
    return paths


def printAttributes(obj,values=True) -> None:
    """ Print attributes and methods (returns) of an object
    
    Args:
        obj: Arbitrary object.
    """

    attributes = dir(obj)
    for atrbt in attributes:
        if atrbt[0]!='_':
            try:
                if values:
                    print(bcolors.BOLD,atrbt+"():",bcolors.ENDC,eval('obj.'+atrbt+'()'))
                else:
                    print(bcolors.BOLD,atrbt+"(): ...")
            except:
                if values:
                    print(bcolors.BOLD,atrbt+":",bcolors.ENDC,eval('obj.'+atrbt))
                else:
                    print(bcolors.BOLD,atrbt+": ...")

def find_nearest_val(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def find_nearest_idx(arr,val) -> int:
    """ Returns index of value from array which is closest to `val`."""
    return (np.abs(arr-val)).argmin()

def getPlaneABC(p1,p2,p3) -> Tuple[float,float,float]:
    """
    Calculate plane parameters from three points.
    Plane is defined as: z(x,y) = a*x + b*y + c.
    points = [[   p1   ],[   p2   ],[   p3   ]]
    points = [[x1,y1,z1],[x2,y2,z2],[x3,y3,z3]]
    Example:
        `getPlaneABC(np.array([1,0,0]),np.array([0,1,0]),np.array([0,0,1]))`
    """
    
    u = p2 - p1         # First vector in the plane
    v = p3 - p1         # Second vector in the plane
    
    n = np.cross(u,v)   # Orthogonal vector to the plane

    if n[2]==0: raise Exception(f"Points cannot be in one line! u={u}")

    #  = n.(u-v) = (n1,n2,n3).((u1,u2,u3)-(v1,v2,v3))  ...and so on

    a = -n[0]/n[2]
    b = -n[1]/n[2]
    c = (n[0]*p1[0]+n[1]*p1[1]+n[2]*p1[2]) / n[2]

    return a, b, c

def mean(x):
    """ Calculate mean, standard deviation and standard error"""
    return (np.mean(x),np.std(x),np.std(x)/np.sqrt(len(x)))

def wmean(x,w) -> np.ndarray:
    """ Calculate weighted mean and weighted standard deviation """

    E_xw = np.sum(x*w)
    E_w = np.sum(w)

    if E_w == 0:
        return np.array([np.mean(x),0])

    x_w = E_xw/E_w

    E_wxx = np.sum(w*(x-x_w)**2)

    wstd = np.sqrt(E_wxx/E_w)

    return np.array([x_w,wstd])

def scale(data:np.ndarray,min_=0,max_=1):
    """ Scale data from `min_` to `max_` """

    data = (data-np.min(data))/(np.max(data)-np.min(data))
    data = min_ + data*(max_-min_)
    return data

def scatter2bin(xdata:np.ndarray,ydata:np.ndarray,N:int=20):
    """
    Convert scatter data into equally wide bins.
    Value in each bin equals to mean value of all data which x-coordinate fits
    into interval (x0,x0+dx), where dx = (xdata[0]-xdata[-1])/N

    Args:
        xdata:
        ydata:
        N: Required number of bins

    Returns:
    xbin - x-coordinates of bins
    ybin - value of each bin
    nbin - number of data in given bin

    Example:
    x_bin,y_bin,_ = scatter2bin(xdata,ydata)
    plt.plot(x_bin,y_bin)
    """

    # Prepare data: sort
    order = xdata.argsort()
    xdata = xdata[order]
    ydata = ydata[order]

    # Define empty arrays
    xbin = np.linspace(0,xdata[-1],N)   # x-coordinates of bins
    ybin = np.zeros(N)                  # boxed ydata
    nbin = np.zeros(N)                  # number of ydata in each bin
    i = 0                               # auxiliary bin counter
    n = 0                               # counter of values in bin

    for j in range(len(xdata)):
        if xdata[j] > xbin[i+1]:
            ybin[i] = ybin[i]/n if n>0 else 0
            nbin[i] = n
            i += 1
            n = 0
        ybin[i] += ydata[j]
        n += 1

    return (xbin,ybin,nbin)

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly suited for
    smoothing noisy data. The main idea behind this approach is to make for each
    point a least-square fit with a polynomial of high order over a odd-sized
    window centered at the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')