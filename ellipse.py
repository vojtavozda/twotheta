import numpy as np
from numpy import pi, sin, cos
import cv2


def get_ellipse_pts(params:list, npts:int=100, tmin:float=0, tmax:float=2*np.pi):
    """
    Return npts points on the ellipse described by the
    'params = x0, y0, a, b, phi'
    for values of the parametric variable t between tmin and tmax (radians).

    Example:
    x, y = el.get_ellipse_pts((x0, y0, a, b, phi), npts, tmin, tmax)
    """

    x0, y0, a, b, phi = params
    # A grid of the parametric variable, t.
    t = np.linspace(tmin, tmax, npts)
    x = x0 + a * cos(t) * cos(phi) - b * sin(t) * sin(phi)
    y = y0 + a * cos(t) * sin(phi) + b * sin(t) * cos(phi)
    return x, y


def get_sum_of_squares(x:np.ndarray,y:np.ndarray,params:list) -> float:
    """
    Return sum os squares of the points (x_i,y_i) from the ellipse described by
    the 'params = x0, y0, a, b, phi'.
    """

    c = pol_to_cart(params)

    D = np.vstack([x**2, x*y, y**2, x, y, np.ones(len(x))]).T
    N = D @ c
    return np.sum(N**2)

def fit_ellipse_sos(p:list,x:np.ndarray,y:np.ndarray) -> float:
    """
    To use as objective function when fitting ellipse.

    Parameters are p = (x0,y0,a,b,phi)

    Used for fitting using 'scipy.optimize.minimize' as follows:
    ```
    x,y = get_ellipse([100,100,90,40,0])
    ansatz = [0,0,0,0,0]
    bounds = ((None,None),(None,None),(None,None),(None,None),(None,None))
    res = optimize.minimize(fit_ellipse_sos,ansatz,args=(x,y),bounds=bounds)
    print("Fit parameters:",res.x)
    ```
    """
    return get_sum_of_squares(x,y,p)

def fit_ellipse(x, y):
    """

    Fit the coefficients a,b,c,d,e,f, representing an ellipse described by
    the formula F(x,y) = ax^2 + bxy + cy^2 + dx + ey + f = 0 to the provided
    arrays of data points x=[x1, x2, ..., xn] and y=[y1, y2, ..., yn].

    Based on the algorithm of Halir and Flusser, "Numerically stable direct
    least squares fitting of ellipses".

    For code description see:
    https://scipython.com/blog/direct-linear-least-squares-fitting-of-an-ellipse
    and https://autotrace.sourceforge.net/WSCG98.pdf

    Example:
    --------
    ```
    import numpy as np
    import matplotlib.pyplot as plt
    import ellipse as el

    # Get some points on the ellipse
    npts = 20
    x0, y0, a, b, phi = (100,100,80,40,np.pi/3)
    x, y = el.get_ellipse_pts((x0, y0, a, b, phi), npts, 0, np.pi/2)
    noise = 0.5
    x += noise * np.random.normal(size=npts) 
    y += noise * np.random.normal(size=npts)

    coeffs = el.fit_ellipse(x, y)
    print('Exact parameters:')
    print('x0, y0, ap, bp, phi =', x0, y0, a, b, phi)
    print('Fitted parameters:')
    print('a, b, c, d, e, f =', coeffs)
    x0, y0, ap, bp, phi = el.cart_to_pol(coeffs)
    print('x0, y0, ap, bp, phi = ', x0, y0, ap, bp, phi)

    plt.plot(x, y, '.')     # given points
    x, y = el.get_ellipse_pts((x0, y0, ap, bp, phi))
    plt.plot(x, y)
    ax = plt.gca()
    plt.show()
    ```
    """

    D1 = np.vstack([x**2, x*y, y**2]).T
    D2 = np.vstack([x, y, np.ones(len(x))]).T
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2
    T = -np.linalg.inv(S3) @ S2.T
    M = S1 + S2 @ T
    C = np.array(((0, 0, 2), (0, -1, 0), (2, 0, 0)), dtype=float)
    M = np.linalg.inv(C) @ M
    eigval, eigvec = np.linalg.eig(M)
    con = 4 * eigvec[0]* eigvec[2] - eigvec[1]**2
    ak = eigvec[:, np.nonzero(con > 0)[0]]
    return np.concatenate((ak, T @ ak)).ravel()


def mask_sum(data:np.ndarray,cx:float,cy:float,r:float,axr:float,phi:float,
    width:int=1) -> float:
    """
    Create circular mask with cx,cy,r and calculate sum of `data` below the
    mask.
    :param data: Image data (2D array)
    :param cx: Ellipse center (x-coordinate)
    :param cy: Ellipse center (y-coordinate)
    :param r: Half of ellipse width (a = 2*r)
    :param axr: Axis ratio (b = 2*r*a)
    :param phi: Angle of the major axis (in radians)
    :param width: Width of mask circle (in pixels)
    """
    
    mask = data.copy()*0
    cv2.ellipse(mask,(int(cx),int(cy)),(int(r),int(r*axr)),phi*180/pi,0,360,1,width)
    overlap = mask*data
    if np.sum(mask) == 0:
        return 0
    else:
        return np.sum(overlap)/np.sum(mask)

def pol_to_cart(params:list):
    """
    Convert the ellipse params = x0, y0, a, b, phi to cartesian coefficients
    (a,b,c,d,e,f), where F(x,y) = ax^2 + bxy + cy^2 + dx + ey + f = 0.

    Coefficients are provided here: https://math.stackexchange.com/a/2989928
    But they can be easily calculated from the ellipse equation as written here:
    https://math.stackexchange.com/a/434482, multiplying all squares and find
    coefficients for x**2, y**2, ...

    """

    x0, y0, a, b, phi = params
    cP = cos(phi)
    sP = sin(phi)
    s2P = sin(2*phi)
    
    c = np.zeros(6)

    if a==0 or b==0:
        return c

    c[0] = cP**2/a**2 + sP**2/b**2
    c[1] = s2P/a**2 - s2P/b**2
    c[2] = sP**2/a**2 + cP**2/b**2
    c[3] = -2*x0*cP**2/a**2 - y0*s2P/a**2 - 2*x0*sP**2/b**2 + y0*s2P/b**2
    c[4] = -x0*s2P/a**2 - 2*y0*sP**2/a**2 + x0*s2P/b**2 - 2*y0*cP**2/b**2
    c[5] = (x0**2*cP**2/a**2 + x0*y0*s2P/a**2 + y0**2*sP**2/a**2 + 
            x0**2*sP**2/b**2 - x0*y0*s2P/b**2 + y0**2*cP**2/b**2 - 1)

    return c

def cart_to_pol(coeffs:list):
    """
    Convert the cartesian conic coefficients, (a, b, c, d, e, f), to the
    ellipse parameters, where F(x, y) = ax^2 + bxy + cy^2 + dx + ey + f = 0.
    The returned parameters are x0, y0, ap, bp, phi, where (x0, y0) is the
    ellipse centre; (ap, bp) are the semi-major and semi-minor axes,
    respectively; e is the eccentricity; and phi is the rotation of the semi-
    major axis from the x-axis.
    """

    # We use the formulas from https://mathworld.wolfram.com/Ellipse.html
    # which assumes a cartesian form ax^2 + 2bxy + cy^2 + 2dx + 2fy + g = 0.
    # Therefore, rename and scale b, d and f appropriately.
    a = coeffs[0]
    b = coeffs[1] / 2
    c = coeffs[2]
    d = coeffs[3] / 2
    f = coeffs[4] / 2
    g = coeffs[5]

    den = b**2 - a*c
    if den > 0:
        raise ValueError('coeffs do not represent an ellipse: b^2 - 4ac must'
                         ' be negative!')

    # The location of the ellipse centre.
    x0, y0 = (c*d - b*f) / den, (a*f - b*d) / den

    num = 2 * (a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g)
    fac = np.sqrt((a - c)**2 + 4*b**2)
    # The semi-major and semi-minor axis lengths (these are not sorted).
    ap = np.sqrt(num / den / (fac - a - c))
    bp = np.sqrt(num / den / (-fac - a - c))

    # Sort the semi-major and semi-minor axis lengths but keep track of
    # the original relative magnitudes of width and height.
    width_gt_height = True
    if ap < bp:
        width_gt_height = False
        ap, bp = bp, ap

    # The angle of anticlockwise rotation of the major-axis from x-axis.
    if b == 0:
        phi = 0 if a < c else np.pi/2
    else:
        phi = np.arctan((2.*b) / (a - c)) / 2
        if a > c:
            phi += np.pi/2
    if not width_gt_height:
        # Ensure that phi is the angle to rotate to the semi-major axis.
        phi += np.pi/2
    phi = phi % np.pi

    return x0, y0, ap, bp, phi

