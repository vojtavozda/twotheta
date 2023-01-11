import numpy as np
from numpy import pi, sin, cos, tan, sqrt
from numpy.linalg import norm
import cv2

def fetch_params(params):
    """
    Convert parameters (list or dict) into tuple of parameters x0,y0,a,b,phi
    """

    if isinstance(params,dict):
        x0 = params['x0']
        y0 = params['y0']
        a = params['a']
        b = params['b']
        phi = params['phi']
    elif isinstance(params,list):
        x0, y0, a, b, phi = params

    return (x0,y0,a,b,phi)

def get_ellipse_pts(params, npts:int=100, tmin:float=0, tmax:float=2*np.pi):
    """
    Return npts points on the ellipse described by the
    'params = x0, y0, a, b, phi'
    for values of the parametric variable t between tmin and tmax (radians).
    Params can be also dictionary with keys 'x0','y0','a','b','phi'.

    Example:
    x, y = el.get_ellipse_pts((x0, y0, a, b, phi), npts, tmin, tmax)
    """

    x0, y0, a, b, phi = fetch_params(params)

    # A grid of the parametric variable, t.
    t = np.linspace(tmin, tmax, npts)
    x = x0 + a * cos(t) * cos(phi) - b * sin(t) * sin(phi)
    y = y0 + a * cos(t) * sin(phi) + b * sin(t) * cos(phi)
    return x, y


def get_sum_of_squares(x:np.ndarray,y:np.ndarray,params) -> float:
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



def get_ellipse_from_cone(z0,n,two_theta):
    """
    Returns parameters of ellipse which is created at intersection of a cone and
    plane.
    The cone with apex at (0,0,0), axis identical with z-axis and half apex
    angle `two_theta` is intersected by the plane defined by a normal vector `n`
    and point (0,0,z0).
    """
    
    """
    Dandelin spheres:
    -----------------
    This procedure is based on Dandelin spheres: These are two speres touching a
    cone (on a circle) and the plane with normal vector n (one point). Center of
    spere is located at z_D. Distance from z_D to the plane is R which is also
    the radius of the spere. Coordinates of the point where the spere touches
    the plane are R*n. This is one focus of the ellipse.
    """
    # First focus from closer Dandeline sphere
    z_D1 = n[2]*z0/(n[2]+sin(two_theta))
    r_D1 = z_D1*sin(two_theta)
    f1 = np.array([0,0,z_D1])+r_D1*n

    # Second focus from further Dandeline sphere
    z_D2 = -n[2]*z0/(-n[2]+sin(two_theta))
    r_D2= z_D2*sin(two_theta)
    f2 = np.array([0,0,z_D2])-r_D2*n

    """
    How to find semi-major axis:
    ----------------------------
    So now we have positions of both foci and we need to calculate lengths
    of semi-major and semi-minor axis. Semi-minor axis is not `b=z0*tg(2t)`
    because center of the ellipse is shifted from cone axis. However, from two
    foci we can calculate parameters of a line identical with major axis. Then
    we find intersection of the line and cone which is defined as
    x^2+y^2=tg(2t)^2*z^2. Distance from this point to the ellipse center is
    semi-major axis.
    """

    if f1[2]==f2[2]:
        # Degenerate ellipse (circle)
        P1 = np.array([sqrt(tan(two_theta)**2*z0**2),0,z0])
        
    else:

        # Find lower point where major axis intersects the cone
        c = (f2[2]-f1[2])/(f2[0]-f1[0]) if f2[0]!=f1[0] else None
        d = f1[2]-f1[0]*c if c is not None else None
        e = (f2[2]-f1[2])/(f2[1]-f1[1]) if f2[1]!=f1[1] else None
        f = f1[2]-f1[1]*e if e is not None else None

        a2 = tan(two_theta)**2
        b2 = 0
        c2 = 0
        if c is not None:
            a2 += -1/c**2
            b2 += 2*d/c**2
            c2 += -d**2/c**2
        if e is not None:
            a2 += -1/e**2
            b2 += 2*f/e**2
            c2 += -f**2/e**2

        z1 = (-b2 + sqrt(b2**2 - 4*a2*c2))/(2*a2)
        z2 = (-b2 - sqrt(b2**2 - 4*a2*c2))/(2*a2)
        z = np.min((z1,z2))
        x = z/c-d/c if c is not None else 0
        y = z/e-f/e if e is not None else 0
        P1 = np.array([x,y,z])

    # Calculate parameters of an ellipse at plane and cone intersection
    f = (norm(f2-f1))/2             # distance of foci from the center
    c = (f2+f1)/2                   # ellipse center
    phi = np.angle(n[0]+n[1]*1j)    # angle in xy projection

    a = norm(c-P1)
    b = sqrt(a**2-f**2)

    """
    Projection of semi-major axis into xy plane:
    --------------------------------------------
    Finally, for our purposes we have to calculate projection of semi-major axis
    into xy plane as we find points of these projected ellipse, then calculate
    z-coordinates and semi-major axis of the final ellipse is the correct value.
    """

    if f1[2]!=f2[2]:
        # Calculate angle between semi-major axis and xy plane
        a_vec = f2-f1
        angle_a = np.angle(np.sqrt(a_vec[0]**2+a_vec[1]**2)+a_vec[2]*1j)
        # Calculate length of projected semi-major axis
        a = a*cos(angle_a)

    params = {
        'x0': c[0], 'y0': c[1], 'a': a, 'b': b, 'phi': phi,
        'z_D1': z_D1, 'r_D1': r_D1, 'f1': f1,
        'z_D2': z_D2, 'r_D2': r_D2, 'f2': f2,
        'P1':P1,
    }

    return params

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

    x0, y0, a, b, phi = fetch_params(params)

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

