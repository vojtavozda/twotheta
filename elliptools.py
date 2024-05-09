import numpy as np
from numpy import pi, sin, cos, tan, sqrt
from numpy.linalg import norm
from matplotlib import pyplot as plt
from genlib import plt_clrs
from clrprint import printc
from numpy import linalg as LA
import cv2

# =============================[ GENERAL GEOMETRY ]=============================

def rotate_point(x,y,x0,y0,phi):
    """
    Rotate point (x,y) around center (x0,y0) by phi [radians].
    Returns new coordinates (x_new,y_new).
    """

    x -= x0
    y -= y0

    x_new = x*cos(phi) - y*sin(phi)
    y_new = x*sin(phi) + y*cos(phi)

    x_new += x0
    y_new += y0

    return x_new, y_new

def rotate3D(data:np.ndarray,phi:float,theta:float,psi:float):
    """
    Rotate vector in 3D around origin. Angles are Euler angles:
    phi:    rotate around the `z` axis
    theta:  rotate around the `x'` axis
    psi:    rotate around the `z''` axis

    For more details check:
    https://mathworld.wolfram.com/EulerAngles.html
    """

    R_phi = np.array((
        ( cos(phi), sin(phi), 0),
        (-sin(phi), cos(phi), 0),
        (        0,        0, 1)
    ))
    R_theta = np.array((
        (1,           0,          0),
        (0,  cos(theta), sin(theta)),
        (0, -sin(theta), cos(theta))
    ))
    R_psi = np.array((
        ( cos(psi), sin(psi), 0),
        (-sin(psi), cos(psi), 0),
        (        0,        0, 1)
    ))

    R = R_psi@(R_theta@R_phi)

    return R@data

# ===================================[ CONE ]===================================
class Cone:

    def __init__(self,apex:np.ndarray,axis:np.ndarray,theta:float):
        self.apex = apex
        self.n = axis
        self.theta = theta


    def getMesh(self, length:float) -> tuple:
        """
        Generate the coordinates of a cone in 3D space.

        Parameters:
        apex (array): The coordinates of the apex of the cone.
        n (array): Unit vector axis of the cone.
        angle (float): The angle of the cone in radians.
        length (float): The length of the cone.

        Returns:
        tuple: A tuple containing three arrays (X, Y, Z) representing the
        coordinates of the cone in 3D space.
        """
        
        # Generate the coordinates of the cone
        a = np.linspace(0, 2*np.pi, 20)
        r = np.linspace(0, 1, 10)
        T, R = np.meshgrid(a, r)
        X = R * cos(T) * tan(self.theta) * length
        Y = R * sin(T) * tan(self.theta) * length
        Z = R * length

        # Rotate the cone to the desired orientation
        z = np.array([0, 0, 1])
        v = np.cross(z, self.n)
        s = norm(v)
        c = np.dot(z, self.n)
        v_x = np.array([[0, -v[2], v[1]],
                        [v[2], 0, -v[0]],
                        [-v[1], v[0], 0]])
        if s == 0:
            R = np.eye(3) * np.sign(np.sum(self.n))
        else:
            R = np.eye(3) + v_x + np.dot(v_x, v_x) * (1 - c) / (s ** 2)

        X, Y, Z = np.dot(R, np.array([X.flatten(), Y.flatten(), Z.flatten()]))
        X = X.reshape(T.shape) + self.apex[0]
        Y = Y.reshape(T.shape) + self.apex[1]
        Z = Z.reshape(T.shape) + self.apex[2]

        return X, Y, Z


    def plotMesh(self,ax:plt.Axes,length:float):
        """
        Plot a cone in 3D space.
        
        Parameters:
        ax (plt.Axes): The axis to plot the cone on.
        apex (np.ndarray): The coordinates of the apex of the cone.
        n (np.ndarray): Unit vector axis of the cone.
        angle (float): The angle of the cone in radians.
        length (float): The length of the cone.
        """

        cone_X, cone_Y, cone_Z = self.getMesh(length)

        ax.plot(self.apex[0],self.apex[1],self.apex[2],'.',color='k',markersize=5)
        # ax.plot([self.apex[0],self.apex[0]+self.n[0]*1e6],
        #         [self.apex[1],self.apex[1]+self.n[1]*1e6],
        #         [self.apex[2],self.apex[2]+self.n[2]*1e6],
        #         'k',linestyle='--',linewidth=0.5)
        ax.plot_surface(cone_X,cone_Y,cone_Z,alpha=0.2,antialiased=True,color=plt_clrs[0])
        ax.plot_wireframe(cone_X,cone_Y,cone_Z,color=plt_clrs[0],linewidth=0.1)

    def plotWireframe(self,ax:plt.Axes):

        V = self.apex
        # Find coordinates of point where cone axis intersects plane z=0
        S = np.array([V[0]-V[2]*self.n[0]/self.n[2],
                      V[1]-V[2]*self.n[1]/self.n[2],0])

        ax.plot(V[0],V[1],V[2],'.',color='k',markersize=10)     # V  - cone apex
        ax.plot(S[0],S[1],S[2],'.',color='k',markersize=10)     # S  - cone axis
        ax.plot([V[0],S[0]],[V[1],S[1]],[V[2],S[2]],c='k',ls='--')

    def print(self,f:int=2):

        printc("Cone: ",fw='b',fc='g',end="")
        printc(f"Apex=[{self.apex[0]:.{f}f},{self.apex[1]:.{f}f},{self.apex[2]:.{f}f}]",fc='g',end=' ')
        printc(f"Axis=[{self.n[0]:.{f}f},{self.n[1]:.{f}f},{self.n[2]:.{f}f}]",fc='g',end=' ')
        printc(f"theta={self.theta:.{f}f} ({self.theta*180/pi:.{f}f})°",fc='g')

# =================================[ ELLIPSE ]==================================
class Ellipse:
    
    def __init__(self,
                 x0:float = None,
                 y0:float = None,
                 a:float = None,
                 b:float = None,
                 phi:float = None,
                 xData:np.ndarray = None,
                 yData:np.ndarray = None,
                 theta:float = None,
                 color:str = plt_clrs[0]
                 ):
        self.x0 = x0
        self.y0 = y0
        self.a = a
        self.b = b
        self.phi = phi
        self.cone = None
        self.xData = xData
        self.yData = yData
        self.theta = theta
        if self.a is not None and self.b is not None:
            self.c = sqrt(a**2-b**2)

        self.clr = color

    def _get_delta(self,z0,two_theta):
        """
        Solve 'a=(|SA|+|SB|)/2' for delta:
        """
        sTT = sin(two_theta)
        cTT = cos(two_theta)

        det = z0**2*sTT**2*cTT**2-4*self.a**2*(cTT**2-1)

        cos_deltaP = (z0*sTT*cTT + sqrt(det)) / (2*self.a)
        cos_deltaM = (z0*sTT*cTT - sqrt(det)) / (2*self.a)

        deltaP = np.arccos(cos_deltaP)
        deltaM = np.arccos(cos_deltaM)

        print(f"Found delta+ = {deltaP*180/pi:.0f}° and delta- = {deltaM*180/pi:.0f}°")
        return deltaP

    def _get_semi_major(self):
        """ Calculate coordinates of rotated major axis """
        Am = [self.x0-self.a,self.y0,0]
        Am[0],Am[1] = rotate_point(Am[0],Am[1],self.x0,self.y0,self.phi)
        Ap = [self.x0+self.a,self.y0,0]
        Ap[0],Ap[1] = rotate_point(Ap[0],Ap[1],self.x0,self.y0,self.phi)
        return Am, Ap

    def _get_semi_minor(self):
        """ Calculate coordinates of rotated minor axis """
        Bm = [self.x0,self.y0-self.b,0]
        Bm[0],Bm[1] = rotate_point(Bm[0],Bm[1],self.x0,self.y0,self.phi)
        Bp = [self.x0,self.y0+self.b,0]
        Bp[0],Bp[1] = rotate_point(Bp[0],Bp[1],self.x0,self.y0,self.phi)
        return Bm, Bp

    def setPlotColor(self,clr:str):
        self.clr = clr

    def getPoints(self,npts:int=100,tmin:float=0,tmax:float=2*pi) -> tuple:
        """
        Return npts points on the ellipse described by the
        'params = x0, y0, a, b, phi'
        for values of the parametric variable t between tmin and tmax (radians).
        """

        # A grid of the parametric variable, t.
        t = np.linspace(tmin, tmax, npts)
        x = self.x0 + self.a * cos(t) * cos(self.phi) - self.b * sin(t) * sin(self.phi)
        y = self.y0 + self.a * cos(t) * sin(self.phi) + self.b * sin(t) * cos(self.phi)
        return x, y


    def fit(self):
        """

        Fit the coefficients a,b,c,d,e,f, representing an ellipse described by
        the formula F(x,y) = ax^2 + bxy + cy^2 + dx + ey + f = 0 to the provided
        arrays of data points x=[x1, x2, ..., xn] and y=[y1, y2, ..., yn].

        Based on the algorithm of Halir and Flusser, "Numerically stable direct
        least squares fitting of ellipses".

        For code description see:
        https://scipython.com/blog/direct-linear-least-squares-fitting-of-an-ellipse
        and https://autotrace.sourceforge.net/WSCG98.pdf

        This is exactly the same algorithm as in the pyFAI library.
        
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

        x = self.xData
        y = self.yData
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
        params = cart_to_pol(np.concatenate((ak, T @ ak)).ravel())
        self.x0, self.y0, self.a, self.b, self.phi = params

    def findEllipseFromCone(self):

        n = self.cone.n
        assert np.linalg.norm(n) == 1, "Cone axis must be a unit vector!"

        phi = cos(n[0]/np.linalg.norm(n))
        n2 = rotate3D(n,-phi,0,0)
        n3 = rotate3D(n2,0,self.theta,0)
        n4 = rotate3D(n3,phi,0,0)
        return n4


    def getCone(self,theta:float) -> Cone:
        """
        Get a cone from the ellipse.
        """

        z0 = self.b/tan(theta)
        delta = self._get_delta(z0,theta)
        s = z0*sin(theta)/2*(1/cos(theta+delta)-1/(cos(theta-delta)))
        V = np.array([0,0,0]).astype(float)
        V[0] = (self.x0+s+z0*sin(delta))
        V[1] = self.y0
        V[2] = z0*cos(delta)

        # Rotate cone apex by phi around ellipse center
        V[0],V[1] = rotate_point(V[0],V[1],self.x0,self.y0,self.phi)
        # Calculate direction of cone axis
        n = np.array([-sin(delta),0,-cos(delta)])
        # Rotate this vector around origin (it is a vector)
        n[0],n[1] = rotate_point(n[0],n[1],0,0,self.phi)
        n = n/np.linalg.norm(n)

        return Cone(V,n,theta)

    def findCone(self,theta:float=None):
        theta = self.theta if theta is None else theta
        self.cone = self.getCone(theta)
        return self.cone

    def plotData(self,ax:plt.Axes,**kwargs):
        ax.plot(self.xData,self.yData,'.',color=self.clr,**kwargs)

    def plot(self,
             ax:plt.Axes,
             plotAxes:bool=False,
             plotData:bool=False,
             plotCone:bool=False
             ):

        x,y = self.getPoints()
        # Ellipse itself
        ax.plot(x,y,color=self.clr)

        if plotData and self.xData is not None and self.yData is not None:
            self.plotData(ax)

        if plotAxes:
            # Ellipse center
            ax.plot(self.x0,self.y0,0,'.',color=self.clr,markersize=10)
            Am, Ap = self._get_semi_major()
            Bm, Bp = self._get_semi_minor()
            # Semi-major axis
            ax.plot([Am[0],Ap[0]],[Am[1],Ap[1]],[Am[2],Ap[2]],c=self.clr,ls='--',lw=1)
            # Semi-minor axis
            ax.plot([Bm[0],Bp[0]],[Bm[1],Bp[1]],[Bm[2],Bp[2]],c=self.clr,ls='--',lw=1)
            # Foci
            ax.plot([self.x0+self.c*cos(self.phi),self.x0-self.c*cos(self.phi)],
                    [self.y0+self.c*sin(self.phi),self.y0-self.c*sin(self.phi)],
                    [0,0],'.',c=self.clr,markersize=5,lw=1)

        if plotCone:
            self.plotCone(ax)

    def plotCone(self,ax:plt.Axes):

        if self.cone is None:
            if self.theta is None:
                printc("Cone angle not defined!",tag='e')
                return
            else:
                self.findCone(self.theta)

        Am, Ap = self._get_semi_major()
        Bm, Bp = self._get_semi_minor()
        V = self.cone.apex
        n = self.cone.n
        z0 = self.b/tan(self.cone.theta)

        # Find coordinates of point where cone axis intersects plane z=0
        S = np.array([V[0]-V[2]*n[0]/n[2],
                      V[1]-V[2]*n[1]/n[2],0])

        # Calculate cone circle (https://math.stackexchange.com/a/73242)
        # `a` is vector perpendicular to `n` pointing towards `Ap` (from a.n=0)
        a = np.array([Ap[0],Ap[1],(-Ap[0]*n[0]-Ap[0]*n[1])/n[2]])
        a = a/LA.norm(a)
        # `b` is perpendicular to `a` and `n`
        b = np.cross(a,n)

        t = np.linspace(0,2*np.pi,50)   # Parameter of the circle
        c = 0                           # Multiplier (move circle along cone axis)
        r = (z0+c)*tan(self.cone.theta) # Radius of the circle
        cone_x = S[0]-n[0]*c + r*cos(t)*a[0] + r*sin(t)*b[0]
        cone_y = S[1]-n[1]*c + r*cos(t)*a[1] + r*sin(t)*b[1]
        cone_z = S[2]-n[2]*c + r*cos(t)*a[2] + r*sin(t)*b[2]

        ax.plot(cone_x,cone_y,cone_z,c=self.clr,lw=1)

        ax.plot(S[0],S[1],S[2],'.',color='k',markersize=10)     # S  - cone axis
        ax.plot([V[0],S[0]],[V[1],S[1]],[V[2],S[2]],c='k',ls='--')

        # Plot lines connecting apex with semi-major and semi-minor axes
        ax.plot([V[0],Ap[0]],[V[1],Ap[1]],[V[2],Ap[2]],c=self.clr,lw=1)
        ax.plot([V[0],Am[0]],[V[1],Am[1]],[V[2],Am[2]],c=self.clr,lw=1)
        ax.plot([V[0],Bp[0]],[V[1],Bp[1]],[V[2],Bp[2]],c=self.clr,lw=1)
        ax.plot([V[0],Bm[0]],[V[1],Bm[1]],[V[2],Bm[2]],c=self.clr,lw=1)

    def print(self):
        printc("Ellipse: ",fw='b',fc='b',end="")
        printc(f"x0={self.x0:.2f}, y0={self.y0:.2f}, a={self.a:.2f}, b={self.b:.2f}, phi={self.phi:.2f} ({(self.phi*180/pi):.2f})°",fc='b')

# ==================================[ OTHER ]===================================

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
    elif isinstance(params,list) or isinstance(params,tuple):
        x0, y0, a, b, phi = params

    return (x0,y0,a,b,phi)


def get_sum_of_squares(x:np.ndarray,y:np.ndarray,params) -> float:
    """
    Return sum os squares of the points (x_i,y_i) from the ellipse described by
    the 'params = x0, y0, a, b, phi'.
    TODO: Sometimes it returns 1e66?!
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

    # Calculate angle between semi-major axis and xy plane
    a_vec = f2-f1
    angle_a = abs(np.angle(np.sqrt(a_vec[0]**2+a_vec[1]**2)+a_vec[2]*1j))
    # Check wheter cone section is an ellipse or parabola/hyperbola
    if (two_theta > pi/2-angle_a):
        print("Not an ellipse!")

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
        # Calculate length of projected semi-major axis
        a = a*cos(angle_a)

    params = {
        'x0': c[0], 'y0': c[1], 'a': a, 'b': b, 'phi': phi,
        'z_D1': z_D1, 'r_D1': r_D1, 'f1': f1,
        'z_D2': z_D2, 'r_D2': r_D2, 'f2': f2,
        'P1':P1,
    }

    return params


def mask_sum_old(data:np.ndarray,cx:float,cy:float,r:float,axr:float,phi:float,
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

def mask_sum(data:np.ndarray,params,width:int=1) -> float:
    """
    Create elliptic mask with x0,y0,a,b and calculate sum of `data` below the
    mask.
    """

    # Check no of params is nan or inf
    if np.isnan(params).any() or np.isinf(params).any():
        print("Ellipse: NaN or Inf in parameters!")
        return 0
    
    x0, y0, a, b, phi = fetch_params(params)

    mask = data.copy()*0
    cv2.ellipse(mask,(int(x0),int(y0)),(int(a),int(b)),phi*180/pi,0,360,1,width)
    overlap = mask*data
    overlap[overlap is None] = 0
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


def getSphere(x0, y0, z0, radius):
    """
    Generate the coordinates of a sphere centered at (x0, y0, z0) with the given
    radius.

    Parameters:
    x0 (float): x-coordinate of the center of the sphere.
    y0 (float): y-coordinate of the center of the sphere.
    z0 (float): z-coordinate of the center of the sphere.
    radius (float): radius of the sphere.

    Returns:
    X (ndarray): x-coordinates of the points on the sphere.
    Y (ndarray): y-coordinates of the points on the sphere.
    Z (ndarray): z-coordinates of the points on the sphere.
    """

    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    X = x0+cos(u)*sin(v)*radius
    Y = y0+sin(u)*sin(v)*radius
    Z = z0+cos(v)*radius
    return X,Y,Z

def plotSpehre(ax:plt.Axes,x0:float,y0:float,z0:float,radius:float):
    """
    Plot a sphere in 3D space.
    
    Parameters:
    ax (plt.Axes): The axis to plot the sphere on.
    x0 (float): x-coordinate of the center of the sphere.
    y0 (float): y-coordinate of the center of the sphere.
    z0 (float): z-coordinate of the center of the sphere.
    radius (float): radius of the sphere.
    """

    sphere_X, sphere_Y, sphere_Z = getSphere(x0,y0,z0,radius)
    ax.plot(x0,y0,z0,'.',color='k',markersize=5)
    ax.plot_surface(sphere_X,sphere_Y,sphere_Z,alpha=0.2,antialiased=True,color=plt_clrs[1])
    ax.plot_wireframe(sphere_X,sphere_Y,sphere_Z,color=plt_clrs[1],linewidth=0.1)


# ==============================[ DEPRECATED ]==================================

def get_ellipse_pts(params, npts:int=100, tmin:float=0, tmax:float=2*np.pi):
    """
    Return npts points on the ellipse described by the
    'params = x0, y0, a, b, phi'
    for values of the parametric variable t between tmin and tmax (radians).
    Params can be also dictionary with keys 'x0','y0','a','b','phi'.

    Example:
    x, y = el.get_ellipse_pts((x0, y0, a, b, phi), npts, tmin, tmax)
    """

    printc("Deprecated function! Use Ellipse class instead!",tag='w')

    x0, y0, a, b, phi = fetch_params(params)

    # A grid of the parametric variable, t.
    t = np.linspace(tmin, tmax, npts)
    x = x0 + a * cos(t) * cos(phi) - b * sin(t) * sin(phi)
    y = y0 + a * cos(t) * sin(phi) + b * sin(t) * cos(phi)
    return x, y


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

    This is exactly the same algorithm as in the pyFAI library.
    
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

    printc("Deprecated function! Use Ellipse class instead!",tag='w')

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

