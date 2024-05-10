
# %%

import math
import numpy as np
import matplotlib.pyplot as plt
from numpy import sin, cos, pi, sqrt
import elliptools as ellt
import importlib
importlib.reload(ellt)

class Ellipse:
    def __init__(self, x, y, width, height, angle = 0):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.angle = angle
        
    def rotation_matrix(self):
        """
        Returns the rotation matrix for the ellipse's rotation.
        """
        a = math.cos(self.angle)
        b = math.sin(self.angle)
        return np.array([[a, -b], [b, a]])
    
    def get_point(self, angle):
        """
        Returns the point on the ellipse at the specified local angle.
        """
        r = self.rotation_matrix()
        xe = 0.5 * self.width * math.cos(angle)
        ye = 0.5 * self.height * math.sin(angle)
        return np.dot(r, [xe, ye]) + [self.x, self.y]
    
    def get_points(self, count):
        """
        Returns an array of points around the ellipse in the specified count.
        """
        t = np.linspace(0, 2 * math.pi, count)
        xe = 0.5 * self.width * np.cos(t)
        ye = 0.5 * self.height * np.sin(t)
        r = self.rotation_matrix()
        return np.dot(np.column_stack([xe, ye]), r.T) + [self.x, self.y]
    
    def find_distance1(self, x, tolerance = 1e-8, max_iterations = 10000, learning_rate = 0.01):
        """
        Finds the minimum distance between the specified point and the ellipse
        using gradient descent.
        """
        x = np.asarray(x)
        r = self.rotation_matrix()
        x2 = np.dot(r.T, x - [self.x, self.y])
        t = math.atan2(x2[1], x2[0])
        a = 0.5 * self.width
        b = 0.5 * self.height
        iterations = 0
        error = tolerance
        errors = []
        ts = []
        
        while error >= tolerance and iterations < max_iterations:
            cost = math.cos(t)
            sint = math.sin(t)
            x1 = np.array([a * cost, b * sint])
            xp = np.array([-a * sint, b * cost])
            dp = 2 * np.dot(xp, x1 - x2)
            t -= dp * learning_rate
            error = abs(dp)
            errors.append(error)
            ts.append(t)
            iterations += 1
            
        ts = np.array(ts)
        errors = np.array(errors)
        y = np.linalg.norm(x1 - x2)
        success = error < tolerance and iterations < max_iterations
        return dict(x = t, y = y, error = error, iterations = iterations, success = success, xs = ts,  errors = errors)
    
    def find_distance2(self, x, tolerance = 1e-8, max_iterations = 1000):
        """
        Finds the minimum distance between the specified point and the ellipse
        using Newton's method.
        """
        x = np.asarray(x)
        r = self.rotation_matrix()
        x2 = np.dot(r.T, x - [self.x, self.y])
        t = math.atan2(x2[1], x2[0])
        a = 0.5 * self.width
        b = 0.5 * self.height

        print(t,a,b)
        
        # If point is inside ellipse, generate better initial angle based on vertices
        if (x2[0] / a)**2 + (x2[1] / b)**2 < 1:
            ts = np.linspace(0, 2 * math.pi, 24, endpoint = False)
            xe = a * np.cos(ts)
            ye = b * np.sin(ts)
            delta = x2 - np.column_stack([xe, ye])
            t = ts[np.argmin(np.linalg.norm(delta, axis = 1))]
            
        iterations = 0
        error = tolerance
        errors = []
        ts = []
                
        while error >= tolerance and iterations < max_iterations:
            cost = math.cos(t)
            sint = math.sin(t)
            x1 = np.array([a * cost, b * sint])
            xp = np.array([-a * sint, b * cost])
            xpp = np.array([-a * cost, -b * sint])
            delta = x1 - x2
            dp = np.dot(xp, delta)
            dpp = np.dot(xpp, delta) + np.dot(xp, xp)
            t -= dp / dpp
            error = abs(dp / dpp)
            errors.append(error)
            ts.append(t)
            iterations += 1
        
        ts = np.array(ts)
        errors = np.array(errors)
        y = np.linalg.norm(x1 - x2) # Distance
        success = error < tolerance and iterations < max_iterations
        return dict(x = t, y = y, error = error, iterations = iterations, success = success, xs = ts,  errors = errors)
    
    def getPoints(self,npts:int=100,tmin:float=0,tmax:float=2*pi) -> tuple:
        """
        Return npts points on the ellipse described by the
        'params = x0, y0, a, b, phi'
        for values of the parametric variable t between tmin and tmax (radians).
        """

        # A grid of the parametric variable, t.
        t = np.linspace(tmin, tmax, npts)
        x = self.x + self.width * cos(t) * cos(self.angle) - self.height * sin(t) * sin(self.angle)
        y = self.y + self.width * cos(t) * sin(self.angle) + self.height * sin(t) * cos(self.angle)
        return x, y


    def plot(self,
             ax:plt.Axes,
             color=None):


        x,y = self.getPoints()
        # Ellipse itself
        ax.plot(x,y,color=color)


# Generate random points
size = 6
np.random.seed(12345)
points = np.random.rand(1, 2)
points = size * points - size * (1 - points)
points = np.append(points, [[0, 0]], axis = 0)

# Ellipse definition
ellipse = Ellipse(0, 0, 6, 10, np.deg2rad(15))

solutions2 = [ellipse.find_distance2(x) for x in points]
print(f"Method 2 Solutions Successful: {np.all([x['success'] for x in solutions2])}")


fig = plt.figure(figsize=(7.5, 4.5))

# Generate ellipse points
ellipse_points = ellipse.get_points(100)


# Method 2 Plot
ax = fig.add_subplot(111,aspect = "equal")

for solution, point in zip(solutions2, points):
    x = np.array([ellipse.get_point(solution["x"]), point])
    ax.plot(x[:,0], x[:,1])
    
ax.plot(points[:,0], points[:,1], "k.")
ax.plot(ellipse_points[:,0], ellipse_points[:,1], "k-")

P = np.array([1,1])
distP = ellipse.find_distance2(P)
plt.plot(P[0],P[1],'ro',markersize=10,markeredgecolor='k')
P2 = ellipse.get_point(distP["x"])
ax.plot(P2[0],P2[1],'yo',markersize=10,markeredgecolor='k')
print(f"Distance: {sqrt((P2[0]-P[0])**2 + (P2[1]-P[1])**2)}")


el2 = ellt.Ellipse(0,0,6/2,10/2,np.deg2rad(15))
print(el2.find_distance2(P))