import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator
from matplotlib import cm
import math

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
n = 100

def random_three_vector_sphere_1():
    """
    Generates a random 3D unit vector (direction) with a uniform spherical distribution
    Algo from http://stackoverflow.com/questions/5408276/python-uniform-spherical-distribution
    :return:
    """
    phi = np.random.uniform(0,np.pi*2)
    costheta = np.random.uniform(-1,1)

    theta = np.arccos( costheta )
    x = np.sin( theta) * np.cos( phi )
    y = np.sin( theta) * np.sin( phi )
    z = np.cos( theta )
    return (x,y,z)

def random_three_vector_sphere_2(cx = 0 , cy = 0, cz = 0, radius = 1):
    """
    Generates a random 3D unit vector (direction) with a uniform spherical distribution
    Algo from http://mathworld.wolfram.com/SpherePointPicking
    :return:
    """
    U = np.random.uniform(0,1)
    V = np.random.uniform(0,1)
    
    theta = np.pi*2*U
    cosphi = 2*V - 1

    phi = np.arccos( cosphi )
    x = np.sin( phi ) * np.cos( theta )
    y = np.sin( phi ) * np.sin( theta )
    z = np.cos( phi )
    return (x*radius+cx,y*radius+cy,z*radius+cz)

def random_next_vertix(px ,py, pz, delta = 0.001):
    cx = 0 
    cy = 0 
    cz = 0 
    r = 1
    nx, ny, nz = random_three_vector_sphere_2(px, py, pz, delta)
    d = math.sqrt((nx-cx)**2 + (ny-cy)**2 + (nz-cz)**2)
    while d > r:
        nx, ny, nz = random_three_vector_sphere_2(nx, ny, nz, delta)
        d = math.sqrt((nx-cx )**2 + (ny-cy)**2 + (nz-cz)**2)   
    return nx, ny, nz

X = []
Y = []
Z = []
delta = 0.001

#first vertix on unit sphere
x1,y1,z1 = random_three_vector_sphere_2()
print(x1, y1, z1)
X.append(x1)
Y.append(y1)
Z.append(z1)
#ax.scatter(x1, y1, z1)

xs,ys,zs = random_next_vertix(x1,y1,z1,delta)
print(xs, ys, zs)
X.append(xs)
Y.append(ys)
Z.append(zs)
#ax.scatter(xs, ys, zs)

for i in range(n):
    nx,ny,nz = random_next_vertix(xs,ys,zs,delta)
    print(nx, ny, nz)
    xs = nx
    ys = ny
    zs = nz
    #ax.scatter(xs, ys, zs)
    X.append(xs)
    Y.append(ys)
    Z.append(zs)
    
#Test 1 - tri-surface plot    
#ax.plot_trisurf(X, Y, Z, cmap=cm.jet, linewidth=0.2)
ax.set_zlim3d(-1, 1)
#ax.set_ylim3d(-1, 1)
#ax.set_xlim3d(-1, 1)

#Test 2 - wireframe
ax.plot_wireframe(X, Y, Z)

#Test 3 - scatterplot
#ax.scatter(X, Y, Z)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
#ax.set_zlim3d(-1, 1)
#ax.set_ylim3d(-1, 1)
#ax.set_xlim3d(-1, 1)

plt.show()
