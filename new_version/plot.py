import numpy as np
from math import cos, sqrt
import matplotlib.pyplot as plt
import numpy.polynomial.laguerre as lag



N = 15  # gaussian nodes
M = 50 # Grid points
L = 12.0  # range of grid
p0 = 2.3
Lambda = 1.0
pi = 3.14159265358979323846 



def sort_list(list1, list2):
    zipped_pairs = zip(list2, list1)
    z = [x for _, x in sorted(zipped_pairs)]
    return z


root = lag.laggauss(N)[0]

node = np.array([i*L/(M-1) + p0 for i in range(M)]) # np.array([L/2*cos(pi*i/(M-1)) + p0 + L/2 for i in range(M)]) #np.array([i*L/(M-1) + p0 for i in range(M)]) 

x = np.concatenate((root, node), axis=None)

function = np.loadtxt("data_cpd.txt")
sigma = function[2*(N+M):3*(N+M)]
pii = function[(5*(N+M)+1):(6*(N+M)+1)]
sigma1 = sort_list(sigma, x)
pii1 = sort_list(pii, x)
x = np.sort(x)
#print(function)
print(sigma)
print(pii)

fig = plt.figure(1)
ax = fig.gca()
plt.scatter(x, sigma)
plt.scatter(x, pii, c = "r")
plt.grid()
plt.xlabel('momentum')
plt.ylabel('y(t)')
plt.show()


