# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 21:12:59 2019

@author: free_
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import loadmat

##############Q1###################
data=np.loadtxt('locationData.csv')
print(data.shape)

##############Q2###################
plt.figure()
plt.plot(data.T[0], data.T[1])
plt.show()

plt.figure()
ax = plt.subplot(1, 1, 1, projection = "3d")
ax.plot(data.T[0], data.T[1], data.T[2])
plt.show()

##############Q3###################
test1=[]
with open('locationData.csv','r')as f:
    for line in f:
        numbers=[]
        for number in line.split(' '):
            numbers.append(float(number))
        test1.append(numbers)
test1=np.array(test1)
print((test1==data).all())
print((test1==data).any())

##############Q4###################
mat = loadmat("twoClassData.mat")
print(mat.keys())
X = mat["X"]
y = mat["y"].ravel()
X0=X[y==0,:]
X1=X[y==1,:]
plt.figure()
plt.scatter(X0[:,0],X0[:,1],c='red',edgecolors='black')
plt.scatter(X1[:,0],X1[:,1],c='blue',edgecolors='black')
plt.show()

##############Q5###################
x=np.load('x.npy')
y=np.load('y.npy')
A = np.vstack([x, np.ones(len(x))]).T
print(np.linalg.lstsq(A, y,rcond=None)[0])