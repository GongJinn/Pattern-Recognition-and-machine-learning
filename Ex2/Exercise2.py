# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 15:17:18 2019

@author: free_
"""

import numpy as np
import matplotlib.pyplot as plt
import math

##############Q4###################
#partA
n=np.linspace(0,99,100)
w = np.sqrt(0.25) * np.random.randn(100)
f0=0.017
y=[]
for i in n:
    y.append(math.sin(f0*i*2*math.pi)+w[int(i)])
plt.figure()
plt.scatter(n,y)
plt.show()
#partB
scores = []
frequencies = []
x=np.array(y)
for f in np.linspace(0, 0.5, 1000):
    # Create vector e. Assume data is in x.
    n = np.arange(100)
    z = -2*math.pi*1j*f*n
    e = np.exp(z)
    score =abs(np.dot(x,e))
    scores.append(score)
    frequencies.append(f)
fHat = frequencies[np.argmax(scores)]
print(fHat)

##############Q5###################
from matplotlib.image import imread

# Read the data

img = imread("uneven_illumination.jpg")
plt.imshow(img, cmap='gray')
plt.title("Image shape is %dx%d" % (img.shape[1], img.shape[0]))
plt.show()

# Create the X-Y coordinate pairs in a matrix
X, Y = np.meshgrid(range(1300), range(1030))
Z = img

x = X.ravel()
y = Y.ravel()
z = Z.ravel()

# ********* TODO 1 **********
H=np.mat(np.column_stack([np.multiply(x,x),np.multiply(y,y),np.multiply(x,y),x,y,np.ones_like(x)]))
# Create data matrix
# Use function "np.column_stack".
# Function "np.ones_like" creates a vector like the input.

# ********* TODO 2 **********
theta=np.linalg.lstsq(H,z,rcond=None)[0]
# Solve coefficients
# Use np.linalg.lstsq
# Put coefficients to variable "theta" which we use below.

# Predict
z_pred = H @ theta
Z_pred = np.reshape(z_pred, X.shape)

# Subtract & show
S = Z - Z_pred
plt.imshow(S, cmap = 'gray')
plt.show()