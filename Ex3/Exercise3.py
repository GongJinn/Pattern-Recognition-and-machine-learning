# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 11:04:00 2019

@author: free_
"""

import numpy as np
import matplotlib.pyplot as plt
import math


####################Q3####################
#Part A
x=np.arange(0,901,1)
n=np.arange(500,601,1)
y1=np.zeros(500)
y2=np.cos(2 * np.pi * 0.1 * n)
y3=np.zeros(300)
y=np.concatenate((y1,y2,y3))
plt.figure()
plt.title("Noiseless Signal")
plt.xticks(np.linspace(0,900,10))
plt.yticks(np.linspace(-1,1,5))
plt.plot(x, y)
plt.show()

#PartB
y_n = y + np.sqrt(0.5) * np.random.randn(y.size)
plt.figure()
plt.title("Noisy Signal")
plt.xticks(np.linspace(0,900,10))
plt.yticks(np.linspace(-3,3,7))
plt.plot(x, y_n)
plt.show()
##
yT=[]
for i in x:
    yT.append(sum(y_n[0:i])*np.cos(2 * np.pi * 0.1 * i))
plt.plot(x, yT)
plt.show()

#partC
plt.figure()
plt.subplot(3,1,1)
plt.title("Noiseless Signal")
plt.xticks(np.linspace(0,900,10))
plt.yticks(np.linspace(-1,1,5))
plt.plot(x, y)
plt.subplot(3,1,2)
plt.title("Noisy Signal")
plt.xticks(np.linspace(0,900,10))
plt.yticks(np.linspace(-3,3,7))
plt.plot(x, y_n)
plt.subplot(3,1,3)
plt.title("Detection Result")
plt.xticks(np.linspace(0,900,10))
plt.plot(x, yT)
plt.show()
####################Q4####################
#Part A
x=np.arange(0,900,1)
n=np.arange(501,600,1)
y1=np.zeros(501)
y2=np.cos(2 * np.pi * 0.03 * n+4.62)
y3=np.zeros(300)
y=np.concatenate((y1,y2,y3))
plt.figure()
plt.title("Noiseless Signal")
plt.xticks(np.linspace(0,900,10))
plt.yticks(np.linspace(-1,1,5))
plt.plot(x, y)
plt.show()
#PartB
y_n = y + np.sqrt(0.5) * np.random.randn(y.size)
plt.figure()
plt.title("Noisy Signal")
plt.xticks(np.linspace(0,900,10))
plt.yticks(np.linspace(-3,3,7))
plt.plot(x, y_n)
plt.show()
##
h = np.exp(-2 * np.pi * 1j * 0.03 * n)
yR = np.abs(np.convolve(h,y_n[501:599], 'same'))
plt.plot(n, yR)
plt.show()
####################Q5####################
from sklearn import neighbors
from scipy.io import loadmat
from sklearn.metrics import accuracy_score
import random
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


#Part A
mat = loadmat("twoClassData.mat")
X = mat["X"]
Y = mat["y"].ravel()
XY = list(zip(X, Y))#X,Y use the same order to shuffle
random.Random(100).shuffle(XY)
X,Y = zip(*XY)

#Part B
Xtrain=X[0:200]
Ytrain=Y[0:200]
Xtest=X[200:]
Ytest=Y[200:]
#Part C
model=neighbors.KNeighborsClassifier(n_neighbors=8, leaf_size=1)
model.fit(Xtrain,Ytrain)
Ypredict1=model.predict(Xtest)
accuracy_score(Ytest,Ypredict1)
#Part D
clf = LinearDiscriminantAnalysis()
clf.fit(Xtrain,Ytrain)
Ypredict2=clf.predict(Xtest)
accuracy_score(Ytest,Ypredict2)