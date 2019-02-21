# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 20:05:28 2019

@author: free_
"""

from __future__ import division
from keras.datasets import mnist
from keras.models import Sequential
from keras.models import Model
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.utils import to_categorical

import glob
import numpy as np
import os
import matplotlib.pyplot as plt
import sklearn.model_selection


def load_data(folder):
    
    X = []          # Images go here
    y = []          # Class labels go here
    classes = []    # All class names go here
    
    subdirectories = glob.glob(folder + "/*")
    
    # Loop over all folders
    for d in subdirectories:
        
        # Find all files from this folder
        files = glob.glob(d + os.sep + "*.jpg")
        
        # Load all files
        for name in files:
            
            # Load image and parse class name
            img = plt.imread(name)
            class_name = name.split(os.sep)[-2]

            # Convert class names to integer indices:
            if class_name not in classes:
                classes.append(class_name)
            
            class_idx = classes.index(class_name)
            
            X.append(img)
            y.append(class_idx)
    
    # Convert python lists to contiguous numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    return X, y

##############Q3###################
X, y = load_data(".")
X_max=np.max(X)
x_train,x_test, y_train, y_test=sklearn.model_selection.train_test_split(X/X_max,y,test_size=0.2, random_state=0)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
base_model = VGG16(weights='imagenet', include_top=False,input_shape=(64,64,3))
w=base_model.output
w=Flatten()(w)
w=Dense(100,activation='relu')(w)
output=Dense(2,activation='sigmoid')(w)
model=Model(inputs=[base_model.input],outputs=[output])

model.summary() 
model.compile(optimizer='sgd',loss='logcosh',metrics=['accuracy'])
model.fit(x_train,y_train, epochs=20, batch_size=32,validation_data = (x_test, y_test))   
##############Q4###################
from scipy.io import loadmat
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
mat = loadmat("arcene.mat")
X_test=mat['X_test']
X_train=mat['X_train']
y_test=mat['y_test'].ravel()
y_train=mat['y_train'].ravel()
estimator = LogisticRegression(solver='liblinear',multi_class='ovr')
rfe = RFECV(estimator=estimator, step=50,verbose = 1)
rfe.fit(X_train,y_train)
print(rfe.support_)
plt.plot(range(0,10001,50), rfe.grid_scores_)
y_predict=rfe.predict(X_test)
print('RFECV -',accuracy_score(y_test,y_predict)) 
##############Q5###################
from sklearn.model_selection import GridSearchCV

param_test1 ={'C':[1,10,50,100,1000,10000,100000]}  
clf=LogisticRegression(solver='liblinear',penalty='l1')
gsearch1= GridSearchCV(estimator =clf,param_grid =param_test1,scoring='roc_auc',cv=5)  
gsearch1.fit(X_train,y_train)  
print(gsearch1.best_params_, gsearch1.best_score_ )
clf.fit(X_train,y_train)
print(len(np.nonzero(clf.coef_)[0]))
y_predict=clf.predict(X_test)
print('CLF -',accuracy_score(y_test,y_predict)) 