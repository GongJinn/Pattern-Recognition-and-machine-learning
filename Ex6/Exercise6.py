# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 19:57:02 2019

@author: free_
"""
from __future__ import division
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
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
#X_trans=[]
#for i in X:
#    i=i.transpose((2,0,1))
#    X_trans.append(i)
X_max=np.max(X_trans)
X_trans=X_trans/X_max
x_train,x_test, y_train, y_test=sklearn.model_selection.train_test_split(X/X_max,y,test_size=0.2, random_state=0)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
  
##############Q4###################
w, h = 5, 5 # Conv. window size
model= Sequential()
model.add(Conv2D(32, (w, h),input_shape=(64, 64,3),activation = 'relu',padding = 'same'))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Conv2D(32, (w, h),activation ='relu',padding = 'same'))
model.add(MaxPooling2D((4,4)))
model.add(Flatten())
model.add(Dense(100, activation = 'sigmoid'))
model.add(Dense(2, activation = 'sigmoid'))
model.summary()  
##############Q5###################  
model.compile(optimizer='sgd',loss='logcosh',metrics=['accuracy'])
model.fit(x_train,y_train, epochs=20, batch_size=32,validation_data = (x_test, y_test))    
    
    
    