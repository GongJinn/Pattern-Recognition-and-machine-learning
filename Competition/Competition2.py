# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 13:34:47 2019

@author: free_
"""

import numpy as np
import sklearn.model_selection
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from keras.datasets import mnist
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import to_categorical

#load data
xTrain=np.load('X_train_kaggle.npy')
xTest=np.load('X_test_kaggle.npy')
yTrain=np.loadtxt('y_train_final_kaggle.csv',delimiter=',',dtype='str')
groups=np.loadtxt('groups.csv',delimiter=',',dtype='str')

#numerical
le = preprocessing.LabelEncoder()
yTrain1,yTrain2=zip(*yTrain)
yTrain1=np.int64(yTrain1)
le.fit(yTrain2)
yTrain3=le.transform(yTrain2)
yTrain=np.array(list(zip(yTrain1,yTrain3)))
yTrain4= to_categorical(yTrain3)

groups1,groups2,groups3=zip(*groups)
groups1=np.int64(groups1)
groups2=np.int64(groups2)
groups4=le.transform(groups3)
groups=np.array(list(zip(groups1,groups2,groups4)))

#different format of data
xTrain1=[]
xTrain2=[]
xTrain3=[]

for matrix in xTrain:
    xTrain1.append(matrix.ravel())
    xTrain2.append(np.mean(matrix,axis=1))
    xTrain3.append(np.concatenate((np.mean(matrix,axis=1),np.var(matrix,axis=1)),axis=0))
xTrain1=np.float64(xTrain1)
xTrain2=np.float64(xTrain2)
xTrain3=np.float64(xTrain3)

xTest1=[]
for matrix in xTest:
    xTest1.append(np.concatenate((np.mean(matrix,axis=1),np.var(matrix,axis=1)),axis=0))
xTest1=np.float64(xTest1)
    
#xTrainIndex,xTestIndex, yTrainLocal,yTestLocal=sklearn.model_selection.train_test_split(yTrain1,yTrain4,test_size=0.2, random_state=0)    
#local test
rs = GroupShuffleSplit(n_splits=1, test_size=0.2)
    
#def testFunc(model):
#    score=[]
#    for x in [xTrain1,xTrain2,xTrain3]:
#        accuracy=[]
#        for train, test in rs.split(x,yTrain4,groups=groups2):
#            model.fit(x[train],yTrain4[train])
#            accuracy.append(accuracy_score(yTrain4[test],model.predict(x[test])))
#        score.append(np.mean(accuracy))
#    return(score)
#RNN
model= Sequential()
model.add(LSTM(units=100, input_shape=(10,128),kernel_regularizer=regularizers.l1(0.02)))
model.add(Dense(50, activation = 'relu', kernel_regularizer=regularizers.l1(0.01)))
model.add(Dense(20, activation = 'relu', kernel_regularizer=regularizers.l1(0.01)))
model.add(Dense(9, activation = 'softmax', kernel_regularizer=regularizers.l1(0.01)))

model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
for train, test in rs.split(xTrain3,yTrain4,groups=groups2):
    model.fit(xTrain[train],yTrain4[train], epochs=20, batch_size=32,validation_data = (xTrain[test],yTrain4[test]))    
#Print result of ETC
model.fit(xTrain,yTrain4, epochs=20, batch_size=32)
y_pred = model.predict_classes(xTest1)

labels = list(le.inverse_transform(y_pred))
with open("submission.csv", "w") as fp:
    fp.write("# Id,Surface\n")
    for i, label in enumerate(labels):
        fp.write("%d,%s\n" % (i, label))
        
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        