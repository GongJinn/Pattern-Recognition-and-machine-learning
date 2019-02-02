# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 19:35:47 2019

@author: free_
"""

import numpy as np
from sklearn import neighbors
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

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
    
    
#local test
rs = GroupShuffleSplit(n_splits=4, test_size=0.2)
    
def testFunc(model):
    score=[]
    for x in [xTrain1,xTrain2,xTrain3]:
        accuracy=[]
        for train, test in rs.split(x,yTrain3,groups=groups2):
            model.fit(x[train],yTrain3[train])
            accuracy.append(accuracy_score(yTrain3[test],model.predict(x[test])))
        score.append(np.mean(accuracy))
    return(score)

lda=LinearDiscriminantAnalysis()
print(testFunc(lda))

SVClinear=SVC(kernel ='linear')
print(testFunc(SVClinear))
    
SVCrbf=SVC(kernel ='rbf',gamma='scale',C=0.4)
print(testFunc(SVCrbf))  

LR=LogisticRegression(solver='liblinear',multi_class='ovr')
print(testFunc(LR)) 
    
RFC = RandomForestClassifier(n_estimators =57)
print(testFunc(RFC))

ETC = ExtraTreesClassifier(n_estimators=57)
print(testFunc(ETC))

ABC = AdaBoostClassifier(n_estimators=50)
print(testFunc(ABC))

GBC = GradientBoostingClassifier(n_estimators=100)
print(testFunc(GBC))

KNN=neighbors.KNeighborsClassifier(n_neighbors=140)
print(testFunc(KNN))

#Print result of ETC
ETC.fit(xTrain3,yTrain3)
y_pred = ETC.predict(xTest1)
labels = list(le.inverse_transform(y_pred))
with open("submission.csv", "w") as fp:
    fp.write("#Id,Surface\n")
    for i, label in enumerate(labels):
        fp.write("%d,%s\n" % (i, label))