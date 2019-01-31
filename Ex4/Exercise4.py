# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 09:21:25 2019

@author: free_
"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection
from sklearn import neighbors
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

##############Q3###################
digits = load_digits()
#print(digits.keys())
#plt.gray()
#plt.imshow(digits.images[0])
#plt.show()
x_train,x_test, y_train, y_test=sklearn.model_selection.train_test_split(digits.data,digits.target,test_size=0.2, random_state=0)
#KNN
model1=neighbors.KNeighborsClassifier()
model1.fit(x_train,y_train)
Ypredict1=model1.predict(x_test)
print('KNN -',accuracy_score(y_test,Ypredict1))
#LDA
model2=LinearDiscriminantAnalysis()
model2.fit(x_train,y_train)
Ypredict2=model2.predict(x_test)
print('LDA -',accuracy_score(y_test,Ypredict2))
#SVM
model3=SVC(kernel ='linear')
model3.fit(x_train,y_train)
Ypredict3=model3.predict(x_test)
print('SVM -',accuracy_score(y_test,Ypredict3)) 
#LRM
model4=LogisticRegression(solver='liblinear',multi_class='ovr')
model4.fit(x_train,y_train)
Ypredict4=model4.predict(x_test)
print('LRM -',accuracy_score(y_test,Ypredict4)) 

##############Q4###################
import traffic_signs

X, y = traffic_signs.load_data(".")
F = traffic_signs.extract_lbp_features(X)
for model in [model1,model2,model3,model4]:
    print(np.mean(cross_val_score(model,F, y, cv=5)))
    
##############Q5###################
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

model5 = RandomForestClassifier(n_estimators = 100)
model6 = ExtraTreesClassifier(n_estimators=100)
model7 = AdaBoostClassifier(n_estimators=100)
model8 = GradientBoostingClassifier(n_estimators=100)

for model in [model5,model6,model7,model8]:
    print(np.mean(cross_val_score(model,F, y, cv=5)))


