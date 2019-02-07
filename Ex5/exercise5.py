# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 13:11:51 2019

@author: free_
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

####################Q3####################
def log_loss(w, X, y):
    
    L = 0 # Accumulate loss terms here.
    # Process each sample in X:
    for n in range(X.shape[0]):
        L += np.log(1 + np.exp(y[n] * np.dot(w, X[n])))
    return L
    
def grad(w, X, y):
    G = 0 # Accumulate gradient here.
    # Process each sample in X:
    for n in range(X.shape[0]):
        numerator = np.exp(-y[n] * np.dot(w, X[n])) * (-y[n]) * X[n]     # TODO: Correct these lines
        denominator = 1 + np.exp(-y[n] * np.dot(w, X[n]))   # TODO: Correct these lines
        G += numerator / denominator
    return G
    
if __name__ == "__main__":
    # 1) Load X and y.    
    X=np.loadtxt('X.csv',delimiter=',')
    y=np.loadtxt('y.csv',delimiter=',')
    # 2) Initialize w at w = np.array([1, -1])
    w = np.array([1, -1])
    # 3) Set step_size to a small positive value.
    step_size=0.01
    # 4) Initialize empty lists for storing the path and
    W = []
    accuracies = []
    for iteration in range(100):

        # 5) Apply the gradient descent rule.
        w=w-step_size*grad(w,X,y)
        # 6) Print the current state.
        print ("Iteration %d: w = %s (log-loss = %.2f)" % \
              (iteration, str(w), log_loss(w, X, y)))
        # 7) Compute the accuracy (already done for you)
        # Predict class 1 probability
        y_prob = 1 / (1 + np.exp(-np.dot(X, w)))
                # Threshold at 0.5 (results are 0 and 1)
        y_pred = (y_prob > 0.5).astype(int)
                # Transform [0,1] coding to [-1,1] coding
        y_pred = 2*y_pred - 1
        accuracy = np.mean(y_pred == y)
        accuracies.append(accuracy)
        W.append(w)
        
    W = np.array(W)
    plt.figure(figsize = [5,5])
    plt.subplot(211)
    plt.plot(W[:,0], W[:,1], 'ro-')
    plt.xlabel('w$_0$')
    plt.ylabel('w$_1$')
    plt.title('Optimization path')
    
    plt.subplot(212)
    plt.plot(100.0 * np.array(accuracies), linewidth = 2)
    plt.ylabel('Accuracy / %')
    plt.xlabel('Iteration')
    plt.tight_layout()
    plt.savefig("log_loss_minimization.pdf", bbox_inches = "tight")
    
####################Q4####################
import traffic_signs
from sklearn.metrics import accuracy_score
import sklearn.model_selection

X, y = traffic_signs.load_data(".")    
X_flat=[]
for matrix in X:
    X_flat.append(matrix.ravel())
X=np.float64(X_flat)
clf_list = [LogisticRegression(solver='liblinear'), SVC(gamma='scale')]
clf_name = ['LR', 'SVC']
C_range=[1e-5,1e-4,1e-3,1e-2,1e-1,1]
X_train,X_test, y_train, y_test=sklearn.model_selection.train_test_split(X,y,test_size=0.2, random_state=0)
for clf,name in zip(clf_list, clf_name):
    for C in C_range:
        for penalty in ["l1", "l2"]:
            clf.C = C
            clf.penalty = penalty
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            score = accuracy_score(y_test, y_pred)   
            print(score,penalty,C,name)
    
 ####################Q5####################   
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

param_test1 ={'C':[1e-5,1e-4,1e-3,1e-2,1e-1,1]}  
for clf,name in zip(clf_list, clf_name):
    gsearch1= GridSearchCV(estimator =clf,param_grid =param_test1,scoring='roc_auc',cv=5)  

    gsearch1.fit(X,y)  
    
    print(gsearch1.best_params_, gsearch1.best_score_ )
    
tuned_parameters={'C':[1e-5,1e-4,1e-3,1e-2,1e-1,1],'penalty':["l1", "l2"]}
clf=RandomizedSearchCV(LogisticRegression(solver='liblinear'),tuned_parameters,cv=10,scoring='accuracy',n_iter=100)
clf.fit(X,y)
print(clf.best_estimator_)
 
 
 
 
 
 
 
 
 
 
 