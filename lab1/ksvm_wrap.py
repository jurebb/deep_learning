#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 00:31:27 2018

@author: interferon
"""

import numpy as np
import matplotlib.pyplot as plt
import data
from sklearn import svm
from sklearn.metrics import classification_report 

class KSVMWrap:
    def __init__(self, X, Y_, param_svm_c=1, param_svm_gamma='auto'):
        """
        Konstruira omotač i uči RBF SVM klasifikator
        X,Y_:            podatci i točni indeksi razreda
        param_svm_c:     relativni značaj podatkovne cijene
        param_svm_gamma: širina RBF jezgre
        """
        self.clf = svm.SVC(kernel='rbf', C=param_svm_c, gamma=param_svm_gamma)
        self.clf = self.clf.fit(X, Y_)
        
        self.support = self.clf.support_   #Indeksi podataka koji su odabrani za potporne vektore
        
        self.Y_ = Y_
    
    def predict(self, X):
        return self.clf.predict(X)

    def get_scores(self, X):
        return classification_report(self.Y_, self.predict(X))

if __name__ == "__main__":
    # inicijaliziraj generatore slučajnih brojeva
    np.random.seed(100)
    
    #X,Y_ = data.sample_gmm(4, 2, 40)
    X,Y_ = data.sample_gmm(6, 2, 10)
    Yoh_ = data.class_to_onehot(Y_)
  
    svm2 = KSVMWrap(X, Y_)
    
    print(svm2.get_scores(X))
    
    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(svm2.predict, bbox, offset=0.5, width=256, height=256)
    data.graph_data(X, Y_, svm2.predict(X), special = svm2.support)
    
    plt.show()
  