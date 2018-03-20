#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 15:34:18 2018

@author: interferon
"""

#rad s probabilističkim klasifikacijskim modelom s jednim skrivenim slojem

import numpy as np
import matplotlib.pyplot as plt
import random
import pdb
import IPython
import data
import tensorflow as tf

tf.set_random_seed(100)

param_niter=200000
param_delta=0.0005
param_lambda=1e-3   #regularizacija

def fcann2_train(X,Y_, H):
    C = np.max(Y_) + 1
    N, D = X.shape   #D = broj featurea
    b1=np.zeros(H)
    W1=np.random.randn(D, H)
    
    b2=np.zeros(C)
    W2=np.random.randn(H, C)
    
    prethodniLoss=0
    
    for i in range(param_niter):

        s1 = np.dot(X, W1) + b1    # N x H
        h1 = np.maximum(s1, 0)     # N x H    #ovo je  ReLU
        
        s2 = np.dot(h1, W2) + b2    # N x C
        expscores = np.exp(s2)    # N x C
        # nazivnik sofmaksa
        sumexp = np.sum(expscores, axis=1)    # N x 1
        # logaritmirane vjerojatnosti razreda 
        probs = expscores / sumexp.reshape(-1,1)     # N x C
        
        correct_class_prob = probs[range(len(X)), Y_]      # N x 1
        
        correct_class_logprobs = -np.log(correct_class_prob)   # N x 1 
        # gubitak
        loss  = np.sum(correct_class_logprobs)

        # dijagnostički ispis
        if i % 10 == 0:
            print("iteration {}: loss {}".format(i, loss))

        # derivacije komponenata gubitka po rezultatu
        dL_ds2 = probs   # N x C
        dL_ds2[range(len(X)),Y_] -= 1
        
        dL_dW2 = np.dot(dL_ds2.T, h1)   # C x H
        dL_db2 = np.sum(dL_ds2, axis=0) # C x 1
        
        dL_dh1 = np.dot(dL_ds2, W2.T)  # N x H
        dL_ds1 = dL_dh1          # N x H
        dL_ds1[s1 <= 0] = 0
        
        dL_dW1 = np.dot(dL_ds1.T, X) # H x D
        dL_db1 = np.sum(dL_ds1, axis=0)  # H x 1
        
        # gradijenti parametara
        W1 += -param_delta * dL_dW1.T
        b1 += -param_delta * dL_db1.T
        W2 += -param_delta * dL_dW2.T
        b2 += -param_delta * dL_db2.T
        
    return W1,b1,W2,b2


def fcann2_classify(X, W1, b1, W2, b2):
    s1 = np.dot(X, W1) + b1    # N x H
    h1 = np.maximum(s1, 0)     # N x H    #ovo je  ReLU
    
    s2 = np.dot(h1, W2) + b2    # N x C
    expscores = np.exp(s2)    # N x C
    # nazivnik sofmaksa
    sumexp = np.sum(expscores, axis=1)    # N x 1
    # logaritmirane vjerojatnosti razreda 
    probs = expscores / sumexp.reshape(-1,1)     # N x C
    
    return probs


def fcann2_classify_function(W1, b1, W2, b2):
    return lambda X: np.argmax(fcann2_classify(X, W1, b1, W2, b2), axis=1)

    
if __name__=="__main__":
    #Isprobajte njihov rad na umjetnom skupu 2D podataka dvaju razreda dobivenih iz Gaussove mješavine od 6 komponenata
    #dimenzija skrivenog sloja: 5
    
    np.random.seed(100)

    # get the training dataset
    X,Y_ = data.sample_gmm(6,2,10)

    # train the model
    W1,b1,W2,b2 = fcann2_train(X, Y_, 5)

    # evaluate the model on the training dataset
    probs = fcann2_classify(X, W1,b1,W2,b2)
    Y = [np.argmax(ps) for ps in probs]

    # report performance
    accuracy, recall, precision = data.eval_perf_multi(Y, Y_)
    print(accuracy, recall, precision)
    plt.show()
    
    # graph the decision surface
    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(fcann2_classify_function(W1,b1,W2,b2), bbox, offset=0.5, width=256, height=256)
    
    # graph the data points
    data.graph_data(X, Y_, Y)

    # show the plot
    plt.show()
