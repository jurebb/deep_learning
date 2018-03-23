#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 01:06:50 2018

@author: interferon
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import data

class TFLogreg:
  def __init__(self, D, C, param_delta=0.5):
    """Arguments:
       - D: dimensions of each datapoint 
       - C: number of classes
       - param_delta: training step
    """

    # definicija podataka i parametara:
    # definirati self.X, self.Yoh_, self.W, self.b
    self.X  = tf.placeholder(tf.float64, [None, D])                 # N x D
    self.Yoh_ = tf.placeholder(tf.float64, [None, (C)])           # N x C
    self.W = tf.Variable(initial_value = np.random.randn(D, C))     # D x C
    self.b = tf.Variable(initial_value = np.zeros(C))               # C x 1

    # formulacija modela: izračunati self.probs
    #   koristiti: tf.matmul, tf.nn.softmax
    self.probs = tf.nn.softmax(tf.matmul(self.X, self.W) + self.b)  # N x C  
    
    #provjeriti radi li + onako kako treba u tf
        
    # formulacija gubitka: self.loss
    #   koristiti: tf.log, tf.reduce_sum, tf.reduce_mean
    
    self.loss = tf.reduce_sum(- tf.log(tf.reduce_sum(self.Yoh_ * self.probs, 1)))

    # formulacija operacije učenja: self.train_step
    #   koristiti: tf.train.GradientDescentOptimizer,
    #              tf.train.GradientDescentOptimizer.minimize
    trainer = tf.train.GradientDescentOptimizer(param_delta)
    self.train_step = trainer.minimize(self.loss)

    # instanciranje izvedbenog konteksta: self.session
    #   koristiti: tf.Session
    self.session = tf.Session()
    

  def train(self, X, Yoh_, param_niter):
    """Arguments:
       - X: actual datapoints [NxD]
       - Yoh_: one-hot encoded labels [NxC]
       - param_niter: number of iterations
    """
    # incijalizacija parametara
    #   koristiti: tf.initialize_all_variables
    self.session.run(tf.initialize_all_variables())

    # optimizacijska petlja
    #   koristiti: tf.Session.run
    for i in range(param_niter):
        val_loss, _ = self.session.run([self.loss, self.train_step], feed_dict={self.X: X, self.Yoh_: Yoh_})
        if(i%100 == 0):
            print('diagnostic, loss:', i, val_loss)
    #return W, b

  def eval(self, X):
    """Arguments:
       - X: actual datapoints [NxD]
       Returns: predicted class probabilites [NxC]
    """
    #   koristiti: tf.Session.run
    probs = self.session.run([self.probs], feed_dict={self.X: X})
    return probs[0]

def logreg_classify_function(self):
    return lambda X: np.argmax(self.eval(X), axis=1)

if __name__ == "__main__":
  # inicijaliziraj generatore slučajnih brojeva
  np.random.seed(100)
  tf.set_random_seed(100)

  # instanciraj podatke X i labele Yoh_
  X,Y_ = data.sample_gauss(3, 100)
  Yoh_ = data.class_to_onehot(Y_)
  
  # izgradi graf:
  tflr = TFLogreg(X.shape[1], Yoh_.shape[1], 0.004)

  # nauči parametre:
  tflr.train(X, Yoh_, 20000)

  # dohvati vjerojatnosti na skupu za učenje
  probs = tflr.eval(X)
  Y = [np.argmax(ps) for ps in probs]
  
  # ispiši performansu (preciznost i odziv po razredima)
  accuracy, recall, precision = data.eval_perf_multi(Y_, np.array(Y))
  print('acc:', accuracy, '\n rec', recall)
  plt.show()
  
  # iscrtaj rezultate, decizijsku plohu
  bbox=(np.min(X, axis=0), np.max(X, axis=0))
  data.graph_surface(logreg_classify_function(tflr), bbox, offset=0.5, width=256, height=256)
  # graph the data points
  data.graph_data(X, Y_, Y)
  # show the plot
  plt.show()