#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 19:21:52 2018

@author: interferon
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import data

class TFDeep:
  def __init__(self, neurons, activation_function, param_delta=0.0001, param_lambda=0.05):
    """Arguments:
       - D: dimensions of each datapoint 
       - C: number of classes
       - param_delta: training step
    """
    D = neurons[0]
    C = neurons[len(neurons)-1]
    hidden_layers = len(neurons) - 2   #number of hidden layers
    # definicija podataka i parametara:
    # definirati self.X, self.Yoh_, self.W, self.b
    self.X  = tf.placeholder(tf.float64, [None, None])                 # N x D
    self.Yoh_ = tf.placeholder(tf.float64, [None, None])           # N x C
    self.W = []     #list
    self.b = []     #list
    self.h = []     #list
    

    #Nelinearnost u skrivenim slojevima možete izraziti uz pomoć 
    #funkcija TensorFlowa tf.ReLU, tf.sigmoid odnosno tf.tanh.
    for i in range(hidden_layers + 1):
        self.W.append(tf.Variable(initial_value = np.random.randn(neurons[i], neurons[i+1]), name="W"+str(i)))
        self.b.append(tf.Variable(initial_value = np.zeros(neurons[i+1]), name="b"+str(i)))
        

    self.probs = self.X     #u prvoj iteraciji 
        
    for i in range(hidden_layers):
        #s = tf.matmul(self.X, self.W[i]) + self.b[i]
        #self.h.append(tf.ReLU(s))
        self.probs = activation_function(tf.matmul(self.probs, self.W[i]) + self.b[i])
    
    self.probs = tf.nn.softmax(tf.matmul(self.probs, self.W[-1]) + self.b[-1])  # N x C  
    
        
    # formulacija gubitka: self.loss
    #   koristiti: tf.log, tf.reduce_sum, tf.reduce_mean
    
    self.loss = tf.reduce_sum(- tf.log(tf.reduce_sum(self.Yoh_ * self.probs, 1))) + param_lambda * tf.reduce_sum([tf.reduce_sum(x**2) for x in self.W])

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
            print('deep diagn, loss:', i, val_loss)
    #return W, b

  def eval(self, Xg):
    """Arguments:
       - X: actual datapoints [NxD]
       Returns: predicted class probabilites [NxC]
    """
    #   koristiti: tf.Session.run
    probs = self.session.run([self.probs], feed_dict={self.X: Xg})
    return probs[0]

  def count_params(self):
     counter=0 
     print('\n variables:')
     for i in tf.trainable_variables():
          print(i.name)
          if (i.name[0] == 'b' or i.name[0] =='W'):
              if len(i.shape) == 1:
                  counter += i.shape[0].value
              else:
                  counter += i.shape[0].value * i.shape[1].value
     print('no of vars:', counter, '\n')
     return counter

def deep_classify_function(self):
    return lambda Xy: np.argmax(self.eval(Xy), axis=1)

if __name__ == "__main__":
  # inicijaliziraj generatore slučajnih brojeva
  np.random.seed(100)
  tf.set_random_seed(100)

  # instanciraj podatke X i labele Yoh_
  #X,Y_ = data.sample_gmm(4, 2, 40)
  X,Y_ = data.sample_gmm(6, 2, 10)
  Yoh_ = data.class_to_onehot(Y_)
  
  # izgradi graf:
  tfdeep = TFDeep([2,10, 10, 2], tf.nn.relu, 0.0005, 0.04)

  # nauči parametre:
  tfdeep.train(X, Yoh_, 10000)

  
  tfdeep.count_params()
  
  # dohvati vjerojatnosti na skupu za učenje
  probs = tfdeep.eval(X)
  Y = [np.argmax(ps) for ps in probs]
  
  # ispiši performansu (preciznost i odziv po razredima)
  accuracy, recall, precision = data.eval_perf_multi(Y_, np.array(Y))
  print('acc:', accuracy, '\n rec', recall)
  plt.show()
  
  # iscrtaj rezultate, decizijsku plohu
  bbox=(np.min(X, axis=0), np.max(X, axis=0))
  data.graph_surface(deep_classify_function(tfdeep), bbox, offset=0.5, width=256, height=256)
  # graph the data points
  data.graph_data(X, Y_, Y)
  # show the plot
  plt.show()
  tf.reset_default_graph()