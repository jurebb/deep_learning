#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 21:20:46 2018

@author: interferon
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import data

## 1. definicija računskog grafa
# podatci i parametri
X  = tf.placeholder(tf.float32, [None])
Y_ = tf.placeholder(tf.float32, [None])
a = tf.Variable(0.0)
b = tf.Variable(0.0)

# afini regresijski model
Y = a * X + b

N = tf.shape(X)
N = tf.cast(N, tf.float32)

# kvadratni gubitak
loss = tf.divide( (Y-Y_)**2, 2*N)

# optimizacijski postupak: gradijentni spust
trainer = tf.train.GradientDescentOptimizer(0.05)
#train_op = trainer.minimize(loss)
compute_op = trainer.compute_gradients(loss, var_list=[a,b])
apply_op = trainer.apply_gradients(compute_op)

## 2. inicijalizacija parametara
sess = tf.Session()
sess.run(tf.initialize_all_variables())

## 3. učenje
# neka igre počnu!
for i in range(500):
  val_N, val_loss, grads, val_a,val_b = sess.run([N, loss, compute_op, a,b], 
      feed_dict={X: [1,2, 3, 5], Y_: [3,5, 7, 11]})
  sess.run(apply_op, feed_dict={X: [1,2, 3, 5], Y_: [3,5, 7, 11]})
  print(i, ':', val_a, val_b)
  print('gradijenti:', grads)