#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 22:18:32 2017

@author: ldong
"""

import tensorflow as tf
import numpy as np
import pickle as pk
from datetime import datetime

tf.reset_default_graph()

num_class = 10
dim_img = 32
chanl = 3
batch = 1000
n_iter = 10000
log_freq = 100
history = np.zeros([n_iter/log_freq,3])

with open("/home/ldong/Documents/ML/deep_learning_17Spring/codingHW4/cifar_10_tf_train_test.pkl","rb") as f:
#with open("/Users/ldong/Dropbox/Coursework RPI 17Spring/Deep Learning/HW/codingHW4/src/cifar_10_tf_train_test_s.pkl","rb") as f:
  train_x, train_y, test_x, test_y = pk.load(f)
    
def image_normal(data_x):
  for i in xrange(np.shape(data_x)[0]):
    data_x[i] = (data_x[i] - np.mean(data_x[i]))/max([np.std(data_x[i]), 1e-4])
  return data_x
    
train_x = image_normal(train_x.astype(float))
test_x = image_normal(test_x.astype(float))
train_y = np.array(train_y).astype(int)
test_y = np.array(test_y).astype(int)

dim_train = np.shape(train_y)[0]
dim_test = np.shape(test_y)[0]

images = tf.placeholder(tf.float32, [None,dim_img,dim_img,chanl])
labels = tf.placeholder(tf.int32, [None])

def _variable(name, shape, init):
  var = tf.get_variable(name, shape, initializer=init, dtype=tf.float32)     
  return var

# conv1
with tf.variable_scope('conv1') as scope:
  conv1_kernel = _variable('weights',[5, 5, chanl, 32],tf.contrib.layers.xavier_initializer())
  conv = tf.nn.conv2d(images, conv1_kernel, [1, 1, 1, 1], padding='VALID')
  conv1_biases = _variable('biases', [32], tf.constant_initializer(0.0))
  pre_activation = tf.nn.bias_add(conv, conv1_biases,name='conv1_pre_act')
  conv1 = tf.nn.relu(pre_activation, name=scope.name)

# pool1
pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                       padding='VALID', name='pool1')

# norm1
norm1 = tf.nn.lrn(pool1,4,bias=1.,alpha=0.001/9.,beta=0.75,name='norm1')

# conv2
with tf.variable_scope('conv2') as scope:
  conv2_kernel = _variable('weights',[5, 5, 32, 32],tf.contrib.layers.xavier_initializer())
  conv = tf.nn.conv2d(norm1, conv2_kernel, [1, 1, 1, 1], padding='VALID')
  conv2_biases = _variable('biases', [32], tf.constant_initializer(0.0))
  pre_activation = tf.nn.bias_add(conv, conv2_biases)
  conv2 = tf.nn.relu(pre_activation, name=scope.name)

# pool2
pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1],
                       strides=[1, 2, 2, 1], padding='VALID', name='pool2')

# norm2
norm2 = tf.nn.lrn(pool2,4,bias=1.,alpha=0.001/9.,beta=0.75,name='norm2')

# conv3
with tf.variable_scope('conv3') as scope:
  conv3_kernel = _variable('weights',[3, 3, 32, 64],tf.contrib.layers.xavier_initializer())
  conv = tf.nn.conv2d(norm2, conv3_kernel, [1, 1, 1, 1], padding='VALID')
  conv3_biases = _variable('biases', [64], tf.constant_initializer(0.0))
  pre_activation = tf.nn.bias_add(conv, conv3_biases)
  conv3 = tf.nn.relu(pre_activation, name=scope.name)
  
# norm3
norm3 = tf.nn.lrn(conv3,4,bias=1.,alpha=0.001/9.,beta=0.75,name='norm3')

# linear layer(WX + b),
with tf.variable_scope('softmax_linear') as scope:
  reshape = tf.reshape(norm3, [batch,-1])
#  dim = reshape.get_shape()[1].value
  layer4_weights = _variable('weights', [576,num_class],tf.contrib.layers.xavier_initializer())
  layer4_biases = _variable('biases', [num_class],tf.constant_initializer(0.0))
  logits = tf.add(tf.matmul(reshape, layer4_weights), layer4_biases, name=scope.name)
  labels_pred = tf.nn.top_k(logits)

reg_fun = tf.contrib.layers.l1_regularizer(5e-4)
reg = tf.contrib.layers.apply_regularization(reg_fun,[conv1_kernel,conv2_kernel,conv3_kernel,layer4_weights])

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits)) + reg

train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

ema = tf.train.ExponentialMovingAverage(decay=0.9999)
maintain_averages_op = ema.apply([conv1_kernel,conv1_biases,conv2_kernel,conv2_biases,conv3_kernel,conv3_biases,layer4_weights,layer4_biases])
with tf.control_dependencies([train_step]):
  train_op = tf.group(maintain_averages_op)

num_matched = tf.reduce_sum(tf.cast(tf.equal(tf.reshape(labels_pred.indices,[-1]),tf.reshape(labels,[-1])),tf.float32))
#num_matched = tf.reduce_sum(tf.cast(tf.nn.in_top_k(logits,labels,1),tf.float32))

def error_clas(labels_,labels_pred_,clas):
  flag0 = tf.equal(labels,tf.ones([batch],dtype=tf.int32)*clas)
  return tf.reduce_sum(tf.cast(tf.equal(tf.boolean_mask(labels_pred_,flag0),tf.boolean_mask(labels_,flag0)),tf.float32))

num_matched0 = error_clas(tf.reshape(labels,[-1]),tf.reshape(labels_pred.indices,[-1]),0)
num_matched1 = error_clas(tf.reshape(labels,[-1]),tf.reshape(labels_pred.indices,[-1]),1)
num_matched2 = error_clas(tf.reshape(labels,[-1]),tf.reshape(labels_pred.indices,[-1]),2)
num_matched3 = error_clas(tf.reshape(labels,[-1]),tf.reshape(labels_pred.indices,[-1]),3)
num_matched4 = error_clas(tf.reshape(labels,[-1]),tf.reshape(labels_pred.indices,[-1]),4)
num_matched5 = error_clas(tf.reshape(labels,[-1]),tf.reshape(labels_pred.indices,[-1]),5)
num_matched6 = error_clas(tf.reshape(labels,[-1]),tf.reshape(labels_pred.indices,[-1]),6)
num_matched7 = error_clas(tf.reshape(labels,[-1]),tf.reshape(labels_pred.indices,[-1]),7)
num_matched8 = error_clas(tf.reshape(labels,[-1]),tf.reshape(labels_pred.indices,[-1]),8)
num_matched9 = error_clas(tf.reshape(labels,[-1]),tf.reshape(labels_pred.indices,[-1]),9)

saver = tf.train.Saver()
tf.get_collection('validation_nodes')
tf.add_to_collection('validation_nodes',images)
tf.add_to_collection('validation_nodes',labels_pred.indices)

init = tf.global_variables_initializer()
with tf.Session() as sess:
  sess.run(init)
  i_iter = 1
  step = 0
  for i in xrange(n_iter):
    print '%s: n_iter=%d' %(datetime.now(), i_iter)
    samples = np.random.randint(0,dim_train,batch)
    sess.run(train_op,feed_dict={images:train_x[samples],labels:train_y[samples]})
    if i_iter % log_freq == 0:
      loss_val = 0
      num_matched_train = 0
      for i in xrange(dim_train/batch):
        num_matched_train += sess.run(num_matched, feed_dict={images:train_x[i*batch:(i+1)*batch],labels:train_y[i*batch:(i+1)*batch]})
        loss_val += sess.run(loss,{images:train_x[i*batch:(i+1)*batch],labels:train_y[i*batch:(i+1)*batch]})
      print 'loss=',loss_val/(dim_train/batch)
      precision_train = num_matched_train/dim_train
      print 'train precision=',precision_train
      num_matched_test = 0
      for i in xrange(dim_test/batch):
        num_matched_test += sess.run(num_matched, feed_dict={images:test_x[i*batch:(i+1)*batch],labels:test_y[i*batch:(i+1)*batch]})
      precision_test = num_matched_test/dim_test
      print 'test precision=',precision_test
      history[step,0] = loss_val/(dim_train/batch)
      history[step,1] = precision_train
      history[step,2] = precision_test
      if precision_test > 0.71:
        break
      step += 1
      
    i_iter += 1
    
  num_matched_test_clas = np.zeros([10])
  for i in xrange(dim_test/batch):
    num_matched_test_clas[0] += sess.run(num_matched0, feed_dict={images:test_x[i*batch:(i+1)*batch],labels:test_y[i*batch:(i+1)*batch]})
    num_matched_test_clas[1] += sess.run(num_matched1, feed_dict={images:test_x[i*batch:(i+1)*batch],labels:test_y[i*batch:(i+1)*batch]})
    num_matched_test_clas[2] += sess.run(num_matched2, feed_dict={images:test_x[i*batch:(i+1)*batch],labels:test_y[i*batch:(i+1)*batch]})
    num_matched_test_clas[3] += sess.run(num_matched3, feed_dict={images:test_x[i*batch:(i+1)*batch],labels:test_y[i*batch:(i+1)*batch]})
    num_matched_test_clas[4] += sess.run(num_matched4, feed_dict={images:test_x[i*batch:(i+1)*batch],labels:test_y[i*batch:(i+1)*batch]})
    num_matched_test_clas[5] += sess.run(num_matched5, feed_dict={images:test_x[i*batch:(i+1)*batch],labels:test_y[i*batch:(i+1)*batch]})
    num_matched_test_clas[6] += sess.run(num_matched6, feed_dict={images:test_x[i*batch:(i+1)*batch],labels:test_y[i*batch:(i+1)*batch]})
    num_matched_test_clas[7] += sess.run(num_matched7, feed_dict={images:test_x[i*batch:(i+1)*batch],labels:test_y[i*batch:(i+1)*batch]})
    num_matched_test_clas[8] += sess.run(num_matched8, feed_dict={images:test_x[i*batch:(i+1)*batch],labels:test_y[i*batch:(i+1)*batch]})
    num_matched_test_clas[9] += sess.run(num_matched9, feed_dict={images:test_x[i*batch:(i+1)*batch],labels:test_y[i*batch:(i+1)*batch]})
  precision_test_clas = num_matched_test_clas/(dim_test*0.1)
  print 'test precision for each class=',precision_test_clas[:]
  conv1_filter = sess.run([conv1_kernel,conv1_biases])
  saver.save(sess,'my_model')

np.savetxt('history.txt',history)

with open('conv1.pkl','wb') as f:
  pk.dump(conv1_filter,f)