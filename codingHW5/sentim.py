#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 12:41:52 2017

@author: ldong
"""
#from __future__ import print_function
import tensorflow as tf
import numpy as np
import pickle as pk
import time

tf.reset_default_graph()

data = np.load('/home/ldong/Dropbox/Coursework RPI 17Spring/Deep Learning/HW/codingHW5/train_and_val.npz')
train = [data['train_x'], data['train_mask'], data['train_y']]
valid = [data['val_x'], data['val_mask'], data['val_y']]
del data

dim_train = np.shape(train[0])[0]
dim_test = np.shape(valid[0])[0]

lr = 1.
batch = 1000
max_len = 25
vocab_len = 8745
w_embed_len = 25
cell_size = 10
out_keep_prob = 0.5

epoch = 100
n_iter = epoch*dim_train/batch + 1
log_freq = n_iter/200
history = np.zeros([np.ceil(n_iter/float(log_freq)).astype(int),3])

init = tf.contrib.layers.xavier_initializer()

inputs = tf.placeholder(tf.int64,[None,max_len])
mask = tf.placeholder(tf.float32,[None,max_len])
labels = tf.placeholder(tf.int64,[None])

# embedding layer
#w_embed = tf.Variable(tf.truncated_normal([vocab_len, w_embed_len])*0.001)
w_embed = tf.get_variable('w_embed',shape=[vocab_len, w_embed_len],initializer=init)
rnn_input = tf.nn.embedding_lookup(w_embed,inputs)

# RNN cell
cell = tf.contrib.rnn.GRUCell(cell_size)
cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=out_keep_prob)
rnn_output, _ = tf.nn.dynamic_rnn(cell,rnn_input,dtype=tf.float32)

# find last output
length = tf.cast(tf.reduce_sum(mask,1),tf.int32)
out_size = int(rnn_output.get_shape()[2])
flat = tf.reshape(rnn_output,[-1,out_size])
index = tf.range(0,batch)*max_len + length-1
outputs = tf.gather(flat,index)

# fully connected layer
w = tf.get_variable('w',[cell_size,1],initializer=init)
b = tf.Variable([0.])
logits = tf.matmul(outputs, w) + b
                  
# prediction
pred = tf.cast(tf.round(tf.reshape(tf.nn.sigmoid(logits),[-1])),tf.int64)
num_matched = tf.reduce_sum(tf.cast(tf.equal(pred, labels), tf.float32))

# loss function
loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.reshape(logits,[-1]), labels=tf.cast(labels,tf.float32)) 
loss = tf.reduce_sum(loss) / batch

# train
#train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)
train_step = tf.train.AdadeltaOptimizer(lr).minimize(loss)

# save model
saver = tf.train.Saver()
tf.get_collection('validation_nodes')
tf.add_to_collection('validation_nodes',inputs)
tf.add_to_collection('validation_nodes',mask)
tf.add_to_collection('validation_nodes',pred)

init = tf.global_variables_initializer()
with tf.Session() as sess:
  sess.run(init)
  i_iter = 0
  step = 0
  start_time = time.time()
  for i in xrange(n_iter):
    print '\r n_iter =%d / %d' %(i_iter,n_iter),
    
    samples = np.random.randint(0,dim_train,batch)
    sess.run(train_step,feed_dict={inputs:train[0][samples],mask:train[1][samples],labels:train[2][samples]})
    if i_iter % log_freq == 0:
      loss_val = 0
      num_matched_train = 0
      for i in xrange(dim_train/batch):
        num_matched_train += sess.run(num_matched, feed_dict={inputs:train[0][i*batch:(i+1)*batch],mask:train[1][i*batch:(i+1)*batch],labels:train[2][i*batch:(i+1)*batch]})
        loss_val += sess.run(loss,{inputs:train[0][i*batch:(i+1)*batch],mask:train[1][i*batch:(i+1)*batch],labels:train[2][i*batch:(i+1)*batch]})
      print '\nloss=',loss_val/(dim_train/batch)
      precision_train = num_matched_train/dim_train
      print 'train precision=',precision_train
      num_matched_test = 0
      for i in xrange(dim_test/batch):
        num_matched_test += sess.run(num_matched, feed_dict={inputs:valid[0][i*batch:(i+1)*batch],mask:valid[1][i*batch:(i+1)*batch],labels:valid[2][i*batch:(i+1)*batch]})
      precision_test = num_matched_test/dim_test
      print 'test precision=',precision_test
      history[step,0] = loss_val/(dim_train/batch)
      history[step,1] = precision_train
      history[step,2] = precision_test
#      if precision_test > 0.83:
#        break
      step += 1
      print '\n'
      
    i_iter += 1
  
  print 'Total time = %.3f'%(time.time()-start_time)
  word_embed = sess.run(w_embed)
  saver.save(sess,'my_model')

np.savetxt('history.txt',history)

with open('word_embed.pkl','wb') as f:
  pk.dump(word_embed,f)
  
  
def generate_plots(history,n_iter,interv):
    fig, ax1 = plt.subplots()
    x_axis = np.linspace(1,n_iter,np.shape(history)[0],endpoint=True,dtype=int) 
    ax1.plot(x_axis,history[:,0],'b-')
    ax1.set_xlabel('Iteration number')
    ax1.set_ylabel('Loss function value',color='b')
    ax1.tick_params('y',colors='b')
    
    ax2 = ax1.twinx()
    ax2.plot(x_axis,history[:,1],'r.')
    ax2.plot(x_axis,history[:,2],'rx')
    ax2.set_ylabel('Classifcation accuracy',color='r')
    ax2.tick_params('y',colors='r')
    ax2.legend(['accuracy train','accuracy test'])
    
    fig.tight_layout()
    plt.savefig('convergence.png',dpi=600)
    
generate_plots(history,n_iter,log_freq)