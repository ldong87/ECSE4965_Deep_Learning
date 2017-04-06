#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 12:57:35 2017

@author: ldong
"""

import tensorflow as tf
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

dim_img = 32
chanl = 3

def filter_render(kernel,biases,img_noise,ind):
  img = img_noise.copy()
  t_input = tf.placeholder(tf.float32,[1,dim_img,dim_img,chanl])
  kernel = tf.reshape(kernel,[5,5,chanl,1])
  conv = tf.nn.conv2d(t_input, kernel, [1, 1, 1, 1], padding='VALID')
  pre_activation = conv + biases
  t_score = tf.reduce_mean(tf.nn.relu(pre_activation)) # defining the optimization objective
  t_grad = tf.gradients(t_score, t_input)[0] # behold the power of automatic differentiation!
  
  iter_n=1000
  step=20.
  init = tf.global_variables_initializer()
  with tf.Session() as sess:
    sess.run(init)
    for i in range(iter_n):
        g, score = sess.run([t_grad, t_score], {t_input:img})
        # normalizing the gradient, so the same step size should work 
#        g /= g.std()+1e-8         # for different layers and networks
        img += g*step 
        
  img = (img-img.mean())/max(img.std(),1e-4)*0.1+0.5
  img = np.uint8(np.clip(img,0,1)*255)
  plt.imshow(np.squeeze(img))
  plt.axis('off')
  plt.gca().xaxis.set_major_locator(plt.NullLocator())
  plt.gca().yaxis.set_major_locator(plt.NullLocator())
  fname = './conv1_viz/%d.png' % ind
  plt.savefig(fname,bbox_inches='tight',pad_inches=0)
  
def generate_plots(history,n_iter):
    fig, ax1 = plt.subplots()
    x_axis = np.arange(1,n_iter+1,100)
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
  
with open('./conv1.pkl','rb') as f:
  conv1 = pk.load(f)
kernel = conv1[0]
biases = conv1[1]
  
# start with a gray image with a little noise
#img_noise = np.random.uniform(size=(1,32,32,3)) + 100.0
img_noise = np.random.uniform(size=(1,dim_img,dim_img,chanl)) + 100.0
                             
for i in xrange(len(biases)):
  filter_render(kernel[:,:,:,i],biases[i],img_noise,i)
  
history = np.loadtxt('./record/10k_eta0.05_wd/history.txt')
generate_plots(history,10000)