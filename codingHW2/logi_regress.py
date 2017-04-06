#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 23:41:01 2017

@author: ldong
"""

import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os, os.path
import pickle as pk

dim_x = 28*28
dim_y = 5
size_tmp = 2
size_train = 25112
size_test = 4982
n_epoch = 2000
tol = 5e-5


def generate_input(dim_x,path,data):
    path = path+data+"_data/"
    X = np.zeros([1, dim_x]) # dummy init
    
    for f in sorted(os.listdir(path)):
      ext = os.path.splitext(f)[1]
      if ext.lower() == ".jpg":
        print f
        image = mpimg.imread(path+f)
        X = np.append(X,np.reshape(image,[1,dim_x]),axis=0)
        
    X = np.delete(X,0,0) # remove dummy init
    np.savetxt(path+data+"_data.txt",X)
    
def generate_label_vector(dim_y,path,data,size):
    y = np.reshape(np.loadtxt(path+"labels/"+data+"_label.txt",dtype="int"), [size, 1])
    y_vec = np.zeros([size,dim_y])
    for r in range(size):
        y_vec[r][y[r][0]-1] = 1
    np.savetxt(path+"labels/"+data+"_label_vec.txt",y_vec,fmt="%i")
    
def generate_weights_from_training(dim_x,dim_y,data,size,alpha,eta,batch):
    cost_history = np.empty([1],dtype=float)*999
    X = np.loadtxt(path+data+"_data/"+data+"_data.txt")
    y_train = np.reshape(np.loadtxt(path+"labels/"+data+"_label_vec.txt"), [size, dim_y])
    X = X/255 # normalize
    X = np.append(X,np.ones([size,1]),axis=1) # add bias
    X_test = np.loadtxt(path+"test_data/test_data.txt")
    y_test = np.reshape(np.loadtxt(path+"labels/test_label_vec.txt"), [size_test, dim_y])
    
    error_train = np.empty([1],dtype=float)*999
    error_test = np.empty([1],dtype=float)*999
    
#    # create subset for cross validation
#    size_cv = size/k_fold
#    X = X[size_cv*kth_fold:size_cv*(kth_fold+1)]
#    y_train = y_train[size_cv*kth_fold:size_cv*(kth_fold+1)]
    
    x = tf.placeholder(tf.float32, [None, dim_x+1])
#    W = tf.Variable(tf.truncated_normal([dim_x+1, dim_y]))
#    with open(path+"train_W_GD.txt","r") as f:
#        W_prev = pk.load(f)
#    W = tf.Variable(W_prev)
    W = tf.Variable(tf.zeros([dim_x+1, dim_y]))
    y = tf.matmul(x, W)
    y_eval = tf.Variable(tf.zeros([size,dim_y]))
    y_update = tf.assign(y_eval,y)
    
    y_ = tf.placeholder(tf.float32, [None, dim_y])
    cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)) +\
            alpha*tf.nn.l2_loss(W)
    train_step = tf.train.GradientDescentOptimizer(eta).minimize(cross_entropy)
    g_a = -1/float(size)*tf.matmul(tf.transpose(x),tf.sub(y_,tf.nn.softmax(y))) + alpha*W
    g_tf = tf.squeeze(tf.gradients(cross_entropy,W)) 
    W_update = tf.assign(W,W-eta*g_a)

    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epoch):
            idx = np.random.randint(0,size,batch)
            print epoch
#            sess.run(train_step,feed_dict={x:X[idx],y_:y_train[idx]})
            sess.run(W_update,feed_dict={x:X[idx],y_:y_train[idx]})
# debug            print sess.run(W)
            cost_history = np.append(cost_history, sess.run(cross_entropy,feed_dict={x:X,y_:y_train}))
            print "Loss fun. val=",cost_history[-1]
            sess.run(y_update,feed_dict={x:X})
            error_train = np.append(error_train,generate_error(sess.run(y_eval),data,y_train))
            error_test = np.append(error_test,generate_test_error(X_test,sess.run(W),y_test))
            if error_test[-1]<0.046: break
#            if epoch>1 and (abs(cost_history[-1]-cost_history[-2])+abs(cost_history[-2]-cost_history[-3]))/2.0<tol: break

        W_ = sess.run(W)
    
    print "Final loss fun. val=",cost_history[-1]
    cost_history = np.delete(cost_history,[0])
    error_train = np.delete(error_train,[0])
    error_test = np.delete(error_test,[0])
    
    with open(path+"train_error.txt","wb") as f:
        pk.dump(error_train,f)
        
    with open(path+"test_error.txt","wb") as f:
        pk.dump(error_test,f)
        
    with open(path+"loss_fun.txt","wb") as f:
        pk.dump(cost_history,f)
            
    with open(path+data+"_W.txt","wb") as f:
        pk.dump(W_,f)
    
    plt.figure()
    np.delete(error_train,[0])
    np.delete(error_test,[0])
    plt.plot(range(epoch+1),error_train)
    plt.plot(range(epoch+1),error_test)
    plt.legend(["error_train","error_test"])
    plt.axis([0,epoch,0,np.max([np.max(error_train),np.max(error_test)])])
    plt.xlabel("Iteration")
    plt.ylabel("Average error")
    
    plt.figure()
    plt.plot(range(epoch+1),cost_history)
    plt.axis([0,epoch,0,np.max(cost_history)])
    plt.xlabel("Iteration")
    plt.ylabel("Loss function value")
    plt.show()
    
def generate_error(y,data,y_label_vec):
    y_label_ = np.argmax(y_label_vec,axis=1)+1
    y_label = np.argmax(y,axis=1)+1
    idx_1 = y_label_==1
    idx_2 = y_label_==2
    idx_3 = y_label_==3
    idx_4 = y_label_==4
    idx_5 = y_label_==5
    err_1 = np.mean(y_label[idx_1]!=y_label_[idx_1])
    err_2 = np.mean(y_label[idx_2]!=y_label_[idx_2])
    err_3 = np.mean(y_label[idx_3]!=y_label_[idx_3])
    err_4 = np.mean(y_label[idx_4]!=y_label_[idx_4])
    err_5 = np.mean(y_label[idx_5]!=y_label_[idx_5])
    err_avg = np.mean(y_label!=y_label_)
    print data," data: Error average=", err_avg,"\n", \
     "error 1=", err_1, ",error 2=", err_2,",error 3=", err_3,",error 4=", err_4,",error 5=", err_5
    return err_avg

def generate_test_error(X,W,y_test):
    X = X/255
    X = np.append(X,np.ones([size_test,1]),1)
    y = np.matmul(X,W)
    return generate_error(y,"test",y_test)
    
def generate_plot(path,fileW):
    with open(path+"train_W"+fileW+".txt","r") as f:
        W = pk.load(f)
        
    for i in range(5):
        plt.imshow(np.delete(W[:,i],-1).reshape(28,28))
        plt.colorbar()
        plt.show()

path = "/home/ldong/Documents/ML/deep_learning_17Spring/codingHW2/"
data = "train"  
size = size_train               
#generate_input(dim_x,path,data)
#generate_label_vector(dim_y,path,data,size)

alpha = 0
eta = 1
batch = size_train
generate_weights_from_training(dim_x,dim_y,"train",size_train,alpha,eta,batch)

fileW= ""
generate_plot(path,fileW)


