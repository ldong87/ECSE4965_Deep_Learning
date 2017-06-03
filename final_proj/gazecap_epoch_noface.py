#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 21:22:21 2017

@author: ldong
"""

import tensorflow as tf
import numpy as np
import time
import pickle as pk
import matplotlib.pyplot as plt

tf.reset_default_graph()

def img_normalize(img):
  for i_rgb in xrange(3):
    tmp = img[:,:,:,i_rgb]
    mean = np.mean(np.reshape(tmp,[img.shape[0],-1]),1)
    std = np.std(np.reshape(tmp,[img.shape[0],-1]),1)
    for i in xrange(img.shape[0]):
      img[i,:,:,i_rgb] = (img[i,:,:,i_rgb]-mean[i])/std[i]
  return img

npzfile = np.load("/home/ldong/Documents/ML/deep_learning_17Spring/proj/train_and_val.npz")

data_aug = False
data_norm = False

if data_norm:
  train_eye_left = img_normalize(npzfile["train_eye_left"])
  train_eye_right = img_normalize(npzfile["train_eye_right"])
  train_face = img_normalize(npzfile["train_face"])
  val_eye_left = img_normalize(npzfile["val_eye_left"])
  val_eye_right = img_normalize(npzfile["val_eye_right"])
  val_face = img_normalize(npzfile["val_face"])
else:
  train_eye_left = npzfile["train_eye_left"]
  train_eye_right = npzfile["train_eye_right"]
  train_face = npzfile["train_face"]
  val_eye_left = npzfile["val_eye_left"]
  val_eye_right = npzfile["val_eye_right"]
  val_face = npzfile["val_face"]

train_face_mask = npzfile["train_face_mask"]
train_y = npzfile["train_y"]
val_face_mask = npzfile["val_face_mask"]
val_y = npzfile["val_y"]

del npzfile

batch = 250
channel = 3
dim_eye = np.shape(train_eye_left[0])
dim_face = np.shape(train_face[0])
dim_fg = np.shape(train_face_mask[0])
dim_y = np.shape(train_y[0])
dim_train = np.shape(train_y)[0]
dim_test = np.shape(val_y)[0]

epoch = 150
lr = 1e-3
reg_par = 5e-4
lr_decay_cycle = 5
decay_res = 0.005
history = np.zeros([epoch,4]) # loss, train error, test error, learning rate

eye_l = tf.placeholder(tf.float32, [batch,dim_eye[0],dim_eye[1],dim_eye[2]], "eye_left")
eye_r = tf.placeholder(tf.float32, [batch,dim_eye[0],dim_eye[1],dim_eye[2]], 'eye_right')
face = tf.placeholder(tf.float32, [batch,dim_face[0],dim_face[1],dim_face[2]], 'face')
fg = tf.placeholder(tf.float32, [batch,dim_fg[0],dim_fg[1]], 'face_grid')
y = tf.placeholder(tf.float32, [batch,dim_y[0]], 'target')
keep = tf.placeholder(tf.float32, [], 'keep_prob')

init = tf.contrib.layers.xavier_initializer(uniform=False)

#%% data augmentation, alter RGB
def rgb_perturb_dump():
  def rgb_pca(img):
    img_r = np.reshape(img[:,:,:,0],[-1])
    img_g = np.reshape(img[:,:,:,1],[-1])
    img_b = np.reshape(img[:,:,:,2],[-1])
    cov = np.cov(np.array([img_r,img_g,img_b]))
    [eigval, eigvec] = np.linalg.eig(cov)
    return [eigval, eigvec]
  
  [eye_eigval, eye_eigvec] = rgb_pca(np.concatenate((train_eye_left,train_eye_right)))
  [face_eigval, face_eigvec] = rgb_pca(train_face) 
  with open('/home/ldong/Dropbox/Coursework RPI 17Spring/Deep Learning/proj/rgb_eig.pkl','wb') as f:
    pk.dump([eye_eigval, eye_eigvec, face_eigval, face_eigvec],f)

with open('/home/ldong/Dropbox/Coursework RPI 17Spring/Deep Learning/proj/rgb_eig.pkl','rb') as f:
      eye_eigval, eye_eigvec, face_eigval, face_eigvec = pk.load(f)
  
def rgb_perturb(eigval,eigvec,img_batch):
  alpha = np.random.normal(0,0.1,3)
  rgb_add = np.inner(eigvec,alpha*eigval)
  img_batch = np.concatenate((np.expand_dims(img_batch[:,:,:,0],axis=3)+rgb_add[0],
                              np.expand_dims(img_batch[:,:,:,1],axis=3)+rgb_add[1],
                              np.expand_dims(img_batch[:,:,:,2],axis=3)+rgb_add[2]),axis=3)
  return img_batch

#%% eye left
# layer 1
with tf.name_scope('Conv_E1_Left') as scope:
  conv_e1_kernel = tf.get_variable('Conv_E1_Filter',[11,11,channel,96],initializer=init)
  conv_e1_wx_l = tf.nn.conv2d(eye_l, conv_e1_kernel, [1, 4, 4, 1], padding='VALID')
  conv_e1_b = tf.Variable(tf.zeros([96]))
  relu_e1_l = tf.nn.relu(tf.nn.bias_add(conv_e1_wx_l, conv_e1_b), name='ReLU_E1')
  pool_e1_l = tf.nn.max_pool(relu_e1_l, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='Pool_E1')
  norm_e1_l = tf.nn.lrn(pool_e1_l,5,alpha=0.0001, beta=0.75, name='Norm_E1')

# layer 2
with tf.name_scope('Conv_E2_Left'):
  conv_e2_kernel = tf.get_variable('Conv_E2_Filter',[5,5,96,256],initializer=init)
  conv_e2_wx_l = tf.nn.conv2d(norm_e1_l, conv_e2_kernel, [1, 1, 1, 1], padding='SAME')
  conv_e2_b = tf.Variable(tf.zeros([256]))
  relu_e2_l = tf.nn.relu(tf.nn.bias_add(conv_e2_wx_l,conv_e2_b), name='ReLU_E2')
  pool_e2_l = tf.nn.max_pool(relu_e2_l, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='Pool_E2')
  norm_e2_l = tf.nn.lrn(pool_e2_l,5,alpha=0.0001, beta=0.75, name='Norm_E2')

# layer 3
with tf.name_scope('Conv_E3_Left'):
  conv_e3_kernel = tf.get_variable('Conv_E3_Filter',[3,3,256,384],initializer=init)
  conv_e3_wx_l = tf.nn.conv2d(norm_e2_l, conv_e3_kernel, [1, 1, 1, 1], padding='SAME')
  conv_e3_b = tf.Variable(tf.zeros([384]))
  relu_e3_l = tf.nn.relu(tf.nn.bias_add(conv_e3_wx_l,conv_e3_b), name='ReLU_E3')

# layer 4
with tf.name_scope('Conv_E4_Left'):
  conv_e4_kernel = tf.get_variable('Conv_E4_Filter',[1,1,384,64],initializer=init)
  conv_e4_wx_l = tf.nn.conv2d(relu_e3_l, conv_e4_kernel, [1, 1, 1, 1], padding='VALID')
  conv_e4_b = tf.Variable(tf.zeros([64]))
  relu_e4_l = tf.nn.relu(tf.nn.bias_add(conv_e4_wx_l,conv_e4_b), name='ReLU_E4')

#%% eye right, share parameters with eye left
# layer 1
with tf.name_scope('Conv_E1_Right'):
  conv_e1_wx_r = tf.nn.conv2d(eye_r, conv_e1_kernel, [1, 4, 4, 1], padding='VALID')
  relu_e1_r = tf.nn.relu(tf.nn.bias_add(conv_e1_wx_r, conv_e1_b), name='ReLU_E1')
  pool_e1_r = tf.nn.max_pool(relu_e1_r, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='Pool_E1')
  norm_e1_r = tf.nn.lrn(pool_e1_r,5,alpha=0.0001, beta=0.75, name='Norm_E1')

# layer 2
with tf.name_scope('Conv_E2_Right'):
  conv_e2_wx_r = tf.nn.conv2d(norm_e1_r, conv_e2_kernel, [1, 1, 1, 1], padding='SAME')
  relu_e2_r = tf.nn.relu(tf.nn.bias_add(conv_e2_wx_r,conv_e2_b), name='ReLU_E2')
  pool_e2_r = tf.nn.max_pool(relu_e2_r, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='Pool_E2')
  norm_e2_r = tf.nn.lrn(pool_e2_r,5,alpha=0.0001, beta=0.75, name='Norm_E2')

# layer 3
with tf.name_scope('Conv_E3_Right'):
  conv_e3_wx_r = tf.nn.conv2d(norm_e2_r, conv_e3_kernel, [1, 1, 1, 1], padding='SAME')
  relu_e3_r = tf.nn.relu(tf.nn.bias_add(conv_e3_wx_r,conv_e3_b), name='ReLU_E3')

# layer 4
with tf.name_scope('Conv_E4_Right'):
  conv_e4_wx_r = tf.nn.conv2d(relu_e3_r, conv_e4_kernel, [1, 1, 1, 1], padding='VALID')
  relu_e4_r = tf.nn.relu(tf.nn.bias_add(conv_e4_wx_r,conv_e4_b), name='ReLU_E4')
  
#%% FC eye
with tf.name_scope('FC_E1'):
  fc_e1_x_l = tf.reshape(relu_e4_l,[batch,-1])
  fc_e1_x_r = tf.reshape(relu_e4_r,[batch,-1])
  fc_e1_x = tf.concat([fc_e1_x_l,fc_e1_x_r],1)
  fc_e1_w = tf.get_variable('FC_E1_W',[np.shape(fc_e1_x)[1],128],initializer=init)
  fc_e1_b = tf.Variable(tf.ones([128]))
  fc_e1 = tf.nn.bias_add(tf.matmul(fc_e1_x,fc_e1_w),fc_e1_b)
  relu_fc_e1 = tf.nn.relu(fc_e1,name='ReLU_FC_E1')
  drop_fc_e1 = tf.nn.dropout(relu_fc_e1,1.,name='Dropout_FC_E1')
  
#%% FC facegrid
with tf.name_scope('FC_FG1'):
  fc_fg1_x = tf.reshape(fg,[batch,-1])
  fc_fg1_w = tf.get_variable('FC_FG1_W',[np.shape(fc_fg1_x)[1],256],initializer=init)
  fc_fg1_b = tf.Variable(tf.ones([256]))
  fc_fg1 = tf.nn.bias_add(tf.matmul(fc_fg1_x,fc_fg1_w),fc_fg1_b)
  relu_fc_fg1 = tf.nn.relu(fc_fg1,name='ReLU_FC_FG1')
  drop_fc_fg1 = tf.nn.dropout(relu_fc_fg1,1.,name='Dropout_FC_FG1')
  
with tf.name_scope('FC_FG2'):
  fc_fg2_w = tf.get_variable('FC_FG2_W',[np.shape(drop_fc_fg1)[1],128],initializer=init)
  fc_fg2_b = tf.Variable(tf.ones([128]))
  fc_fg2 = tf.nn.bias_add(tf.matmul(drop_fc_fg1,fc_fg2_w),fc_fg2_b)
  relu_fc_fg2 = tf.nn.relu(fc_fg2,name='ReLU_FC_FG2')
  drop_fc_fg2 = tf.nn.dropout(relu_fc_fg2,1.,name='Dropout_FC_FG2')
  
#%% total FC
with tf.name_scope('FC1'):
  fc1_x = tf.concat([drop_fc_e1,drop_fc_fg2],1)
  fc1_w = tf.get_variable('FC1_W',[np.shape(fc1_x)[1],128],initializer=init)
  fc1_b = tf.Variable(tf.ones([128]))
  fc1 = tf.nn.bias_add(tf.matmul(fc1_x,fc1_w),fc1_b)
  relu_fc1 = tf.nn.relu(fc1,name='ReLU_FC1')
  drop_fc1 = tf.nn.dropout(relu_fc1,1., name='Dropout_FC1')
  
with tf.name_scope('FC2'):
  fc2_w = tf.get_variable('FC2_W',[np.shape(drop_fc1)[1],2],initializer=init)
  fc2_b = tf.Variable(tf.zeros([2]))
  fc2 = tf.nn.bias_add(tf.matmul(drop_fc1,fc2_w),fc2_b)

#%%
global_step = tf.Variable(0,trainable=False)

#%%
lrate = tf.placeholder(tf.float32,[])
  
reg_fun = tf.contrib.layers.l2_regularizer(reg_par)
reg = tf.contrib.layers.apply_regularization(reg_fun,tf.trainable_variables())
mismatch = tf.reduce_mean(tf.norm(fc2-y,axis=1))
loss = mismatch + reg
train_step = tf.train.AdamOptimizer(lrate).minimize(loss,global_step)
ema = tf.train.ExponentialMovingAverage(decay=0.9999)
maintain_averages_op = ema.apply(tf.trainable_variables())
with tf.control_dependencies([train_step]):
  train_op = tf.group(maintain_averages_op)
  
#%% 
saver = tf.train.Saver()
tf.get_collection("validation_nodes")
tf.add_to_collection("validation_nodes", eye_l)
tf.add_to_collection("validation_nodes", eye_r)
tf.add_to_collection("validation_nodes", face)
tf.add_to_collection("validation_nodes", fg)
tf.add_to_collection("validation_nodes", fc2)

#$$
#config = tf.ConfigProto(device_count = {'GPU': 0})
  
#%%
init_all = tf.global_variables_initializer()
#with tf.Session(config=config) as sess: # no gpu
with tf.Session() as sess:
  sess.run(init_all)
  i_epoch = 0
  start_time = time.time()
  for i in xrange(epoch):  
    i_start_time = time.time()      
    print '\rn_epoch = %d / %d, lr = %e' %(i_epoch,epoch-1,lr),
    arr = np.arange(dim_train)
    np.random.shuffle(arr)
    samples = np.reshape(arr, [dim_train/batch,-1])
    
    if data_aug:
      # data augmentation on the fly, alter rgb
      train_eye_left_aug = rgb_perturb(eye_eigval,eye_eigvec,train_eye_left)
      train_eye_right_aug = rgb_perturb(eye_eigval,eye_eigvec,train_eye_right)
      train_face_aug = rgb_perturb(face_eigval,face_eigvec,train_face)
    else:
      train_eye_left_aug = train_eye_left
      train_eye_right_aug = train_eye_right
      train_face_aug = train_face
            
    for i in xrange(dim_train/batch):  
      sess.run(train_op,feed_dict={eye_l:train_eye_left_aug[samples[i]],eye_r:train_eye_right_aug[samples[i]],face:train_face_aug[samples[i]],fg:train_face_mask[samples[i]],y:train_y[samples[i]],lrate:lr})
    mismatch_train = 0
    num_train_eval = dim_train/batch
    for i in xrange(num_train_eval):
      mismatch_train += sess.run(mismatch,feed_dict={eye_l:train_eye_left[samples[i]],eye_r:train_eye_right[samples[i]],face:train_face[samples[i]],fg:train_face_mask[samples[i]],y:train_y[samples[i]]})
    precision_train = mismatch_train/num_train_eval
    loss_val = precision_train + sess.run(reg)
    print '\nloss=',loss_val
    print 'train precision=',precision_train
    mismatch_test = 0
    for i in xrange(dim_test/batch):
      mismatch_test += sess.run(mismatch,feed_dict={eye_l:val_eye_left[i*batch:(i+1)*batch],eye_r:val_eye_right[i*batch:(i+1)*batch],face:val_face[i*batch:(i+1)*batch],fg:val_face_mask[i*batch:(i+1)*batch],y:val_y[i*batch:(i+1)*batch]})
    precision_test = mismatch_test/(dim_test/batch)
    print 'test precision=',precision_test

    history[i_epoch,0] = loss_val
    history[i_epoch,1] = precision_train
    history[i_epoch,2] = precision_test
    history[i_epoch,3] = lr
    
    if ((i_epoch+1) % lr_decay_cycle) and i_epoch+1>=2*lr_decay_cycle:
      val_curr_step = np.mean(history[i_epoch-lr_decay_cycle-1:i_epoch+1,2])
      val_last_step = np.mean(history[i_epoch-2*lr_decay_cycle-1:i_epoch-lr_decay_cycle-1,2])
      if val_curr_step > val_last_step or abs(val_curr_step - val_last_step)<decay_res:
        lr = lr*0.5
    
#      if precision_test < 1.748:
#        break

    i_epoch += 1
    print 'epoch time = %.3f'%(time.time()-i_start_time)
    print '\n'
        
  print 'Total time = %.3f'%(time.time()-start_time)
  saver.save(sess,'my-model')
  
  writer = tf.summary.FileWriter('log', sess.graph)
  writer.add_graph(sess.graph)

np.savetxt('history.txt',history)
  
#%%
def generate_plots(history,n_iter):
    fig, ax1 = plt.subplots()
    x_axis = np.linspace(1,n_iter,np.shape(history)[0],endpoint=True,dtype=int) 
    ax1.plot(x_axis,history[:,0],'b-')
    ax1.set_xlabel('Epoch number')
    ax1.set_ylabel('Loss function value',color='b')
    ax1.tick_params('y',colors='b')
    
    ax2 = ax1.twinx()
    ax2.plot(x_axis,history[:,1],'r.')
    ax2.plot(x_axis,history[:,2],'rx')
    ax2.set_ylabel('Regression accuracy',color='r')
    ax2.tick_params('y',colors='r')
    ax2.legend(['accuracy train','accuracy test'])
    
    fig.tight_layout()
    plt.savefig('convergence.png',dpi=600)
    
    plt.figure()
    plt.plot(x_axis,history[:,3])
    plt.xlabel('Epoch number')
    plt.ylabel('Learning rate')
    plt.tight_layout()
    plt.savefig('learning_rate.png',dpi=600)
    
    
generate_plots(history,epoch)
  
  
  
  