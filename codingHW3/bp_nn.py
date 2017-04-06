# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
import os, os.path
import pickle as pk
import matplotlib.pyplot as plt
import scipy.stats as stats


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
        y_vec[r][y[r][0]] = 1
    np.savetxt(path+"labels/"+data+"_label_vec.txt",y_vec,fmt="%i")
    
def forward_prop(W1,W2,W3,x,y,dim_data):
        x = np.concatenate((x,np.ones([dim_data,1])),axis=1) # add bias
        
        z1 = np.matmul(x,W1)
        H1 = np.maximum(z1,0)
        
        z2 = np.matmul(np.concatenate((H1,np.ones([dim_data,1])),1),W2)
        H2 = np.maximum(z2,0)
        
        z3 = np.matmul(np.concatenate((H2,np.ones([dim_data,1])),1),W3)
        y_ = np.zeros(np.shape(y))
        for m in xrange(dim_data):
            y_[m] = np.exp(z3[m]-np.max(z3[m]))/np.sum(np.exp(z3[m]-np.max(z3[m])))
        
#        np.savetxt('./y_.txt',np.argmax(y_,1),fmt="%i")
        
        loss = 1/float(dim_data)*0.5*np.sum(np.multiply(y_-y,y_-y))
        error = np.mean(np.equal(np.argmax(y_,1).astype('int'),np.argmax(y,1)).astype('float'))
        error_digit = np.zeros([10])
        for d in xrange(10):
            error_digit[d] = np.mean(np.equal(np.argmax(y_[np.equal(np.argmax(y,1),d)],1),d).astype('float'))
        return [loss, error, error_digit]
    
def generate_weights_from_training():
    
    if flagTF == 0:
        
        trunc_norm = stats.truncnorm(-2,2)
        
        W1 = trunc_norm.rvs([dim_in,dim_layer1])*0.01 
        W1 = np.concatenate((W1,np.zeros([1,dim_layer1])),axis=0)
        W2 = trunc_norm.rvs([dim_layer1,dim_layer2])*0.01 
        W2 = np.concatenate((W2,np.zeros([1,dim_layer2])),axis=0)
        W3 = trunc_norm.rvs([dim_layer2,dim_out])*0.01
        W3 = np.concatenate((W3,np.zeros([1,dim_out])),axis=0)
                    
        for _ in xrange(n_iter):
            
            sample = np.random.randint(0,dim_data,batch)
            x = data_x[sample]
            x = np.concatenate((x,np.ones([batch,1])),axis=1) # add bias
            y = data_y[sample]
            
            z1 = np.matmul(x,W1)
            H1 = np.maximum(z1,0)
            
            z2 = np.matmul(np.concatenate((H1,np.ones([batch,1])),1),W2)
            H2 = np.maximum(z2,0)
            
            z3 = np.matmul(np.concatenate((H2,np.ones([batch,1])),1),W3)
            y_ = np.zeros(np.shape(y))
            for m in xrange(batch):
                y_[m] = np.exp(z3[m]-np.max(z3[m]))/np.sum(np.exp(z3[m]-np.max(z3[m])))    
            
            D_sigM_ik = np.zeros((batch,dim_out,dim_out))
            for m in xrange(batch):
                D_sigM_ik[m] = -np.multiply.outer(np.transpose(y_[m]),y_[m])
                np.fill_diagonal(D_sigM_ik[m],np.squeeze(np.multiply(y_[m],1-y_[m])))
            dy_D_sigM = np.zeros((batch,dim_out))
            for m in xrange(batch):
                dy_D_sigM[m] = np.transpose(np.matmul((y_[m]-y[m]),D_sigM_ik[m]))
            g_W3 = 1/float(batch)*np.matmul(np.concatenate((H2,np.ones([batch,1])),1).T,dy_D_sigM)
            
            g_H2 = np.zeros((batch,dim_layer2))
            for m in xrange(batch):
                g_H2[m] = np.matmul(np.multiply(y_[m],(y_[m]-y[m])),np.transpose(W3[0:-1,0:])-np.matmul(y_[m],np.transpose(W3[0:-1,0:])))
            
            D_phi2_ik = np.zeros([batch,dim_layer2,dim_layer2])
            for m in xrange(batch):
                np.fill_diagonal(D_phi2_ik[m],np.multiply(np.ones(dim_layer2),np.squeeze(np.not_equal(np.maximum(z2[m],0),0.).astype('float'))))
            dH2_D_phi2 = np.zeros((batch,dim_layer2))
            for m in xrange(batch):
                dH2_D_phi2[m] = np.transpose(np.matmul(g_H2[m],D_phi2_ik[m]))
            g_W2 = 1/float(batch)*np.matmul(np.concatenate((H1,np.ones([batch,1])),1).T,dH2_D_phi2)
            
            g_H1 = np.matmul(g_H2,W2[0:-1,0:])
            
            D_phi1_ik = np.zeros([batch,dim_layer1,dim_layer1])
            for m in xrange(batch):
                np.fill_diagonal(D_phi1_ik[m],np.multiply(np.ones(dim_layer1),np.squeeze(np.not_equal(np.maximum(z1[m],0),0.).astype('float'))))
            dH1_D_phi1 = np.zeros((batch,dim_layer1))
            for m in xrange(batch):
                dH1_D_phi1[m] = np.transpose(np.matmul(g_H1[m],D_phi1_ik[m]))
            g_W1 = 1/float(batch)*np.matmul(np.array(x).T,dH1_D_phi1)
            
            W1 = W1 - eta*np.squeeze(g_W1) 
            W2 = W2 - eta*np.squeeze(g_W2)
            W3 = W3 - eta*np.squeeze(g_W3)
            
            [loss, error_train, _] = forward_prop(W1,W2,W3,data_x,data_y,dim_data)
            [_, error_test, error_digit] = forward_prop(W1,W2,W3,data_x_test,data_y_test,dim_test)
            print "Loss function value = ",loss, "accuracy train = ", error_train,"accuracy test = ",error_test
            history[0] = np.append(history[0], loss)
            history[1] = np.append(history[1], error_train)
            history[2] = np.append(history[2], error_test)
            
            if _ % output_interval == 0:
                    generate_plots(history,_+1)
            
        print "Loss function value = ",loss, "accuracy train = ", error_train,"accuracy test = ",error_test, "accuracy digits = ",error_digit
        Theta = [np.transpose(W1), np.transpose(W2), np.transpose(W3)]
    else:
        tf.reset_default_graph()

        x = tf.placeholder(tf.float32,[None,dim_in])
            
        W1 = tf.Variable(tf.truncated_normal([dim_in,dim_layer1])*0.01) # plus bias
        b1 = tf.Variable(tf.zeros([dim_layer1]))
        z1 = tf.matmul(x,W1) + b1
        H1 = tf.nn.relu(z1)
        
        W2 = tf.Variable(tf.truncated_normal([dim_layer1,dim_layer2])*0.01) # plus bias
        b2 = tf.Variable(tf.zeros([dim_layer2]))
        z2 = tf.matmul(H1,W2) + b2
        H2 = tf.nn.relu(z2)
        
        W3 = tf.Variable(tf.truncated_normal([dim_layer2,dim_out])*0.01) # plus bias
        b3 = tf.Variable(tf.zeros([dim_out]))
        z3 = tf.matmul(H2,W3) + b3
        y_ = tf.nn.softmax(z3)
        
        y = tf.placeholder(tf.float32,[None,dim_out])
        loss = 0.5*tf.reduce_mean(tf.reduce_sum(tf.multiply(y_-y,y_-y),axis=1))

        
        error = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(y_,1),tf.argmax(y,1))))
        
        train_step = tf.train.GradientDescentOptimizer(eta).minimize(loss)
        
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            for _ in range(n_iter):
                sample = np.random.randint(0,dim_data,batch)
                sess.run(train_step,feed_dict={x:data_x[sample],y:data_y[sample]})
                loss_val = sess.run(loss,feed_dict={x:data_x,y:data_y})
                error_train = sess.run(error,feed_dict={x:data_x,y:data_y})
                error_test = sess.run(error,feed_dict={x:data_x_test,y:data_y_test})
                print "Loss function value = ",loss_val, "accuracy train = ", error_train,"accuracy test = ",error_test
                history[0] = np.append(history[0], loss_val)
                history[1] = np.append(history[1], error_train)
                history[2] = np.append(history[2], error_test)
                if _ % output_interval == 0:
                    generate_plots(history,_+1)
            print "Loss function value = ",loss_val, "accuracy train = ", error_train,"accuracy test = ",error_test
                
            W1_ = sess.run(W1,feed_dict={x:data_x,y:data_y})
            b1_ = sess.run(b1,feed_dict={x:data_x,y:data_y})
            W2_ = sess.run(W2,feed_dict={x:data_x,y:data_y})
            b2_ = sess.run(b2,feed_dict={x:data_x,y:data_y})
            W3_ = sess.run(W3,feed_dict={x:data_x,y:data_y})
            b3_ = sess.run(b3,feed_dict={x:data_x,y:data_y})
            W1_ = np.concatenate((W1_,np.reshape(b1_,[1,dim_layer1])))
            W2_ = np.concatenate((W2_,np.reshape(b2_,[1,dim_layer2])))
            W3_ = np.concatenate((W3_,np.reshape(b3_,[1,dim_out])))
        [_, error_test, error_digit] = forward_prop(W1_,W2_,W3_,data_x_test,data_y_test,dim_test)
        print "Loss function value = ",history[0][-1], "accuracy train = ", history[1][-1],"accuracy test = ",error_test, "accuracy digit = ",error_digit
        Theta = [np.transpose(W1_), np.transpose(W2_), np.transpose(W3_)]
        
    with open("./nn_parameters.txt","wb") as f:
        pk.dump(Theta,f,protocol=2)
        
def generate_plots(history,n_iter):
    history = np.delete(history,[0],axis=1)

    fig, ax1 = plt.subplots()
    ax1.plot(range(n_iter),history[0],'b-')
    ax1.set_xlabel('Iteration number')
    ax1.set_ylabel('Loss function value',color='b')
    ax1.tick_params('y',colors='b')
    
    ax2 = ax1.twinx()
    ax2.plot(range(n_iter),history[1],'r.')
    ax2.plot(range(n_iter),history[2],'rx')
    ax2.set_ylabel('Classifcation accuracy',color='r')
    ax2.tick_params('y',colors='r')
    ax2.legend(['accuracy train','accuracy test'])
    
    fig.tight_layout()
    plt.show()



flagTF = 0
n_iter = 3000
dim_in = 28*28
dim_layer1 = 100
dim_layer2 = 100
dim_out = 10
eta = 1.  #25.
batch = 50
output_interval = 50

dim_tmp = 9
dim_train = 50000
dim_test = 5000

#path = "/home/ldong/Documents/ML/deep_learning_17Spring/codingHW3/"
#data = "test"  
#generate_input(dim_test,path,data)
#generate_label_vector(dim_out,path,data,dim_test)

history = np.zeros([3]).tolist() # loss_val, train_error, test_error

#data_x = np.loadtxt("./tmp_data/tmp_data.txt")
#dim_data = len(data_x)
#data_y = np.reshape(np.loadtxt("./labels/tmp_label_vec.txt"), [dim_tmp, dim_out])

data_x = np.loadtxt("./train_data/train_data.txt")
dim_data = len(data_x)
data_y = np.reshape(np.loadtxt("./labels/train_label_vec.txt"), [dim_train, dim_out])

data_x = data_x/255. # normalize
data_x_test = np.loadtxt("./test_data/test_data.txt")
data_y_test = np.reshape(np.loadtxt("./labels/test_label_vec.txt"), [dim_test, dim_out])

generate_weights_from_training()

generate_plots(history,n_iter)