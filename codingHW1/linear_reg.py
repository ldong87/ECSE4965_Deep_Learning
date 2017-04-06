import numpy as np
#import tensorflow as tf
import matplotlib.pyplot as plt

n_sample = 10000
n_epoch = 100
n_epoch_sgd = 500
batch = [10, 15, 20]
eta = [1e-3, 2.5e-3, 5e-3] # tf:5e-3; gd:5e-7;
eta_sgd = 5e-2
tol = 1e-2

x_train = np.loadtxt('Prog1_data.txt',usecols=range(10))
x_train = np.append(x_train,np.ones((n_sample,1)),axis=1)
y_train = np.loadtxt('Prog1_data.txt',usecols=[10]) 
y_train = np.reshape(y_train,[n_sample,1])

Theta = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(x_train),x_train)),np.transpose(x_train)),y_train)
print Theta

plt.figure()
for eta_ in eta:
  Theta_gd = np.full([11,1],0.01)
  epoch = 1
  loss_val_gd = [np.linalg.norm(y_train-np.dot(x_train,Theta_gd))]
  while np.linalg.norm(Theta_gd-Theta)>tol:
    epoch = epoch + 1
    Theta_gd = Theta_gd + eta_*1/float(n_sample)*np.dot(np.transpose(x_train),y_train-np.dot(x_train,Theta_gd))
    loss_val_gd.append(np.linalg.norm(y_train-np.dot(x_train,Theta_gd)))
    if epoch == n_epoch:
      break
  plt.plot(range(1,epoch+1),loss_val_gd)
  plt.xlabel("Iteration number")
  plt.ylabel("Loss function value")
plt.legend(["Learning rate=1e-3","Learning rate=2.5e-3","Learning rate=5e-3"])
plt.savefig("gd.png")
print "Iteration used for GD: ", epoch, "   error: ", np.linalg.norm(Theta_gd-Theta), "\n", Theta_gd

plt.figure()
for batch_ in batch:
  Theta_sgd = np.full([11,1],0.01)
  epoch = 1
  loss_val_sgd = [np.linalg.norm(y_train-np.dot(x_train,Theta_sgd))]
  while np.linalg.norm(Theta_sgd-Theta)>tol:
    epoch = epoch + 1
    idx = np.random.randint(0,n_sample,batch_)
    Theta_sgd = Theta_sgd + eta_sgd*1/float(n_sample)*np.dot(np.transpose(x_train[idx]),(y_train[idx]-np.dot(x_train[idx],Theta_sgd)))
    loss_val_sgd.append(np.linalg.norm(y_train-np.dot(x_train,Theta_sgd)))
    if epoch == n_epoch_sgd: 
      break
  plt.plot(range(1,epoch+1),loss_val_sgd)
  plt.xlabel("Iteration number")
  plt.ylabel("Loss function value")
plt.legend(["Batch size=10","Batch size=15","Batch size=20"])
plt.savefig("sgd.png")
print "Iteration used for SGD: ", epoch, "   error: ", np.linalg.norm(Theta_sgd-Theta), "\n", Theta_sgd

'''
# tensorflow
cost_history = np.empty([1],dtype=float)
x = tf.placeholder(tf.float32,[None, 11])
Theta_gd = tf.Variable(tf.zeros([11,1]))
#Theta_gd = tf.Variable(np.float32(Theta))
#b = tf.Variable(tf.zeros([1]))
y = tf.matmul(x, Theta_gd)
y_ = tf.placeholder(tf.float32, [None,1])

cost = tf.reduce_mean(tf.square(y-y_))
optimizer = tf.train.GradientDescentOptimizer(eta).minimize(cost)
init = tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(init)
  for epoch in range(n_epoch):
    sess.run(optimizer,feed_dict={x:x_train,y_:y_train})
    cost_history = np.append(cost_history,sess.run(cost,feed_dict={x:x_train,y_:y_train}))
  print(sess.run(Theta_gd))

plt.plot(range(len(cost_history)),cost_history)
plt.axis([0,n_epoch,0,np.max(cost_history)])
plt.show()
'''
