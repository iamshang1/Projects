import tensorflow as tf
import numpy as np
from sklearn.cross_validation import train_test_split
import random
import time
import sys

#load training data
args = (sys.argv)
num_gpus = int(args[1])
batch_size = int(args[2])
input = np.load('X_train.npy')  
input = input.transpose(0,2,3,1)
labels = np.genfromtxt('./data/y_train.txt')

#cross validation splitting
X_train, X_test, y_train, y_test = train_test_split(input, labels, test_size=0.1, random_state=42, stratify=labels)

#one hot encoding for labels
def one_hot(y):
    retVal = np.zeros((len(y), 10))
    retVal[np.arange(len(y)), y.astype(int)] = 1
    return retVal

y_train = one_hot(y_train)
y_test = one_hot(y_test)

#load test data
X_full = np.load('X_test.npy')
X_full = X_full.transpose(0,2,3,1)

#model parameters
noOfIterations = 80000
image_size = 32
num_channels = 3
num_labels = 10

#layer initialization functions
def conv_ortho_weights(filter_h,filter_w,chan_in,chan_out):
    bound = np.sqrt(6./(chan_in*filter_h*filter_w + chan_out*filter_h*filter_w))
    W = np.random.random((chan_out, chan_in * filter_h * filter_w))
    u, s, v = np.linalg.svd(W,full_matrices=False)
    if u.shape[0] != u.shape[1]:
        W = u.reshape((filter_h, filter_w, chan_in, chan_out))
    else:
        W = v.reshape((filter_h, filter_w, chan_in, chan_out))
    return W.astype(np.float32)

def dense_ortho_weights(fan_in,fan_out):
    bound = np.sqrt(2./(fan_in+fan_out))
    W = np.random.randn(fan_in,fan_out)*bound
    u, s, v = np.linalg.svd(W,full_matrices=False)
    if u.shape[0] != u.shape[1]:
        W = u
    else:
        W = v
    return W.astype(np.float32)
    
def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return initial

def conv2d(x, W, stride=[1, 1, 1, 1]):
    return tf.nn.conv2d(x, W, strides=stride, padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

#neural net architecture
#for architecture details see single-gpu model
tfx = tf.placeholder(tf.float32, shape=[None,image_size,image_size,num_channels])
tfy = tf.placeholder(tf.float32, shape=[None,num_labels])
kp1 = tf.placeholder(tf.float32)
kp2 = tf.placeholder(tf.float32)
w1 = tf.Variable(conv_ortho_weights(3,3,3,80),name='w1')
b1 = tf.Variable(bias_variable([80]),name='b1')
w2 = tf.Variable(conv_ortho_weights(3,3,80,80),name='w2')
b2 = tf.Variable(bias_variable([80]),name='b2')
w3 = tf.Variable(conv_ortho_weights(3,3,80,160),name='w3')
b3 = tf.Variable(bias_variable([160]),name='b3')
w4 = tf.Variable(conv_ortho_weights(3,3,160,160),name='w4')
b4 =  tf.Variable(bias_variable([160]),name='b4')
w5 = tf.Variable(conv_ortho_weights(3,3,160,320),name='w5')
b5 =  tf.Variable(bias_variable([320]),name='b5')
w6 = tf.Variable(conv_ortho_weights(3,3,320,320),name='w6')
b6 =  tf.Variable(bias_variable([320]),name='b6')
w7 = tf.Variable(dense_ortho_weights(8 * 8 * 320, 2000),name='w7')
b7 = tf.Variable(bias_variable([2000]),name='b7')
w8 = tf.Variable(dense_ortho_weights(2000, num_labels),name='w8')
b8 = tf.Variable(bias_variable([num_labels]),name='b8')
optimizer = tf.train.AdamOptimizer(0.0002,0.9,0.99)

#split operations over multiple gpu devices
#each GPU takes 1/num_gpus of the input data and operates on that portion
#each GPU calculates the loss and gradients over 1/num_gpus of the input data
#for architecture details see single-gpu model
tfx_split = []
tfy_split = []
d1 = []
l1 = []
l2 = []
maxpool1 = []
l3 = []
l4 = []
maxpool2 = []
l5 = []
l6 = []
flattened = []
l7 = []
drop = []
lastLayer = []
loss = []
grad = []

for i in xrange(num_gpus):
    with tf.device('/gpu:%d' % i):
        tfx_split.append(tf.slice(tfx,[i*batch_size/num_gpus,0,0,0],[batch_size/num_gpus,32,32,3]))
        tfy_split.append(tf.slice(tfy,[i*batch_size/num_gpus,0],[batch_size/num_gpus,num_labels]))
        d1.append(tf.nn.dropout(tfx_split[i], kp1))
        l1.append(tf.nn.elu(conv2d(d1[i],w1) + b1))
        l2.append(tf.nn.elu(conv2d(l1[i], w2) + b2))
        maxpool1.append(max_pool_2x2(l2[i]))
        l3.append(tf.nn.elu(conv2d(maxpool1[i], w3) + b3))
        l4.append(tf.nn.elu(conv2d(l3[i], w4) + b4))
        maxpool2.append(max_pool_2x2(l4[i]))
        l5.append(tf.nn.elu(conv2d(maxpool2[i], w5) + b5))
        l6.append(tf.nn.elu(conv2d(l5[i], w6) + b6))
        flattened.append(tf.reshape(l6[i], [-1, 8 * 8 * 320]))
        l7.append(tf.nn.elu(tf.matmul(flattened[i], w7) + b7))
        drop.append(tf.nn.dropout(l7[i], kp2))
        lastLayer.append(tf.matmul(drop[i], w8) + b8)
        loss.append(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=lastLayer[i],labels=tfy_split[i])))
        grad.append(optimizer.compute_gradients(loss[i]))

#function to average gradients, taken from tensorflow CIFAR-10 example
def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(0, grads)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

#average the gradients over each of the GPUs  
grads = average_gradients(grad)

#average the loss over each of the GPUs
losses = tf.reduce_mean(loss, 0)

#apply the averaged gradient
apply_gradient = optimizer.apply_gradients(grads)

#combine last layer output of GPUs for additional calculations during cross validation
lastLayers = tf.concat(0, lastLayer)
prediction = tf.nn.softmax(lastLayers)
correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(tfy,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#run model
init_op = tf.initialize_all_variables()
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(init_op)
open('accuracy.txt', 'w').close()

for i in range(noOfIterations):
    start = time.time()
    indices = np.random.permutation(X_train.shape[0])[:batch_size]
    
    #generate random minibatch
    X_batch = X_train[indices,:,:,:]
    
    #random translation of image by 5 image
    crop1 = np.random.randint(-5,6)
    crop2 = np.random.randint(-5,6)
    if crop1 > 0:
        X_batch = np.concatenate((X_batch[:,crop1:,:,:],np.zeros((batch_size,crop1,image_size,num_channels))),axis=1)
    elif crop1 < 0:
        X_batch = np.concatenate((np.zeros((batch_size,-crop1,image_size,num_channels)),X_batch[:,:crop1,:,:]),axis=1)
    if crop2 > 0:
        X_batch = np.concatenate((X_batch[:,:,crop2:,:],np.zeros((batch_size,image_size,crop2,num_channels))),axis=2)
    elif crop2 < 0:
        X_batch = np.concatenate((np.zeros((batch_size,image_size,-crop2,num_channels)),X_batch[:,:,:crop2,:]),axis=2)   
    y_batch = y_train[indices,:]
    
    #randomly flip image
    if random.random() < .5:
        X_batch = X_batch[:,:,::-1,:]
    
    #train
    feed_dict = {tfx:X_batch,tfy:y_batch,kp1:0.8,kp2:0.5}
    l,_ = sess.run([losses,apply_gradient], feed_dict=feed_dict)
    end = time.time()
    print 'time elapsed: ', end - start
    print 'iteration %i loss: %.4f' % (i, l)
    
    #cross validation accuracy
    if (i % 100 == 0):
        test_accuracies = []
        for j in range(0,X_test.shape[0],batch_size):
            feed_dict={tfx:X_test[j:j+batch_size,:,:,:],tfy:y_test[j:j+batch_size,:],kp1:1.,kp2:1.}
            test_accuracies.append(sess.run(accuracy, feed_dict=feed_dict)*100)
        print 'iteration %i test accuracy: %.4f%%' % (i, np.mean(test_accuracies))
        with open("accuracy.txt", "a") as f:
            f.write('iteration %i test accuracy: %.4f%%\n' % (i, np.mean(test_accuracies)))
            f.write('iteration %i training loss: %.4f\n' % (i, l))
    
    #run model on test
    if (i % 5000 == 0):
        preds = []
        for j in range(0,X_full.shape[0],batch_size):
            feed_dict={tfx:X_full[j:j+batch_size,:,:,:],kp1:1.,kp2:1.}
            p = sess.run(prediction, feed_dict=feed_dict)
            preds.append(np.argmax(p, 1))
        pred = np.concatenate(preds)
        np.savetxt('prediction.txt',pred,fmt='%.0f')