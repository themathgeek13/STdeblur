import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from copy import copy
from scipy.ndimage import rotate
import pandas as pd
import argparse
import os

from keras.layers.convolutional import Convolution3D

"""ARGUMENT PARSING"""
ap=argparse.ArgumentParser()
ap.add_argument("-lr", "--lr", required=False, default= 0.00001, help="Initial learning rate")
ap.add_argument("-batch_size","--batch_size", required=False, default=1, help="Batch size, 1 or multiples of 5")
ap.add_argument("-init","--init",required=False, default=1, help="Initialization: Xavier (1) or He (2)")
ap.add_argument("-save_dir","--save_dir", required=False, default="pa1/", help="Directory to save the model")

"""EXTRACT PARAMETERS"""
args=vars(ap.parse_args())
lr=float(args["lr"])                        #done
batch_size=args["batch_size"]               #done
init=args["init"]                           #done
save_dir=args["save_dir"]                   #done

train="train.csv"
test="test.csv"
val="val.csv"

init_method=tf.contrib.layers.xavier_initializer()

# This replaces the tensorflow mnist.train.next_batch(batch_size)
# SOURCE: https://stackoverflow.com/questions/40994583/how-to-implement-tensorflows-next-batch-for-own-data
def next_batch(num, data, labels):
	'''
	Return a total of `num` random samples and labels. 
	'''
	idx = np.arange(0 , len(data))
	np.random.shuffle(idx)
	idx = idx[:num]
	data_shuffle = [data[ i] for i in idx]
	labels_shuffle = [labels[ i] for i in idx]
	return np.asarray(data_shuffle), np.asarray(labels_shuffle)

# Function to convert class labels (array) 0-9 into one hot vectors (matrix)
def onehot(x):
    arr=np.zeros(10)
    arr[x]=1
    return arr

# Helper function to create CONV layers given data, filters, etc.
# Uses the initializer method as per input argument
def create_new_conv_layer(input_data, num_input_channels, num_filters, filter_shape, name):
        # setup the filter input shape for tf.nn.conv_2d
        conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels, num_filters]

        # initialise weights and bias for the filter
	weights = tf.get_variable(name+"_W", shape=conv_filt_shape, initializer=init_method)
	bias=tf.get_variable(name+"_b", shape=[num_filters], initializer=init_method)        

        # setup the convolutional layer operation
        out_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='SAME',name=name)

        # add the bias
        out_layer += bias

        # apply a ReLU non-linear activation
        out_layer = tf.nn.relu(out_layer)
        return out_layer,weights

# Helper function to perform max pooling, since sometimes 2 CONV layers are back to back
# Uses a stride of 2 along x and y directions by default, can be changed.
def perform_max_pooling(out_layer, pool_shape):
        # now perform max pooling
        # ksize is the argument which defines the size of the max pooling window (i.e. the area over which the maximum is
        # calculated).  It must be 4D to match the convolution - in this case, for each image we want to use a 2 x 2 area
        # applied to each channel
        ksize = [1, pool_shape[0], pool_shape[1], 1]
        # strides defines how the max pooling area moves through the image - a stride of 2 in the x direction will lead to
        # max pooling areas starting at x=0, x=2, x=4 etc. through your image.  If the stride is 1, we will get max pooling
        # overlapping previous max pooling areas (and no reduction in the number of parameters).  In this case, we want
        # to do strides of 2 in the x and y directions.
        strides = [1, 2, 2, 1]
        out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides, padding='SAME')
        return out_layer

# Begin the tensorflow graph definition

train_images=np.load("x_train.npy")
train_labels=np.load("y_train.npy")
print len(train_images)

# Python optimisation variables
learning_rate = lr
epochs = 10

# declare the training data placeholders
# input x - for 28 x 28 pixels = 784 - this is the flattened image data
x = tf.placeholder(tf.float32, [None, 128, 128, 5],name='input')
# reshape the input data so that it is a 4D tensor.  The first value (-1) tells function to dynamically shape that
# dimension based on the amount of data passed to it.  The two middle dimensions are set to the image size (i.e. 28
# x 28). 
x_shaped = tf.reshape(x, [-1, 128, 128, 5])
# now declare the output data placeholder - 10 digits
y = tf.placeholder(tf.float32, [None, 128, 128],name='y')
y = tf.reshape(y, [-1, 128, 128, 1])

# create some convolutional layers
layer1,filt = create_new_conv_layer(x_shaped, 5, 32, [3, 3], name='layer1')
layer2,_ = create_new_conv_layer(layer1, 32, 64, [3, 3], name='layer2')
layer3,_ = create_new_conv_layer(layer2, 64, 64, [3, 3], name='layer3')
layer4,_ = create_new_conv_layer(layer3, 64, 1, [3, 3], name='layer4')

# add batch normalization
#dense_layer3 = tf.contrib.layers.batch_norm(dense_layer3, center=True, scale=True, is_training=phase, scope='bn')
# perform softmax on the output
#y_ = tf.nn.softmax(dense_layer3, name="output")

# define the error function
#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dense_layer3, labels=y))

print layer4.shape
print y.shape
mse = tf.reduce_mean(tf.losses.mean_squared_error(y,layer4))

# add an optimiser
optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(mse)

# define an accuracy assessment operation
#correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# setup the initialisation operator
init_op = tf.global_variables_initializer()

# setup recording variables
# add a summary to store the accuracy
#tf.summary.scalar('accuracy', accuracy)
#filter_summary=tf.image_summary(filt)
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('Tensorboard/')
saver = tf.train.Saver()
prev_cost=10
with tf.Session() as sess:
	# initialise the variables from the saved checkpoint if possible, else start a new session
	try:
		saver.restore(sess, os.path.join(save_dir, "model.ckpt"))
		print("successfully restored session")
	except:
	        sess.run(init_op)
		print("started a new session")
        total_batch = int(len(train_labels) / batch_size)
        for epoch in range(epochs):
            avg_cost = 0
            for i in range(total_batch):
                batch_x,batch_y=next_batch(batch_size,train_images,train_labels)
                batch_y=batch_y.reshape(len(batch_y),128,128,1)
                _, c = sess.run([optimiser, mse], feed_dict={x: batch_x, y: batch_y})
                image=tf.reshape(batch_y[:1],[-1, 128, 128, 1])
                tf.summary.image("image",image)
                #avg_cost += c / total_batch 
            #valid_cost = sess.run(accuracy, feed_dict={x: valfeat, y: valclasses, prob: 1.0, phase: False})
	    train_cost = sess.run(mse, feed_dict={x: batch_x, y: batch_y})
            print("Epoch:", (epoch + 1))
            print " train cost: {:.3f}".format(train_cost)
	    """
	    #EARLY STOPPING
	    if epoch%5==0:
		if valid_cost>prev_cost:
			break
		else:
		        saver.save(sess, os.path.join(save_dir,"model.ckpt"))
			prev_cost=valid_cost
	    """
	print("\nTraining complete!")
