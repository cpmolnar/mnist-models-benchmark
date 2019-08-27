import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras.utils.np_utils
import tensorflow as tf
from tensorflow.python.framework import ops
import time
	
from functools import reduce

from utils.mnist_utils import *

# Turning off logging clutter
import logging
logging.getLogger('tensorflow').disabled = True
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

MINI_BATCH_SIZE=4200

class vgg19:
    def __init__(self, dropout=0.5):
        self.dropout = dropout

    def build(self, input):
        self.conv1_1 = self.conv_layer(input, 1, 64, 'conv1_1')
        self.conv1_2 = self.conv_layer(self.conv1_1, 64, 64, 'conv1_2')
        self.pool1 = self.avg_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, 64, 128, 'conv2_1')
        self.conv2_2 = self.conv_layer(self.conv2_1, 128, 128, 'conv2_2')
        self.pool2 = self.avg_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, 128, 256, 'conv3_1')
        self.conv3_2 = self.conv_layer(self.conv3_1, 256, 256, 'conv3_2')
        self.conv3_3 = self.conv_layer(self.conv3_2, 256, 256, 'conv3_3')
        self.conv3_4 = self.conv_layer(self.conv3_3, 256, 256, 'conv3_4')
        self.pool3 = self.avg_pool(self.conv3_4, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, 256, 512, 'conv4_1')
        self.conv4_2 = self.conv_layer(self.conv4_1, 512, 512, 'conv4_2')
        self.conv4_3 = self.conv_layer(self.conv4_2, 512, 512, 'conv4_3')
        self.conv4_4 = self.conv_layer(self.conv4_3, 512, 512, 'conv4_4')
        self.pool4 = self.avg_pool(self.conv4_4, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, 512, 512, 'conv5_1')
        self.conv5_2 = self.conv_layer(self.conv5_1, 512, 512, 'conv5_2')
        self.conv5_3 = self.conv_layer(self.conv5_2, 512, 512, 'conv5_3')
        self.conv5_4 = self.conv_layer(self.conv5_3, 512, 512, 'conv5_4')
        self.pool5 = self.avg_pool(self.conv5_4, 'pool5')

        self.fc6 = self.fc_layer(self.pool5, 4096, 'fc6')
        self.relu6 = tf.nn.relu(self.fc6, 'relu6')
        self.dropout6 = tf.nn.dropout(self.relu6, self.dropout, name='dropout6')

        self.fc7 = self.fc_layer(self.relu6, 4096, 'fc7')
        self.relu7 = tf.nn.relu(self.fc7, 'relu7')
        self.dropout7 = tf.nn.dropout(self.relu7, self.dropout, name='dropout7')

        self.fc8 = self.fc_layer(self.relu7, 10, 'fc8')

    def avg_pool(self, val, name):
        return tf.nn.avg_pool2d(val, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, val, name):
        return tf.nn.max_pool(val, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, val, in_channels, out_channels, name):
        filter = tf.get_variable(name, shape=[3, 3, in_channels, out_channels], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(seed = 1))
        conv = tf.nn.conv2d(val, filter, [1, 1, 1, 1], 'SAME')
        conv = tf.nn.relu(conv)
        return conv

    def fc_layer(self, val, out_size, name):
        fc = tf.contrib.layers.flatten(val)
        fc = tf.contrib.layers.fully_connected(fc, out_size, activation_fn=None)
        return fc

def compute_cost(logits, labels):

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = labels))

    return cost

def model(X_train, X_val, y_train, y_val, print_cost = True, learning_rate = 0.00008, minibatch_size = 64, num_epochs = 20):
    ops.reset_default_graph() # to be able to rerun the model without overwriting tf variables

    input = tf.placeholder(tf.float32, [None, 28, 28, 1])
    labels = tf.placeholder(tf.float32, [None, 10])

    model = vgg19()
    model.build(input)
    output = model.fc8
    cost = compute_cost(output, labels)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    print("Trainable variables:", np.sum([np.prod(var.shape) for var in tf.trainable_variables()]))

    costs = []
    seed = 0
    config = tf.ConfigProto()
    config.intra_op_parallelism_threads = 44
    config.inter_op_parallelism_threads = 44
    config.gpu_options.allow_growth = True
    acc, acc_op = tf.metrics.accuracy(labels=tf.argmax(labels, 1), 
                                  predictions=tf.argmax(output,1))
    with tf.Session(config=config) as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        for epoch in range(num_epochs):
            start_time = time.time()

            epoch_cost = 0.                       # Defines a cost related to an epoch
            num_minibatches = int(X_train.shape[0] / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, y_train, minibatch_size)

            for i in range(num_minibatches):
                (minibatch_X, minibatch_Y) = minibatches[i]

                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={input: minibatch_X, labels: minibatch_Y})
                # print("Minibatch %i of %i, cost: %f" % (i, num_minibatches, minibatch_cost))

                epoch_cost += minibatch_cost / num_minibatches
            
            if print_cost:
                print("Cost after epoch %i: %f, elapsed: %fs" % (epoch, epoch_cost, time.time() - start_time))
                costs.append(epoch_cost)

        # Plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('Cost')
        plt.xlabel('Iterations')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        def compute_accuracy(X, Y, mini_batch_size):
            accuracy=[]
            num_minibatches = int(X.shape[0] / mini_batch_size)
            minibatches = random_mini_batches(X, Y, mini_batch_size)
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                accuracy.append(sess.run(acc_op, {input:minibatch_X, labels: minibatch_Y}))
            return tf.reduce_mean(accuracy).eval()
        
        print ("Train Accuracy:", compute_accuracy(X_train, y_train, MINI_BATCH_SIZE))
        print ("Test Accuracy:", compute_accuracy(X_val, y_val, MINI_BATCH_SIZE))

        return tf.trainable_variables()

def main():
    #sample_input(X_train, y_train)
    #visualize_input(X_train)

    print("Loading the data...")
    X_train, X_val, y_train, y_val, X_test = load_data(split=0.8)

    # Normalize the input
    X_train = X_train.astype('float32')/255
    X_val = X_val.astype('float32')/255
    X_test = X_test.astype('float32')/255

    # One-hot encode the labels
    y_train = to_categorical(y_train, 10)
    y_val = to_categorical(y_val, 10)

    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_val:", X_val.shape)
    print("y_val:", y_val.shape)

    parameters = model(X_train, X_val, y_train, y_val)

if __name__ == '__main__':
    main()