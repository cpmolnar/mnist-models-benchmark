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

def circular_learning_rate(global_step, min_lr=0.0001, max_lr=0.0008, step_size=2000, gamma=0.99994, mode='triangular2', name=None):
    # Circular learning rate formula is from [https://arxiv.org/pdf/1506.01186.pdf]
    cycle = tf.floor(1. + tf.cast(global_step, "float32") / (2. * step_size))
    x = tf.abs(1. + ((tf.cast(global_step, "float32") / step_size) - (2. * cycle)))
    clr = tf.maximum(0., (1. - x)) * (max_lr - min_lr)

    if mode == 'triangular2':
        clr = (clr / (2 ** (cycle-1)))
    if mode == 'exp_range':
        clr = ((gamma ** tf.cast(global_step, "float32")) * clr)
    return (clr + min_lr)

def model(X_train, X_val, y_train, y_val, num_epochs=20, print_cost = True, minibatch_size = 64):
    input = tf.placeholder(tf.float32, [None, 28, 28, 1])
    labels = tf.placeholder(tf.float32, [None, 10])
    global_step = tf.train.get_or_create_global_step()
    num_minibatches = int(X_train.shape[0] / minibatch_size)
    step_size = (2 * num_minibatches)
    min_lr = 0.000
    max_lr = 0.0008

    model = vgg19()
    model.build(input)
    logits = model.fc8
    tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits, weights=1.0)

    losses = tf.get_collection(tf.GraphKeys.LOSSES)
    loss_op = tf.reduce_mean(tf.add_n(losses))
    optimizer = tf.train.AdamOptimizer(learning_rate=circular_learning_rate(global_step, min_lr, max_lr, step_size))
    train_op = optimizer.minimize(loss_op, global_step=global_step)
    acc, acc_op = tf.metrics.accuracy(labels=tf.argmax(labels, 1), predictions=tf.argmax(logits,1))

    def compute_accuracy(X, Y, mini_batch_size=MINI_BATCH_SIZE):
        accuracy=[]
        minibatches = random_mini_batches(X, Y, mini_batch_size)
        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = minibatch
            accuracy.append(sess.run(acc_op, {input:minibatch_X, labels: minibatch_Y}))
        return np.mean(accuracy)

    print("Trainable variables:", np.sum([np.prod(var.shape) for var in tf.trainable_variables()]))

    costs = []
    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        for epoch in range(1, num_epochs+1):
            start_time = time.time()
            minibatches = random_mini_batches(X_train, y_train, minibatch_size)

            epoch_cost = 0.
            for i in range(num_minibatches):
                (minibatch_X, minibatch_Y) = minibatches[i]

                _ , minibatch_cost = sess.run([train_op, loss_op], feed_dict={input: minibatch_X, labels: minibatch_Y})
                print("Minibatch %i of %i, cost: %f" % (i, num_minibatches, minibatch_cost))

                epoch_cost += minibatch_cost / num_minibatches
            
            if print_cost:
                print("Cost after epoch %i: %f, elapsed: %fs" % (epoch, epoch_cost, time.time() - start_time))
                costs.append(epoch_cost)

        # Plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('Cost')
        plt.xlabel('Iterations')
        plt.title("Cost per iteration")
        plt.show()
        
        print ("Train Accuracy:", compute_accuracy(X_train, y_train))
        print ("Test Accuracy:", compute_accuracy(X_val, y_val))

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