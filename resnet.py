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

class resnet:
    def build(self, input):
        # self.conv1_1 = self.conv_layer(input, num_outputs=64, kernel_size=7, stride=2, padding='VALID')
        # self.pool1 = self.avg_pool(self.conv1_1)

        self.id_block1_1 = self.identity_block(input, num_outputs=64)
        self.id_block1_2 = self.identity_block(self.id_block1_1, num_outputs=64)
        self.id_block1_3 = self.identity_block(self.id_block1_2, num_outputs=64)

        self.conv_block2_1 = self.conv_block(self.id_block1_3, num_outputs=128)
        self.id_block2_2 = self.identity_block(self.conv_block2_1, num_outputs=128)
        self.id_block2_3 = self.identity_block(self.id_block2_2, num_outputs=128)
        self.id_block2_4 = self.identity_block(self.id_block2_3, num_outputs=128)

        self.conv_block3_1 = self.conv_block(self.id_block2_4, num_outputs=256)
        self.id_block3_2 = self.identity_block(self.conv_block3_1, num_outputs=256)
        self.id_block3_3 = self.identity_block(self.id_block3_2, num_outputs=256)
        self.id_block3_4 = self.identity_block(self.id_block3_3, num_outputs=256)
        self.id_block3_5 = self.identity_block(self.id_block3_4, num_outputs=256)

        self.conv_block4_1 = self.conv_block(self.id_block3_5, num_outputs=512)
        self.id_block4_2 = self.identity_block(self.conv_block4_1, num_outputs=512)
        self.id_block4_3 = self.identity_block(self.id_block4_2, num_outputs=512)

        self.fc5 = self.fc_layer(self.id_block4_3, 10)


    def avg_pool(self, val):
        return tf.contrib.layers.avg_pool2d(val, kernel_size=2, stride=2)

    def max_pool(self, val):
        return tf.contrib.layers.max_pool2d(val, kernel_size=2, stride=2)

    def conv_layer(self, val, num_outputs, kernel_size=3, stride=1, padding='SAME'):
        # We initialize the weights as in https://arxiv.org/pdf/1502.01852.pdf (Xavier initialization)
        conv = tf.contrib.layers.conv2d(val, num_outputs, kernel_size, stride, padding, weights_initializer=tf.contrib.layers.xavier_initializer(seed = 1), activation_fn=None)
        # We adopt batch normalization (BN) right after each convolution and before activation, following https://arxiv.org/pdf/1502.03167.pdf
        conv = tf.contrib.layers.batch_norm(conv)
        conv = tf.nn.relu(conv)
        return conv

    def fc_layer(self, val, out_size):
        fc = tf.contrib.layers.flatten(val)
        fc = tf.contrib.layers.fully_connected(fc, out_size, activation_fn=None)
        return fc

    def identity_block(self, val, num_outputs, kernel_size=3):
        conv = tf.contrib.layers.conv2d(val, num_outputs, kernel_size, stride=1, padding='SAME', weights_initializer=tf.contrib.layers.xavier_initializer(seed = 1), activation_fn=None)
        conv = tf.contrib.layers.batch_norm(conv)
        conv = tf.nn.relu(conv)
        conv = tf.contrib.layers.conv2d(val, num_outputs, kernel_size, stride=1, padding='SAME', weights_initializer=tf.contrib.layers.xavier_initializer(seed = 1), activation_fn=None)
        conv = tf.contrib.layers.batch_norm(conv)
        conv = tf.add(conv, val)
        conv = tf.nn.relu(conv)

        return conv

    def conv_block(self, val, num_outputs, kernel_size=3):
        conv = tf.contrib.layers.conv2d(val, num_outputs, kernel_size, stride=2, padding='SAME', weights_initializer=tf.contrib.layers.xavier_initializer(seed = 1), activation_fn=None)
        conv = tf.contrib.layers.batch_norm(conv)
        conv = tf.nn.relu(conv)
        conv = tf.contrib.layers.conv2d(conv, num_outputs, kernel_size, stride=1, padding='SAME', weights_initializer=tf.contrib.layers.xavier_initializer(seed = 1), activation_fn=None)
        conv = tf.contrib.layers.batch_norm(conv)

        conv_skip = tf.contrib.layers.conv2d(val, num_outputs, kernel_size=1, stride=2, padding='SAME', weights_initializer=tf.contrib.layers.xavier_initializer(seed = 1), activation_fn=None)
        conv_skip = tf.contrib.layers.batch_norm(conv_skip)

        conv = tf.add(conv, conv_skip)
        conv = tf.nn.relu(conv)

        return conv
    
def compute_cost(logits, labels):

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = labels))

    return cost

def model(X_train, X_val, y_train, y_val, print_cost = True, learning_rate = 0.00008, minibatch_size = 64, num_epochs = 20):
    ops.reset_default_graph()

    input = tf.placeholder(tf.float32, [None, 28, 28, 1])
    labels = tf.placeholder(tf.float32, [None, 10])

    model = resnet()
    model.build(input)
    output = model.fc5
    cost = compute_cost(output, labels)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    print(np.sum([np.prod(var.shape) for var in tf.trainable_variables()]))

    costs = []
    seed = 0
    config = tf.ConfigProto(log_device_placement=False)
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

            epoch_cost = 0.
            num_minibatches = int(X_train.shape[0] / minibatch_size)
            seed = seed + 1
            minibatches = random_mini_batches(X_train, y_train, minibatch_size)

            i = 0
            for minibatch in minibatches:
                i += 1
                (minibatch_X, minibatch_Y) = minibatch

                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={input: minibatch_X, labels: minibatch_Y})
                print("Minibatch %i of %i, cost: %f" % (i, num_minibatches, minibatch_cost))

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

        # print ("Train Accuracy:", sess.run(acc_op, {input:X_train, labels: y_train}))
        print ("Test Accuracy:", sess.run(acc_op, {input:X_val, labels: y_val}))

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