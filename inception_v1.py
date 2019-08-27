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

class inception_v1:
    def build(self, input):
        # self.conv1_1 = self.conv_layer(input, num_outputs=64, kernel_size=7, stride=2, padding='SAME')
        # self.pool1 = self.max_pool(self.conv1_1)
        
        self.conv2_1 = self.conv_layer(input, num_outputs=64, kernel_size=1, stride=1, padding='SAME')
        self.conv2_2 = self.conv_layer(self.conv2_1, num_outputs=192, kernel_size=3, stride=1, padding='SAME')
        self.pool2 = self.max_pool(self.conv2_2)

        self.inception3_1 = self.inception_block(self.pool2, 64, 96, 128, 16, 32, 32)
        self.inception3_2 = self.inception_block(self.inception3_1, 128, 128, 192, 32, 96, 64)
        self.pool3 = self.max_pool(self.inception3_2)

        self.inception4_1 = self.inception_block(self.pool3, 192, 96, 208, 16, 48, 64)
        self.inception4_2 = self.inception_block(self.inception4_1, 160, 112, 224, 24, 64, 64)
        self.inception4_3 = self.inception_block(self.inception4_2, 128, 128, 256, 24, 64, 64)
        self.inception4_4 = self.inception_block(self.inception4_3, 112, 144, 288, 32, 64, 64)
        self.inception4_5 = self.inception_block(self.inception4_4, 256, 160, 320, 32, 128, 128)
        self.pool4 = self.max_pool(self.inception4_5)

        self.inception5_1 = self.inception_block(self.pool4, 256, 160, 320, 32, 128, 128)
        self.inception5_2 = self.inception_block(self.inception5_1, 384, 192, 384, 48, 128, 128)
        self.pool5 = self.avg_pool(self.inception5_2)

        self.fc6 = self.fc_layer(self.pool5, num_outputs=10)
               
    def avg_pool(self, val, kernel_size=3, stride=2, padding='VALID'):
        return tf.contrib.layers.avg_pool2d(val, kernel_size, stride, padding)

    def max_pool(self, val, kernel_size=3, stride=2, padding='SAME'):
        return tf.contrib.layers.max_pool2d(val, kernel_size, stride, padding)

    def conv_layer(self, val, num_outputs, kernel_size=3, stride=1, padding='SAME'):
        conv = tf.contrib.layers.conv2d(val, num_outputs, kernel_size, stride, padding, weights_initializer=tf.contrib.layers.xavier_initializer(seed = 1), activation_fn=None)
        conv = tf.contrib.layers.batch_norm(conv)
        conv = tf.nn.relu(conv)
        return conv

    def fc_layer(self, val, num_outputs):
        fc = tf.contrib.layers.flatten(val)
        fc = tf.contrib.layers.fully_connected(fc, num_outputs, activation_fn=None)
        return fc

    def inception_block(self, val, num_1x1, num_3x3_reduce, num_3x3, num_5x5_reduce, num_5x5, pool_proj):
        branch_0 = self.conv_layer(val, num_outputs=num_1x1, kernel_size=1, stride=1, padding='SAME')
        
        branch_1 = self.conv_layer(val, num_outputs=num_3x3_reduce, kernel_size=1, stride=1, padding='SAME')
        branch_1 = self.conv_layer(branch_1, num_outputs=num_3x3, kernel_size=3, stride=1, padding='SAME')

        branch_2 = self.conv_layer(val, num_outputs=num_5x5_reduce, kernel_size=1, stride=1, padding='SAME')
        branch_2 = self.conv_layer(branch_2, num_outputs=num_5x5, kernel_size=5, stride=1, padding='SAME')

        branch_3 = self.max_pool(val, stride=1)
        branch_3 = self.conv_layer(branch_3, num_outputs=pool_proj, kernel_size=1, stride=1, padding='SAME')

        concat = tf.concat(axis=-1, values=[branch_0, branch_1, branch_2, branch_3])

        return concat
    
def compute_cost(logits, labels):

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = labels))

    return cost

def model(X_train, X_val, y_train, y_val, print_cost = True, learning_rate = 0.0002, minibatch_size = 64, num_epochs = 10):
    ops.reset_default_graph()

    input = tf.placeholder(tf.float32, [None, 28, 28, 1])
    labels = tf.placeholder(tf.float32, [None, 10])

    model = inception_v1()
    model.build(input)
    output = model.fc6
    cost = compute_cost(output, labels)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    print("Trainable variables:", np.sum([np.prod(var.shape) for var in tf.trainable_variables()]))

    costs = []
    seed = 0
    config = tf.ConfigProto(log_device_placement=False)
    config.intra_op_parallelism_threads = 44
    config.inter_op_parallelism_threads = 44
    config.gpu_options.allocator_type = 'BFC'
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

            for i in range(num_minibatches):
                (minibatch_X, minibatch_Y) = minibatches[i]

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