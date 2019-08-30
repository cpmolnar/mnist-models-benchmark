import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras.utils.np_utils
import tensorflow as tf
import time
from utils.mnist_utils import *

# Turning off logging clutter
import logging
logging.getLogger('tensorflow').disabled = True
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

MINI_BATCH_SIZE=4200

class inception_v4:
    def build(self, input):
        stem = self.conv_layer(input, num_outputs=32, kernel_size=3, stride=2, padding='VALID')
        #stem = self.conv_layer(stem, num_outputs=32, kernel_size=3, stride=1, padding='VALID')
        #stem = self.conv_layer(stem, num_outputs=64, kernel_size=3, stride=1, padding='SAME')
        #stem_branch_1a = self.conv_layer(stem, num_outputs=96, kernel_size=3, stride=2, padding='VALID')
        #stem_branch_1b = tf.contrib.layers.max_pool2d(stem, kernel_size=3, stride=2, padding='VALID')
        #stem = tf.concat(axis=-1, values=[stem_branch_1a, stem_branch_1b])
        #stem_branch_2a = self.conv_layer(stem, num_outputs=64, kernel_size=1, stride=1, padding='SAME')
        #stem_branch_2a = self.conv_layer(stem_branch_2a, num_outputs=64, kernel_size=[7, 1], stride=1, padding='SAME')
        #stem_branch_2a = self.conv_layer(stem_branch_2a, num_outputs=64, kernel_size=[1, 7], stride=1, padding='SAME')
        #stem_branch_2a = self.conv_layer(stem_branch_2a, num_outputs=96, kernel_size=3, stride=1, padding='VALID')
        #stem_branch_2b = self.conv_layer(stem, num_outputs=64, kernel_size=1, stride=1, padding='SAME')
        #stem_branch_2b = self.conv_layer(stem_branch_2b, num_outputs=96, kernel_size=3, stride=1, padding='VALID')
        #stem = tf.concat(axis=-1, values=[stem_branch_2a, stem_branch_2b])
        #stem_branch_3a = self.conv_layer(input, num_outputs=192, kernel_size=3, stride=1, padding='VALID')
        #stem_branch_3b = tf.contrib.layers.max_pool2d(input, kernel_size=3, stride=2, padding='VALID')
        #self.stem = tf.concat(axis=-1, values=[stem_branch_3a, stem_branch_3b])

        inception_A = self.inception_A(input)
        inception_A = self.inception_A(inception_A)
        inception_A = self.inception_A(inception_A)
        self.inception_A = self.inception_A(inception_A)

        self.reduction_A = self.reduction_A(self.inception_A)

        inception_B = self.inception_B(self.reduction_A)
        inception_B = self.inception_B(inception_B)
        inception_B = self.inception_B(inception_B)
        inception_B = self.inception_B(inception_B)
        inception_B = self.inception_B(inception_B)
        inception_B = self.inception_B(inception_B)
        self.inception_B = self.inception_B(inception_B)

        self.reduction_B = self.reduction_B(self.inception_B)

        inception_C = self.inception_C(self.reduction_B)
        inception_C = self.inception_C(inception_C)
        self.inception_C = self.inception_C(inception_C)

        self.avg_pool = tf.contrib.layers.avg_pool2d(self.inception_C, kernel_size=3, stride=2, padding='VALID')
        self.dropout = tf.nn.dropout(self.avg_pool, rate=0.2)
        self.fc = self.fc_layer(self.dropout, num_outputs=10)

    def conv_layer(self, val, num_outputs, kernel_size=3, stride=1, padding='SAME'):
        conv = tf.contrib.layers.conv2d(val, num_outputs, kernel_size, stride, padding, weights_initializer=tf.contrib.layers.xavier_initializer(seed = 1), activation_fn=None)
        conv = tf.contrib.layers.batch_norm(conv)
        conv = tf.nn.relu(conv)
        return conv

    def fc_layer(self, val, num_outputs):
        fc = tf.contrib.layers.flatten(val)
        fc = tf.contrib.layers.fully_connected(fc, num_outputs, activation_fn=None)
        return fc

    def inception_A(self, val):
        branch_1 = tf.contrib.layers.avg_pool2d(val, kernel_size=3, stride=1, padding='SAME')
        branch_1 = self.conv_layer(branch_1, num_outputs=96, kernel_size=1, stride=1, padding='SAME')

        branch_2 = self.conv_layer(val, num_outputs=96, kernel_size=1, stride=1, padding='SAME')

        branch_3 = self.conv_layer(val, num_outputs=64, kernel_size=1, stride=1, padding='SAME')
        branch_3 = self.conv_layer(branch_3, num_outputs=96, kernel_size=3, stride=1, padding='SAME')

        branch_4 = self.conv_layer(val, num_outputs=64, kernel_size=1, stride=1, padding='SAME')
        branch_4 = self.conv_layer(branch_4, num_outputs=96, kernel_size=3, stride=1, padding='SAME')
        branch_4 = self.conv_layer(branch_4, num_outputs=96, kernel_size=3, stride=1, padding='SAME')

        concat = tf.concat(axis=-1, values=[branch_1, branch_2, branch_3, branch_4])
        return concat

    def reduction_A(self, val):
        # Inception-v4: k=192 l=224 m=256 n=384
        branch_1 = tf.contrib.layers.max_pool2d(val, kernel_size=3, stride=2, padding='VALID')

        branch_2 = self.conv_layer(val, num_outputs=384, kernel_size=3, stride=2, padding='VALID')

        branch_3 = self.conv_layer(val, num_outputs=192, kernel_size=1, stride=1, padding='SAME')
        branch_3 = self.conv_layer(branch_3, num_outputs=224, kernel_size=3, stride=1, padding='SAME')
        branch_3 = self.conv_layer(branch_3, num_outputs=256, kernel_size=3, stride=2, padding='VALID')

        concat = tf.concat(axis=-1, values=[branch_1, branch_2, branch_3])
        return concat

    def inception_B(self, val):
        branch_1 = tf.contrib.layers.avg_pool2d(val, kernel_size=3, stride=1, padding='SAME')
        branch_1 = self.conv_layer(branch_1, num_outputs=128, kernel_size=1, stride=1, padding='SAME')

        branch_2 = self.conv_layer(val, num_outputs=384, kernel_size=1, stride=1, padding='SAME')

        branch_3 = self.conv_layer(val, num_outputs=192, kernel_size=1, stride=1, padding='SAME')
        branch_3 = self.conv_layer(branch_3, num_outputs=224, kernel_size=[1, 7], stride=1, padding='SAME')
        branch_3 = self.conv_layer(branch_3, num_outputs=256, kernel_size=[7, 1], stride=1, padding='SAME')

        branch_4 = self.conv_layer(val, num_outputs=192, kernel_size=1, stride=1, padding='SAME')
        branch_4 = self.conv_layer(branch_4, num_outputs=192, kernel_size=[1, 7], stride=1, padding='SAME')
        branch_4 = self.conv_layer(branch_4, num_outputs=224, kernel_size=[7, 1], stride=1, padding='SAME')
        branch_4 = self.conv_layer(branch_4, num_outputs=224, kernel_size=[1, 7], stride=1, padding='SAME')
        branch_4 = self.conv_layer(branch_4, num_outputs=256, kernel_size=[7, 1], stride=1, padding='SAME')

        concat = tf.concat(axis=-1, values=[branch_1, branch_2, branch_3, branch_4])
        return concat

    def reduction_B(self, val):
        branch_1 = tf.contrib.layers.max_pool2d(val, kernel_size=3, stride=2, padding='VALID')

        branch_2 = self.conv_layer(val, num_outputs=192, kernel_size=1, stride=1, padding='SAME')
        branch_2 = self.conv_layer(branch_2, num_outputs=192, kernel_size=3, stride=2, padding='VALID')

        branch_3 = self.conv_layer(val, num_outputs=256, kernel_size=1, stride=1, padding='SAME')
        branch_3 = self.conv_layer(branch_3, num_outputs=256, kernel_size=[1, 7], stride=1, padding='SAME')
        branch_3 = self.conv_layer(branch_3, num_outputs=320, kernel_size=[7, 1], stride=1, padding='SAME')
        branch_3 = self.conv_layer(branch_3, num_outputs=320, kernel_size=3, stride=2, padding='VALID')

        concat = tf.concat(axis=-1, values=[branch_1, branch_2, branch_3])
        return concat

    def inception_C(self, val):
        branch_1 = tf.contrib.layers.avg_pool2d(val, kernel_size=3, stride=1, padding='SAME')
        branch_1 = self.conv_layer(branch_1, num_outputs=256, kernel_size=1, stride=1, padding='SAME')

        branch_2 = self.conv_layer(val, num_outputs=256, kernel_size=1, stride=1, padding='SAME')

        branch_3 = self.conv_layer(val, num_outputs=384, kernel_size=1, stride=1, padding='SAME')
        branch_3a = self.conv_layer(branch_3, num_outputs=256, kernel_size=[1, 3], stride=1, padding='SAME')
        branch_3b = self.conv_layer(branch_3, num_outputs=256, kernel_size=[3, 1], stride=1, padding='SAME')

        branch_4 = self.conv_layer(val, num_outputs=384, kernel_size=1, stride=1, padding='SAME')
        branch_4 = self.conv_layer(branch_4, num_outputs=448, kernel_size=[1, 3], stride=1, padding='SAME')
        branch_4 = self.conv_layer(branch_4, num_outputs=512, kernel_size=[3, 1], stride=1, padding='SAME')
        branch_4a = self.conv_layer(branch_4, num_outputs=256, kernel_size=[1, 3], stride=1, padding='SAME')
        branch_4b = self.conv_layer(branch_4, num_outputs=256, kernel_size=[3, 1], stride=1, padding='SAME')

        concat = tf.concat(axis=-1, values=[branch_1, branch_2, branch_3a, branch_3b, branch_4a, branch_4b])
        return concat

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
    max_lr = 0.008

    model = inception_v4()
    model.build(input)
    logits = model.fc
    tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits, weights=1.0, label_smoothing=0.1)

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