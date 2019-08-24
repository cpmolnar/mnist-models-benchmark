import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

def load_data(split = 0.8):
    #(X_train, y_train), (X_test, y_test) = mnist.load_data()
    #X_train = np.vstack((X_train, X_test))
    #y_train = np.concatenate([y_train, y_test])

    #X_train = X_train.reshape(-1, 28, 28, 1)
    #print(X_train.shape, y_train.shape)

    data = pd.read_csv('datasets/train.csv').values
    cutoff = int(data.shape[0] * 0.8)

    y_train = data[:cutoff,0].astype('int32')
    y_val = data[cutoff:,0].astype('int32')

    X = data[:,1:].astype('float32')
    X = X.reshape(-1,28,28,1)
    X_train = X[:cutoff, :]
    X_val = X[cutoff:, :]

    X_test = pd.read_csv('datasets/test.csv').values.astype('float32')
    X_test = X_test.reshape(-1, 28, 28, 1)

    return X_train, X_val, y_train, y_val, X_test

def visualize_input(X_train):
    fig = plt.figure(figsize = (12,12)) 
    ax = fig.add_subplot(111)
    img = X_train[0].reshape(28,28)

    ax.imshow(img, cmap='gray')
    width, height = img.shape
    thresh = img.max()/2.5
    for x in range(width):
        for y in range(height):
            ax.annotate(str(round(img[x][y],2)), xy=(y,x),
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='white' if img[x][y]<thresh else 'black')

    plt.show()

def sample_input(X_train, y_train):
    fig = plt.figure(figsize=(20,20))
    for i in range(6):
        ax = fig.add_subplot(1, 6, i+1, xticks=[], yticks=[])
        ax.imshow(X_train[i].reshape(28,28), cmap='gray')
        ax.set_title(str(y_train[i]))

    plt.show()

def to_categorical(input, num_categories):
    """
    Turns an array into a one-hot array
    """
    return np.eye(num_categories)[input]

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :]
    shuffled_Y = Y[permutation, :]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches