import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))


def bce(X, y, theta):
    h = sigmoid(np.dot(X,theta))
    return (1/len(y))*((np.dot((-y).T,np.log(h)))-(np.dot((1-y).T, np.log(1-h))))


def gradient_descent(X_train, y_train, lr=0.01, amt_epochs=100):
    """
    shapes:
        X_train = nxm
        y_train = nx1
        W = mx1
    """

    n = X_train.shape[0]
    m = X_train.shape[1]

    # initialize random weights
    W = np.random.randn(m).reshape(m, 1)  # mx1
    hist_cos = np.zeros((amt_epochs, 1))
    for i in range(amt_epochs):
        W = W - (lr / n) * np.dot(X_train.T, (sigmoid(np.dot(X_train, W)) - y_train))
        hist_cos[i] = bce(X_train, y_train, W)
    return hist_cos, W


def stochastic_gradient_descent(X_train, y_train, lr=0.01, amt_epochs=100):
    """
    shapes:
        X_train = nxm
        y_train = nx1
        W = mx1
    """

    n = X_train.shape[0]
    m = X_train.shape[1]

    # initialize random weights
    W = np.random.randn(m).reshape(m, 1)
    hist_cos = np.zeros((amt_epochs, 1))
    # Iterate over the n_epochs
    for i in range(amt_epochs):

        # Shuffle all the samples
        idx = np.random.permutation(X_train.shape[0])
        X_train = X_train[idx]
        y_train = y_train[idx]

        # Iterate over each sample
        for j in range(n):
            # Update all the weights
            W = W - (lr/n) * np.dot(X_train[j].reshape(1, -1).T, (sigmoid(np.dot(X_train[j], W)) - y_train[j]))
        hist_cos[i] = bce(X_train, y_train, W)
    return hist_cos, W


def mini_batch_gradient_descent(X_train, y_train, lr=0.01, amt_epochs=100, b=16):
    """
    shapes:
        X_train = nxm
        y_train = nx1
        W = mx1
    """

    n = X_train.shape[0]
    m = X_train.shape[1]

    # initialize random weights
    W = np.random.randn(m).reshape(m, 1)
    hist_cos = np.zeros((amt_epochs, 1))
    # iterate over the n_epochs
    for i in range(amt_epochs):

        # Shuffle all the samples
        idx = np.random.permutation(X_train.shape[0])
        X_train = X_train[idx]
        y_train = y_train[idx]

        # Calculate the batch size in samples as a function of the number of batches
        batch_size = int(len(X_train) / b)

        # Iterate over the batches
        for i in range(0, len(X_train), batch_size):
            end = i + batch_size if i + batch_size <= len(X_train) else len(X_train)
            batch_X = X_train[i: end]  # batch_size*m
            batch_y = y_train[i: end]  # batch_size*1

            # Update the weights
            W = W - (lr / batch_size) * np.dot(batch_X.T, (sigmoid(np.dot(batch_X, W)) - batch_y))

        hist_cos[i] = bce(batch_X, batch_y, W)

    return hist_cos, W