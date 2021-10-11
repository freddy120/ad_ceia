import numpy as np


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

    for i in range(amt_epochs):
        # Calculate the predictions for all samples
        prediction = np.matmul(X_train, W)  # nx1

        # Calculate the error for all samples
        error = y_train - prediction  # nx1

        # Calculate the gradient for all samples
        grad_sum = np.sum(error * X_train, axis=0)
        grad_mul = -2 / n * grad_sum  # 1xm
        gradient = np.transpose(grad_mul)  # mx1 (it also works with reshape)

        # Update the parameters
        W = W - (lr * gradient)

    return W


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

    # Iterate over the n_epochs
    for i in range(amt_epochs):

        # Shuffle all the samples
        idx = np.random.permutation(X_train.shape[0])
        X_train = X_train[idx]
        y_train = y_train[idx]

        # Iterate over each sample
        for j in range(n):
            # Calculate the prediction for each sample
            prediction = np.matmul(X_train[j].reshape(1, -1), W)  # 1x1
            # Calculate the error for each sample
            error = y_train[j] - prediction  # 1x1
            # Calculate the gradient for each sample
            grad_sum = error * X_train[j]  # 1xm
            grad_mul = -2 / n * grad_sum  # 1xm
            gradient = np.transpose(grad_mul).reshape(-1, 1)  # mx1
            # Update all the weights
            W = W - (lr * gradient)  # mx1

    return W


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

            # Calculate the prediction for the whole batch
            prediction = np.matmul(batch_X, W)  # batch_sizex1
            # Calculate the error for the whole batch
            error = batch_y - prediction  # batch_sizex1

            # Calculate the gradient for the batch

            # error[batch_sizex1]*batch_X[batch_size*m]--> broadcasting --> batch_size*m
            grad_sum = np.sum(error * batch_X, axis=0)  # 1xm
            grad_mul = -2 / batch_size * grad_sum  # 1xm
            gradient = np.transpose(grad_mul)  # mx1

            # Update the weights
            W = W - (lr * gradient)

    return W