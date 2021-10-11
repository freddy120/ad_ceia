import numpy as np


class BaseModel(object):

    def __init__(self):
        self.model = None

    def fit(self, X, Y):
        return NotImplemented

    def predict(self, X):
        return NotImplemented


class ConstantModel(BaseModel):

    def fit(self, X, Y):
        W = Y.mean()
        self.model = W

    def predict(self, X):
        return np.ones(len(X)) * self.model


class LinearRegression(BaseModel):

    def fit(self, X, y):
        if len(X.shape) == 1:
            W = X.T.dot(y) / X.T.dot(X)
        else:
            W = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        self.model = W

    def predict(self, X):
        return self.model * X


class LinearRegressionWithB(BaseModel):

    def fit(self, X, y):
        X_expanded = np.hstack((np.ones((len(X),1)),X))
        W = np.linalg.inv(X_expanded.T.dot(X_expanded)).dot(X_expanded.T).dot(y)
        self.model = W

    def predict(self, X):
        X_expanded = np.hstack((np.ones((len(X),1)),X))
        return X_expanded.dot(self.model)


class LinearRegressionWithGradient(BaseModel):
    """
        :param X: X_train
        :param y: y_train
        :param lr: learning rate
        :param epochs: number of epochs
        :param b: batch size for mini-batch
        :param gradient: type of gradient descent: 'SGD', 'MINI', 'BGD'
    """

    def fit(self, X, y, lr=0.01, epochs=100, b=16, gradient='SGD'):
        X_expanded = np.hstack((np.ones((len(X),1)),X))

        if gradient == 'BGD':
            w = self.gradient_descent(X_expanded, y, lr, epochs)
            self.model = w
            return w
        elif gradient == 'SGD':
            w = self.stochastic_gradient_descent(X_expanded, y, lr, epochs)
            self.model = w
            return w
        elif gradient == 'SGD':
            w = self.mini_batch_gradient_descent(X_expanded, y, lr, epochs, b)
            self.model = w
            return w


    def predict(self, X):
        X_expanded = np.hstack((np.ones((len(X),1)),X))
        return X_expanded.dot(self.model)

    def gradient_descent(self, X_train, y_train, lr=0.01, amt_epochs=100):
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

    def stochastic_gradient_descent(self, X_train, y_train, lr=0.01, amt_epochs=100):
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

    def mini_batch_gradient_descent(self, X_train, y_train, lr=0.01, amt_epochs=100, b=16):
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


class LogisticRegression(object):

    def __init__(self):
        self.w = None

    """
        :param X: X_train
        :param y: y_train
        :param lr: learning rate
        :param epochs: number of epochs
        :param b: batch size for mini-batch
        :param gradient: type of gradient descent: 'SGD', 'MINI', 'BGD'
    """
    def fit(self, X=None, y=None, lr=0.01, epochs=100, b=16, gradient='SGD'):
        print(y)
        X_expanded = np.hstack((np.ones((len(X),1)),X))

        if gradient == 'BGD':
            h, w = self.gradient_descent(X_expanded, y, lr, epochs)
            self.w = w
            return h, w
        elif gradient == 'SGD':
            h, w = self.stochastic_gradient_descent(X_expanded, y, lr, epochs)
            self.w = w
            return h, w
        elif gradient == 'SGD':
            h, w = self.mini_batch_gradient_descent(X_expanded, y, lr, epochs, b)
            self.w = w
            return h, w

    def predict(self, X):
        X_expanded = np.hstack((np.ones((len(X),1)),X))
        pred = self.sigmoid(np.dot(X_expanded, self.w))
        return [1 if i >= 0.5 else 0 for i in pred]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def bce(self, X, y, theta):
        h = self.sigmoid(np.dot(X, theta))
        return (1 / len(y)) * ((np.dot((-y).T, np.log(h))) - (np.dot((1 - y).T, np.log(1 - h))))

    def gradient_descent(self, X_train, y_train, lr=0.01, amt_epochs=100):
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
            W = W - (lr / n) * np.dot(X_train.T, (self.sigmoid(np.dot(X_train, W)) - y_train))
            hist_cos[i] = self.bce(X_train, y_train, W)
        return hist_cos, W

    def stochastic_gradient_descent(self, X_train, y_train, lr=0.01, amt_epochs=100):
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
                W = W - (lr / n) * np.dot(X_train[j].reshape(1, -1).T, (self.sigmoid(np.dot(X_train[j], W)) - y_train[j]))
            hist_cos[i] = self.bce(X_train, y_train, W)
        return hist_cos, W

    def mini_batch_gradient_descent(self, X_train, y_train, lr=0.01, amt_epochs=100, b=16):
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
                W = W - (lr / batch_size) * np.dot(batch_X.T, (self.sigmoid(np.dot(batch_X, W)) - batch_y))

            hist_cos[i] = self.bce(batch_X, batch_y, W)

        return hist_cos, W
