import numpy as np
from Model import Model



class FactorizationMachine(Model):

    def __init__(self, num_features, num_factors):

        self._num_features = num_features
        self._num_factors = num_factors # a.k.a. k
        self.__w = np.zeros(self._num_features, dtype=np.float)    # w vector
        self.__b = 0    # bias
        self.__v = np.zeros((self._num_features, self._num_factors), dtype=np.float)    # second order factor matrix


    def getPrediction(self, x):

        v2 = np.power(self.__v, 2)
        x2 = np.power(x, 2)
        factor_member = np.sum(np.power(np.dot(x, v), 2) - np.dot(x2, v2), axis=1) / 2

        prediction = self.__b + np.dot(x, self.__w) + factor_member
        return prediction


    def updateParameters(self, x_batch, z_batch, batch_size, lr=0.01):  # dL / dw ?

        # d(RMSE) / dw = (1/2) * d(MSE) / RMSE
        loss = np.sum(np.square(z_batch - self.getPrediction(x_batch))) / batch_size
        loss = np.sqrt(loss)

        dw = -1 * np.dot((z_batch - self.getPrediction(x_batch)), x_batch) / \
            (batch_size * loss)
        db = -1 * np.sum(z_batch - self.getPrediction(x_batch)) / \
            (batch_size * loss)

        # N = batch_size; n = num_features; k = num_factors
        # i, j = {1..n}; f = {1..k}
        x2 = np.repeat(np.power(x_batch, 2)[:, :, np.newaxis], self._num_factors, axis=2)  # construct R^{N x n x k} 
        m = np.dot(x, self.__v) # sum v_{jf} * x_j
        xm = np.dot(x.reshape((batch_size, -1, 1)), m.reshape((batch_size, 1, -1))) # R^{N x n} to R^{N x n x 1} and R^{N x k} to R^{N x 1 x k}
        x2v = np.multiply(v, x2)    # x_i^2 * v_{if}
        dv = -1 * np.tensordot((z_batch - self.getPrediction(x_batch)), (xm - x2v), axes=(0, 0))

        self.__w -= lr * dw
        self.__b -= lr * db
        self.__v -= lr * dv


    def resetWeights(self):

        self.__b = 0.0
        self.__w = np.zeros(self._num_features, dtype=np.float)
        self.__v = np.zeros((self._num_factors, self._num_factors), dtype=np.float)

    
    def getWeights(self):

        return self.__b, self.__w, self.__v

