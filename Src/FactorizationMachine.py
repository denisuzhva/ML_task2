import numpy as np
from Model import Model



class FactorizationMachine(Model):

    def __init__(self, num_features, num_factors):

        self._num_features = num_features
        self._num_factors = num_factors # a.k.a. k
        self.__b = 0    # bias
        self.__w = np.random.randn(self._num_features)    # w vector
        self.__v = np.random.randn(self._num_features, self._num_factors)    # second order factor matrix


    def getPrediction(self, x, batch_size):

        xw = self.__w[x[:, 0]] + self.__w[x[:, 1]]
        xv = np.zeros((x.shape[0]), dtype=np.float)
        for t in range(batch_size):
            xv[t] = np.dot(self.__v[x[t, 0], :], self.__v[x[t, 1], :])
                
        prediction = self.__b + xw + xv
        return prediction


    def updateParameters(self, x_batch, z_batch, batch_size, lr=0.01):  # dL / dw ?

        pred_batch = self.getPrediction(x_batch, batch_size)
        diff_part = -2 * (z_batch - pred_batch) / batch_size
        batch_range = range(batch_size)

        db = np.sum(diff_part)

        dw = np.zeros((self._num_features), dtype=np.float)
        for t in batch_range:
            dw[x_batch[t, 0]] += diff_part[t]
            dw[x_batch[t, 1]] += diff_part[t]

        dv = np.zeros((self._num_features, self._num_factors), dtype=np.float)
        for t in batch_range:
            dv[x_batch[t, 0]] += diff_part[t] * self.__v[x_batch[t, 1]]
            dv[x_batch[t, 1]] += diff_part[t] * self.__v[x_batch[t, 0]]

        self.__b -= lr * db
        self.__w -= lr * dw
        self.__v -= lr * dv


    def resetWeights(self):

        self.__b = 0.0
        self.__w = np.zeros(self._num_features, dtype=np.float)
        self.__v = np.zeros((self._num_features, self._num_factors), dtype=np.float)

    
    def getWeights(self):

        return self.__b, self.__w, self.__v

