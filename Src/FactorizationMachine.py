import numpy as np
import opt_einsum as oe
from Model import Model



class FactorizationMachine(Model):

    def __init__(self, num_features, num_factors):

        self._num_features = num_features
        self._num_factors = num_factors # a.k.a. k
        self.__b = 0    # bias
        self.__w = np.random.randn(self._num_features)    # w vector
        self.__v = np.random.randn(self._num_features, self._num_factors)    # second order factor matrix


    def getPrediction(self, x):

        v2 = np.power(self.__v, 2)
        x2 = x.power(2)
        factor_member = np.sum(np.power(x.dot(self.__v), 2) - x2.dot(v2), axis=1) / 2
                
        prediction = self.__b + x.dot(self.__w) + factor_member
        return prediction


    def updateParameters(self, x_batch, z_batch, batch_size, lr=0.01):  # dL / dw ?

        pred_batch = self.getPrediction(x_batch)
        diff_part = -2 * (z_batch - pred_batch) / batch_size

        db = np.sum(diff_part)

        xt = x_batch.transpose()
        dw = xt.dot(diff_part)

        m = x_batch.dot(self.__v) # sum v_{jf} * x_j
        x2t = xt.power(2)
        diff_vx2 = np.multiply(self.__v, x2t.dot(diff_part).reshape((self._num_features, 1)))

        #x_batch_dense = np.array(x_batch.todense())
        #diff_xvx = np.einsum('b,bi,bf->if', diff_part, x_batch_dense, m)
        diff_xvx = oe.contract('b,bi,bf->if', diff_part, x_batch, m)

        #batch_range = range(batch_size)
        #diff_xvx = np.zeros((self._num_features, self._num_factors), dtype=np.float)
        #for t in batch_range:
        #    diff_xvx += diff_part[t] * x_batch_dense[t, :].reshape((self._num_features, 1)) .dot(m[t, :].reshape((1, self._num_factors)))

        dv = diff_xvx - diff_vx2

        self.__w -= lr * dw
        self.__b -= lr * db
        self.__v -= lr * dv


    def resetWeights(self):

        self.__b = 0.0
        self.__w = np.zeros(self._num_features, dtype=np.float)
        self.__v = np.zeros((self._num_features, self._num_factors), dtype=np.float)

    
    def getWeights(self):

        return self.__b, self.__w, self.__v

