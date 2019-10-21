import numpy as np
import abc


class Model(metaclass=abc.ABCMeta):

    ## Model Tools
    @abc.abstractmethod
    def getPrediction(self, x):
        pass


    @abc.abstractmethod
    def updateParameters(self, x_batch, z_batch, batch_size, lr):
        pass
    

    @abc.abstractmethod
    def resetWeights(self):
        pass


    @abc.abstractmethod
    def getWeights(self):
        pass
