import numpy as np



def mseMetric(fx_batch, z_batch):
    batch_size = fx_batch.shape[0]
    metric = np.sum(np.square(z_batch - fx_batch)) / batch_size
    return metric


def rmseMetric(fx_batch, z_batch):
    batch_size = fx_batch.shape[0]
    metric = np.sum(np.square(z_batch - fx_batch)) / batch_size
    return np.sqrt(metric)


def r2Metric(fx_batch, z_batch):
    batch_size = fx_batch.shape[0]
    mse_metric = np.sum(np.square(z_batch - fx_batch)) / batch_size
    metric = 1 - mse_metric / (np.sum(np.square(z_batch - np.mean(z_batch))) / batch_size)
    return metric


# with regularization
def mseMetricReg(fx_batch, z_batch, weights, order=2):
    batch_size = fx_batch.shape[0]
    metric = np.sum(np.square(z_batch - fx_batch)) / batch_size
    metric = metric + np.linalg.norm(weights, order)
    return metric


def rmseMetricReg(fx_batch, z_batch, weights, order=2):
    batch_size = fx_batch.shape[0]
    metric = np.sum(np.square(z_batch - fx_batch)) / batch_size
    metric = metric + np.linalg.norm(weights, order)
    return np.sqrt(metric)


def r2MetricReg(fx_batch, z_batch, weights, order=2):
    batch_size = fx_batch.shape[0]
    mse_metric = np.sum(np.square(z_batch - fx_batch)) / batch_size
    mse_metric = mse_metric + np.linalg.norm(weights, order)
    metric = 1 - mse_metric / (np.sum(np.square(z_batch - np.mean(z_batch))) / batch_size)
    return metric


