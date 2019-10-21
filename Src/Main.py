import numpy as np
from LinearRegressor import LinearRegressor
from Session import Session
import DatasetToolbox as dt


NUM_FEATURES = 53   # number of features
LR = 1 * 1e-3   # learning rate constant
BATCH_LIST = [1600] # batches to test
#BATCH_LIST = [1600]
NUM_EPOCHS = 4000   # number of epochs
EPOCH_QUANTIZER = 100   # write metric values "EPOCH_QUANTIZER" times
NUM_FOLDS = 5   # number of folds
REG_GAMMA = 0.1 # gamma parameter (for regularization)


if __name__ == '__main__':
    ## Load a dataset and labels
    dataset = np.load('../Dataset/FV1_ds.npy')
    labels = np.load('../Dataset/FV1_l.npy')

    ####################################
    ## Process the dataset and labels ##

    # reduce the number of data samples so it could be divisible by max_batch_size*num_folds 
    reduce_quantizer = NUM_FOLDS * BATCH_LIST[-1]
    dataset, labels, dataset_size = dt.reduceByQuant(dataset, labels, reduce_quantizer)

    # normalize dataset
    dataset = dt.normalize(dataset, NUM_FOLDS)

    # shuffle data samples and labels
    dataset, labels = dt.shuffle(dataset, labels)

    ####################
    ## Create a model ##

    linear_regressor = LinearRegressor(NUM_FEATURES)

    #####################
    ## Train the model ##

    metrics_tensor = np.zeros((len(BATCH_LIST),  # write at each batch
                               NUM_FOLDS,  # at each fold
                               EPOCH_QUANTIZER,   # ...at each self._epoch_quantize_param's epoch
                               2,    # for all the metrics to write
                               2),   # for train and validation metric
                               dtype=np.float)  

    weight_tensor = np.zeros((len(BATCH_LIST),
                              NUM_FOLDS,
                              EPOCH_QUANTIZER,
                              NUM_FEATURES + 1),  # w1..wN, w0; N = 53
                              dtype=np.float) 

    time_tensor = np.zeros(len(BATCH_LIST), dtype=np.float)   # vector of time values (for each batch size)

    sess = Session()

    # check different batch sizes
    for batch_size_counter, batch_size in enumerate(BATCH_LIST, start=0):
        print('=== Current batch size: %d ===' % batch_size)      

        metrics_tensor[batch_size_counter], 
        weight_tensor[batch_size_counter], 
        time_tensor[batch_size_counter] = sess.crossValidation(linear_regressor, 
                                                               dataset, labels, 
                                                               NUM_EPOCHS, EPOCH_QUANTIZER, 
                                                               batch_size, 
                                                               NUM_FOLDS,
                                                               LR)

    ################
    ## Write data ##

    np.save('./TrainData/metrics.npy', metrics_tensor)
    np.save('./TrainData/weights.npy', weight_tensor)
    np.save('./TrainData/time.npy', time_tensor)
