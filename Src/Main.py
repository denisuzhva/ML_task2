import numpy as np
import scipy.sparse as sp
from FactorizationMachine import FactorizationMachine
from Session import Session
import DatasetToolbox as dt



DATASET = 'movielens_small'

if DATASET == 'netflix':
    NUM_FEATURES = 17770 + 2649429   # number of features
    NUM_SAMPLES = 100480507    # number of data samples
    DATASET_NAME = 'netflix_dataset_oh'  # name of the dataset
elif DATASET == 'movielens_small':
    NUM_FEATURES = 610 + 193609
    NUM_SAMPLES = 100836
    DATASET_NAME = 'movielens_small_dataset_oh'

NUM_FACTORS = 4 # number of factors (aka k)
LR = 1 * 1e-3   # learning rate constant
BATCH_LIST = [2048] # batches to test
NUM_EPOCHS = 100  # number of epochs
EPOCH_QUANTIZER = 100   # write metric values "EPOCH_QUANTIZER" times
NUM_FOLDS = 5   # number of folds
REG_GAMMA = 0.1 # gamma parameter (for regularization)


if __name__ == '__main__':
    ## Load a dataset and labels
    dataset = sp.load_npz('../Dataset/{}/{}.npz'.format(DATASET, DATASET_NAME))
    labels = np.load('../Dataset/{}/target.npy'.format(DATASET))

    ####################################
    ## Process the dataset and labels ##

    # reduce the number of data samples so it could be divisible by max_batch_size*num_folds 
    reduce_quantizer = NUM_FOLDS * BATCH_LIST[-1]
    dataset, labels, NUM_SAMPLES = dt.reduceByQuant(dataset, labels, NUM_SAMPLES, reduce_quantizer)

    # normalize dataset
    #dataset = dt.normalize(dataset, NUM_FOLDS)

    # shuffle data samples and labels
    dataset, labels = dt.shuffle(dataset, labels, NUM_SAMPLES)

    ####################
    ## Create a model ##

    factor_machine = FactorizationMachine(NUM_FEATURES, NUM_FACTORS)

    #####################
    ## Train the model ##

    metrics_tensor = np.zeros((len(BATCH_LIST),  # write at each batch
                               NUM_FOLDS,  # at each fold
                               EPOCH_QUANTIZER,   # ...at each self._epoch_quantize_param's epoch
                               1,    # for all the metrics to write
                               2),   # for train and validation metric
                               dtype=np.float)  

    time_tensor = np.zeros(len(BATCH_LIST), dtype=np.float)   # vector of time values (for each batch size)

    sess = Session()

    # check different batch sizes
    for batch_size_counter, batch_size in enumerate(BATCH_LIST, start=0):
        print('=== Current batch size: %d ===' % batch_size)      

        metrics_tensor[batch_size_counter], time_tensor[batch_size_counter] = sess.crossValidation(factor_machine, 
                                                                                                   dataset, labels, 
                                                                                                   NUM_FEATURES, NUM_SAMPLES,
                                                                                                   NUM_EPOCHS, EPOCH_QUANTIZER, 
                                                                                                   batch_size, 
                                                                                                   NUM_FOLDS,
                                                                                                   LR)

    ################
    ## Write data ##

    np.save('../TrainData/{}/metrics.npy'.format(DATASET), metrics_tensor)
    np.save('../TrainData/{}/time.npy'.format(DATASET), time_tensor)
