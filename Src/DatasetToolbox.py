import numpy as np



def shuffle(dataset, labels):
    dataset_size = labels.shape[0]
    p = np.random.permutation(dataset_size)

    dataset = dataset[p, :]
    labels = labels[p]

    return dataset, labels


def reduceByQuant(dataset, labels, quantizer=1):  # make its size divisible by batch_size*num_folds
    dataset_size = labels.shape[0]
    multiple_size = (dataset_size // quantizer) * quantizer

    dataset = dataset[0:multiple_size, :]
    labels = labels[0:multiple_size]
    new_dataset_size = multiple_size

    return dataset, labels, new_dataset_size


def normalize(dataset, num_folds=1):
    dataset_size = dataset.shape[0]
    num_features = dataset.shape[1]
    fold_size = dataset_size // num_folds

    for fold_iter in range(num_folds):
        for feature_iter in range(num_features):
            f_mean = np.mean(dataset[fold_iter*fold_size:(fold_iter + 1)*fold_size, 
                                                    feature_iter])
            f_std = np.std(dataset[fold_iter*fold_size:(fold_iter + 1)*fold_size, 
                                                feature_iter])
            f_max = np.max(dataset[fold_iter*fold_size:(fold_iter + 1)*fold_size, 
                                                feature_iter])
            if not f_std == 0:
                dataset[fold_iter*fold_size:(fold_iter + 1)*fold_size, feature_iter] -= f_mean
                dataset[fold_iter*fold_size:(fold_iter + 1)*fold_size, feature_iter] /= f_std
            elif not f_max == 0:
                dataset[fold_iter*fold_size:(fold_iter + 1)*fold_size, feature_iter] /= f_max

    return dataset


def makeFolds(dataset, labels, start_index, end_index, is_train):
    if is_train:
        data_fold = np.delete(dataset,
                              slice(start_index, end_index),
                              axis=0)
        label_fold = np.delete(labels,
                               slice(start_index, end_index),
                               axis=0)
    # is_val
    else:
        data_fold = dataset[start_index:end_index, :]
        label_fold = labels[start_index:end_index]

    return data_fold, label_fold 

