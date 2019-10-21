import numpy as np
import DatasetToolbox as dt



# update parameters of a model once (on a mini-batch)
def optimize(model, train_data, train_labels, batch_size, learning_rate):

    train_data, train_labels = dt.shuffle(train_data, train_labels)

    batches_per_fold = train_data.shape[0] // batch_size
    
    for batch_iter in range(batches_per_fold):
        train_dataset_batch = train_data[batch_iter*batch_size:(batch_iter+1)*batch_size, :]
        train_labels_batch = train_labels[batch_iter*batch_size:(batch_iter+1)*batch_size]

        model.updateParameters(train_dataset_batch, 
                               train_labels_batch,
                               batch_size, 
                               learning_rate)

