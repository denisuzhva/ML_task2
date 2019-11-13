import numpy as np
import time
import DatasetToolbox as dt
import Evaluator as eva
import Optimizer as opt



class Session:
    def __init__(self):
        pass


    ## perform cross-validation
    def crossValidation(self, model, 
                        dataset, labels,
                        num_features, num_samples,
                        epochs, epoch_quant,
                        batch_size,
                        num_folds,
                        learning_rate):

        metrics_tensor = np.zeros((num_folds,
                                   epoch_quant,
                                   1,
                                   2),
                                  dtype=float)  

        time_var = 0

        start_time = time.time()

        for fold_iter in range(num_folds):
            print('== Current validation fold: %d ==' % fold_iter)
            model.resetWeights()
            
            start_index = (num_samples // num_folds) * fold_iter
            end_index = (num_samples // num_folds) * (fold_iter+1)

            train_folds, train_labels = dt.makeFolds(dataset, labels, 
                                                     start_index, end_index, 
                                                     is_train=True)
            val_folds, val_labels = dt.makeFolds(dataset, labels, 
                                                 start_index, end_index, 
                                                 is_train=False)

            epoch_quant_iter = 0            
            for epoch_iter in range(epochs):

                opt.optimize(model, train_folds, train_labels, batch_size, learning_rate)

                if epoch_iter % (epochs // epoch_quant) == 0:

                    train_pred = model.getPrediction(train_folds)
                    val_pred = model.getPrediction(val_folds)

                    train_rmse = eva.rmseMetric(train_pred, train_labels)
                    val_rmse = eva.rmseMetric(val_pred, val_labels)

                    assert ~np.isnan(train_rmse)
                    assert ~np.isnan(val_rmse)

                    metrics_tensor[fold_iter][epoch_quant_iter][0][0] = train_rmse 
                    metrics_tensor[fold_iter][epoch_quant_iter][0][1] = val_rmse 

                    epoch_quant_iter  += 1

        end_time = time.time()
        time_var = end_time - start_time

        return metrics_tensor, time_var

