import numpy as np
import scipy.sparse as sp
import csv
import os



KINO_NUM = 193609
USR_NUM = 610 
RATE_NUM = 100836


if __name__ == '__main__':

    dataset_ind_path = '../../RawData/ml-latest-small/'

    dataset_ind = np.zeros((RATE_NUM, 2), dtype=np.int)
    target_data = np.zeros((RATE_NUM,), dtype=np.int8)

    with open(dataset_ind_path + 'ratings.csv') as csv_data:
        csv_lines = csv.reader(csv_data, delimiter=',')
        next(csv_data, None)
        for row_iter, row_val in enumerate(csv_lines):
            if row_iter % 100 == 0:
                print('Sample #%i' % row_iter)

            usr_id = int(row_val[0]) - 1
            kino_id = int(row_val[1]) - 1
            rating = int(float(row_val[2]))
    
            dataset_ind[row_iter, 0] = usr_id
            dataset_ind[row_iter, 1] = USR_NUM + kino_id
            target_data[row_iter] = rating

    np.save('../../Dataset/movielens/ratings_dataset_ind_movielens_index.npy', dataset_ind)
    #np.save('../../Dataset/movielens/target.npy', target_data)