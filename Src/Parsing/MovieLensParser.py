import numpy as np
import scipy.sparse as sp
import csv
import os



KINO_NUM = 193609
USR_NUM = 610 
RATE_NUM = 100836


if __name__ == '__main__':

    dataset_path = '../../RawData/ml-latest-small/'

    #kino_dict = {}

    kino_dataset = sp.dok_matrix((RATE_NUM, USR_NUM + KINO_NUM), dtype=np.int8)
    target_data = np.zeros((RATE_NUM,), dtype=np.int8)

    with open(dataset_path + 'ratings.csv') as csv_data:
        csv_lines = csv.reader(csv_data, delimiter=',')
        next(csv_data, None)
        for row_iter, row_val in enumerate(csv_lines):
            if row_iter % 100 == 0:
                print('Sample #%i' % row_iter)

            usr_id = int(row_val[0]) - 1
            kino_id = int(row_val[1]) - 1
            rating = int(float(row_val[2]))
    
            kino_dataset[row_iter, usr_id] = 1
            kino_dataset[row_iter, USR_NUM + kino_id] = 1
            target_data[row_iter] = rating

    kino_csr = kino_dataset.tocsr()
    
    sp.save_npz('../../Dataset/movielens/ratings_dataset_movielens.npz', kino_csr)
    np.save('../../Dataset/movielens/target.npy', target_data)