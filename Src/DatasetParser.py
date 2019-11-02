import numpy as np
import scipy.sparse as sp
import os



KINO_NUM = 17770
USR_NUM = 2649429 


if __name__ == '__main__':

    dataset_path = '../RawData/training_set/'

    kino_dict = {}

    for kino_iter, kino_id in enumerate(os.listdir(dataset_path)):
        kino_users = {}
        
        with open(dataset_path + kino_id, 'r') as kino_file:
            if kino_iter % 100 == 0:
                print('Processing file #' + kino_id + '...')
            kino_content = kino_file.readlines()
            kino_content  = kino_content[1:]
            for usr in kino_content:
                usr_data = usr.split(",")
                usr_id = int(usr_data[0])-1
                usr_rating = int(usr_data[1])
                #usr_day = yearToDay(usr_data[2], min_year)
                kino_users[usr_id] = usr_rating

        kino_dict[kino_iter] = kino_users 

    kino_matrix = sp.dok_matrix((USR_NUM, KINO_NUM), dtype=np.int8)
    print('Constructing a DOK matrix ... ')
    for kino_id, usr_id_per_kino in kino_dict.items():
        print('Processing kino id ' + kino_id + '...')
        for usr_id in usr_id_per_kino:
            kino_matrix[usr_id, kino_id] = kino_dict[kino_id][usr_id]

    kino_matrix = kino_matrix.tocsr()

    sp.save_npz('../Dataset/sparse.npz', kino_matrix)
