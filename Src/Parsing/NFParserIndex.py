import numpy as np
import os



KINO_NUM = 17770
USR_NUM = 2649429 
RATE_NUM = 100480507


if __name__ == '__main__':

    dataset_path = '../../RawData/training_set/'

    #kino_dataset = np.zeros((RATE_NUM, 2), dtype=np.int32)
    target_rating = np.zeros((RATE_NUM,), dtype=np.int8)

    rating_iter = 0
    for kino_iter, kino_id in enumerate(os.listdir(dataset_path)):
        with open(dataset_path + kino_id, 'r') as kino_file:
            if kino_iter % 100 == 0:
                print('Processing file #' + kino_id + '...')
            kino_content = kino_file.readlines()
            kino_content  = kino_content[1:]
            for usr in kino_content:
                usr_data = usr.split(",")
                #usr_id = int(usr_data[0])-1
                usr_rating = int(usr_data[1])
                #usr_day = yearToDay(usr_data[2], min_year)
                #kino_dataset[rating_iter, 0] = usr_id
                #kino_dataset[rating_iter, 1] = USR_NUM + kino_iter
                target_rating[rating_iter] = usr_rating
                rating_iter += 1

    #np.save('../../Dataset/netflix/netflix_dataset_index.npy', kino_dataset)
    np.save('../../Dataset/netflix/target.npy', target_rating)
