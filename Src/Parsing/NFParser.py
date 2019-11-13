import numpy as np
import gc
import scipy.sparse as sp
import os



KINO_NUM = 17770
USR_NUM = 2649429 
RATE_NUM = 100480507
N_CHUNKS = 100


if __name__ == '__main__':

    chunk_size = RATE_NUM // N_CHUNKS
    print('Chunk size: %i' % chunk_size)
    dataset_path = '../../Dataset/netflix/netflix_dataset_index.npy'
    dataset_ind = np.load(dataset_path)

    chunk_range_start = 80
    chunk_range_end = 100
    chunk_range = range(chunk_range_start, chunk_range_end)
    kino_matrix = sp.dok_matrix((chunk_size * (chunk_range_end - chunk_range_start), USR_NUM + KINO_NUM), dtype=np.int8)

    for chunk_num in chunk_range:
        print('Processing chunk #%i' % chunk_num)
        for rating_iter in range(chunk_num * chunk_size, (chunk_num + 1) * chunk_size):
            kino_matrix[rating_iter - chunk_range_start * chunk_size, dataset_ind[rating_iter, 0]] = 1
            kino_matrix[rating_iter - chunk_range_start * chunk_size, dataset_ind[rating_iter, 1]] = 1
        
    print('Converting to CSR...')
    kino_matrix_csr = kino_matrix.tocsr()
    sp.save_npz('../../Dataset/netflix/chunks/netflix_dataset_oh_{}_{}.npz'.format(chunk_range_start, chunk_range_end), kino_matrix_csr)

    # last ratings
    #for rating_iter in range(N_CHUNKS * chunk_size, RATE_NUM):
    #    kino_matrix[rating_iter, dataset_ind[rating_iter, 0]] = 1
    #    kino_matrix[rating_iter, dataset_ind[rating_iter, 1]] = 1



