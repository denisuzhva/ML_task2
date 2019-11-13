import numpy as np
import os
import scipy.sparse as sp



N_CHUNKS = 5
CHUNK_SIZE = 1004805
KINO_NUM = 17770
USR_NUM = 2649429 


if __name__ == '__main__':

    chunks_path = '../../Dataset/netflix/chunks/'

    dataset_dok = sp.dok_matrix((0, USR_NUM + KINO_NUM), dtype=np.int8)
    dataset_csr = dataset_dok.tocsr()
    for chunk_iter, chunk_name in enumerate(os.listdir(chunks_path)):
        print(chunks_path + chunk_name)
        chunk_csr = sp.load_npz(chunks_path + chunk_name)
        print('Chunk %s loaded' % chunk_name)
        dataset_csr = sp.vstack([dataset_csr, chunk_csr])

    print('Saving dataset')
    sp.save_npz('../../Dataset/netflix/netflix_dataset_oh.npz', dataset_csr)
   